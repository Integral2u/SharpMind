using System.Text;

namespace SharpMind.Model.Audio;

/// <summary>Sample layout used when writing a WAV file.</summary>
public enum WavSampleFormat
{
    /// <summary>16-bit signed PCM, the format every player and the ASR smoke test accept.</summary>
    Pcm16,
    /// <summary>32-bit IEEE float, lossless for model output and reference comparisons.</summary>
    Float32,
}

/// <summary>Decoded mono audio and its source sample rate.</summary>
public sealed record WavData(float[] Samples, int SampleRate, int Channels);

/// <summary>
/// Minimal RIFF/WAVE reader and writer. Reads PCM 8/16/24/32 and IEEE float 32/64,
/// WAVE_FORMAT_EXTENSIBLE included, and downmixes to mono. Writing supports 16-bit PCM
/// and 32-bit float.
/// </summary>
public static class WavFile
{
    private const ushort FormatPcm = 1;
    private const ushort FormatFloat = 3;
    private const ushort FormatExtensible = 0xFFFE;

    /// <summary>Reads a WAV file and returns the mono mix at its original sample rate.</summary>
    public static WavData Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);

        if (new string(reader.ReadChars(4)) != "RIFF")
            throw new InvalidDataException("Not a RIFF file.");
        _ = reader.ReadUInt32();
        if (new string(reader.ReadChars(4)) != "WAVE")
            throw new InvalidDataException("Not a WAVE file.");

        ushort format = 0;
        ushort channels = 0;
        uint sampleRate = 0;
        ushort blockAlign = 0;
        ushort bitsPerSample = 0;
        long dataOffset = -1;
        int dataSize = 0;

        long length = stream.Length;
        while (stream.Position + 8 <= length)
        {
            string id = new(reader.ReadChars(4));
            uint size = reader.ReadUInt32();
            long body = stream.Position;

            if (id == "fmt ")
            {
                format = reader.ReadUInt16();
                channels = reader.ReadUInt16();
                sampleRate = reader.ReadUInt32();
                _ = reader.ReadUInt32(); // byte rate
                blockAlign = reader.ReadUInt16();
                bitsPerSample = reader.ReadUInt16();

                if (format == FormatExtensible && size >= 40)
                {
                    _ = reader.ReadUInt16(); // cbSize
                    _ = reader.ReadUInt16(); // valid bits
                    _ = reader.ReadUInt32(); // channel mask
                    format = reader.ReadUInt16(); // first field of the sub-format GUID
                }
            }
            else if (id == "data")
            {
                dataOffset = body;
                dataSize = checked((int)Math.Min(size, (uint)(length - body)));
            }

            stream.Position = body + size + (size & 1);
        }

        if (dataOffset < 0) throw new InvalidDataException("WAV file has no data chunk.");
        if (channels == 0) throw new InvalidDataException("WAV file has no channel count.");
        if (bitsPerSample == 0) throw new InvalidDataException("WAV file has no sample width.");
        if (format != FormatPcm && format != FormatFloat)
            throw new NotSupportedException($"WAV format tag {format} is not supported.");

        int bytesPerSample = bitsPerSample / 8;
        int frameSize = bytesPerSample * channels;
        if (frameSize == 0) throw new InvalidDataException("WAV file has an invalid block size.");
        if (blockAlign != 0 && frameSize != blockAlign)
            throw new InvalidDataException($"WAV block align {blockAlign} does not match {frameSize}.");

        stream.Position = dataOffset;
        byte[] raw = reader.ReadBytes(dataSize);
        int frames = raw.Length / frameSize;
        var mono = new float[frames];

        // Sum the channels and divide once; this keeps a silent stereo pair exactly zero.
        for (int frame = 0; frame < frames; frame++)
        {
            int offset = frame * frameSize;
            double sum = 0;
            for (int channel = 0; channel < channels; channel++)
                sum += DecodeSample(raw, offset + channel * bytesPerSample, format, bitsPerSample);
            mono[frame] = (float)(sum / channels);
        }

        return new WavData(mono, checked((int)sampleRate), channels);
    }

    /// <summary>Reads a WAV file as mono, optionally resampling to <paramref name="targetSampleRate"/>.</summary>
    public static float[] ReadMono(string path, int? targetSampleRate = null)
    {
        WavData data = Read(path);
        if (targetSampleRate is not { } target || target == data.SampleRate) return data.Samples;
        return Resample(data.Samples, data.SampleRate, target);
    }

    /// <summary>
    /// Writes mono (or interleaved multichannel) samples. Values outside [-1, 1] are clamped
    /// for PCM16 and preserved for Float32.
    /// </summary>
    public static void Write(
        string path,
        ReadOnlySpan<float> samples,
        int sampleRate,
        int channels = 1,
        WavSampleFormat format = WavSampleFormat.Pcm16)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        if (samples.Length % channels != 0)
            throw new ArgumentException("Sample count must be a multiple of the channel count.", nameof(samples));

        ushort bitsPerSample = format == WavSampleFormat.Pcm16 ? (ushort)16 : (ushort)32;
        ushort formatTag = format == WavSampleFormat.Pcm16 ? FormatPcm : FormatFloat;
        int bytesPerSample = bitsPerSample / 8;
        int dataSize = samples.Length * bytesPerSample;

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize);
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16);
        writer.Write(formatTag);
        writer.Write((ushort)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * bytesPerSample);
        writer.Write((ushort)(channels * bytesPerSample));
        writer.Write(bitsPerSample);
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);

        foreach (float sample in samples)
        {
            if (format == WavSampleFormat.Pcm16)
            {
                float clamped = Math.Clamp(sample, -1f, 1f);
                writer.Write((short)Math.Round(clamped * short.MaxValue, MidpointRounding.AwayFromZero));
            }
            else
            {
                writer.Write(sample);
            }
        }
    }

    /// <summary>Linear-interpolation resampler, adequate for 16 kHz speech input.</summary>
    public static float[] Resample(ReadOnlySpan<float> samples, int sourceRate, int targetRate)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sourceRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetRate);
        if (sourceRate == targetRate) return samples.ToArray();
        if (samples.Length == 0) return [];
        if (samples.Length == 1)
        {
            int count = Math.Max(1, (int)Math.Round((double)targetRate / sourceRate));
            var single = new float[count];
            Array.Fill(single, samples[0]);
            return single;
        }

        int outputLength = Math.Max(1, (int)Math.Round(samples.Length * (double)targetRate / sourceRate));
        var output = new float[outputLength];
        double step = (double)(samples.Length - 1) / Math.Max(1, outputLength - 1);
        for (int i = 0; i < outputLength; i++)
        {
            double position = i * step;
            int left = (int)position;
            int right = Math.Min(left + 1, samples.Length - 1);
            double fraction = position - left;
            output[i] = (float)(samples[left] + (samples[right] - samples[left]) * fraction);
        }

        return output;
    }

    private static double DecodeSample(byte[] raw, int offset, ushort format, ushort bitsPerSample)
    {
        if (format == FormatFloat)
        {
            return bitsPerSample switch
            {
                32 => BitConverter.Int32BitsToSingle(
                    raw[offset] | raw[offset + 1] << 8 | raw[offset + 2] << 16 | raw[offset + 3] << 24),
                64 => BitConverter.Int64BitsToDouble(
                    (long)raw[offset] | (long)raw[offset + 1] << 8 | (long)raw[offset + 2] << 16 |
                    (long)raw[offset + 3] << 24 | (long)raw[offset + 4] << 32 | (long)raw[offset + 5] << 40 |
                    (long)raw[offset + 6] << 48 | (long)raw[offset + 7] << 56),
                _ => throw new NotSupportedException($"Float WAV with {bitsPerSample} bits is not supported."),
            };
        }

        return bitsPerSample switch
        {
            8 => (raw[offset] - 128) / 128.0,
            16 => (short)(raw[offset] | raw[offset + 1] << 8) / 32768.0,
            24 => ((raw[offset] | raw[offset + 1] << 8 | raw[offset + 2] << 16)
                   << 8 >> 8) / 8388608.0,
            32 => (raw[offset] | raw[offset + 1] << 8 | raw[offset + 2] << 16 | raw[offset + 3] << 24) /
                  2147483648.0,
            _ => throw new NotSupportedException($"PCM WAV with {bitsPerSample} bits is not supported."),
        };
    }
}
