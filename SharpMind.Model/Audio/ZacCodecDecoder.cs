using System.Buffers;
using System.Numerics;
using SharpMind.Model.Format;

namespace SharpMind.Model.Audio;

/// <summary>
/// ZONOS3 Mini's ZAC codec decoder ("mossdual_istft"): semantic RVQ + causal ConvNeXt
/// head, residual RVQ, five windowed-causal MOSS transformer stages with patch
/// upsampling, the multi-resolution iSTFT head and the exact 44160 -> 44100 Hz
/// resampler. The mmproj weights are read as floats; the codec is far below real time
/// and TTS output is produced offline.
/// </summary>
public sealed class ZacCodecDecoder : IDisposable
{
    public const int SampleRate = 44100;
    public const int SamplesPerFrame = 2940;
    public const int Codebooks = 10;
    public const int CodebookSize = 4096;

    private const int LatentDim = 1024;
    private const int SemDim = 768;
    private const int HeadDim = 768;
    private const int HeadOutDim = 2948;
    private const int Hop = 184;
    private const int HeadFramesPerFrame = 16;
    private const int RsOrig = 736;
    private const int RsNew = 735;
    private const int FirTaps = 127;
    private const float LnEps = 1e-5f;
    private const float ConvEps = 1e-6f;
    private const int AttentionHeadDim = 64;

    private static readonly int[] Nfft = [736, 2208];
    private static readonly int[] StageContext = [150, 300, 480, 480, 480];

    private readonly float[] _semTable;
    private readonly float[] _semBias;
    private readonly ConvNext[] _semBlocks;
    private readonly float[] _semOutW;
    private readonly float[] _semOutB;
    private readonly float[] _resTable;
    private readonly float[] _resBias;
    private readonly Stage[] _stages;
    private readonly ConvNext[] _headBlocks;
    private readonly float[] _headNormW;
    private readonly float[] _headNormB;
    private readonly float[] _headOutW;
    private readonly float[] _headOutB;
    private readonly float[] _mixer;
    private readonly float[] _resample;
    private readonly BluesteinInverse[] _ifft;
    private readonly float[][] _window;
    private bool _disposed;

    public ZacCodecDecoder(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        _semTable = reader.ReadFloat32("a.gen.wav.sem.table");
        _semBias = reader.ReadFloat32("a.gen.wav.sem.bias");
        _semBlocks = LoadConvNexts(reader, "a.gen.wav.sem.", 4);
        _semOutW = reader.ReadFloat32("a.gen.wav.sem.out.weight");
        _semOutB = reader.ReadFloat32("a.gen.wav.sem.out.bias");
        _resTable = reader.ReadFloat32("a.gen.wav.res.table");
        _resBias = reader.ReadFloat32("a.gen.wav.res.bias");

        _stages = new Stage[5];
        for (int s = 0; s < _stages.Length; s++)
            _stages[s] = LoadStage(reader, s);

        _headBlocks = LoadConvNexts(reader, "a.gen.wav.head.", 8);
        _headNormW = reader.ReadFloat32("a.gen.wav.head.norm.weight");
        _headNormB = reader.ReadFloat32("a.gen.wav.head.norm.bias");
        _headOutW = reader.ReadFloat32("a.gen.wav.head.out.weight");
        _headOutB = reader.ReadFloat32("a.gen.wav.head.out.bias");

        _mixer = reader.ReadFloat32("a.gen.wav.mixer.weight");
        _resample = reader.ReadFloat32("a.gen.wav.resample.weight");

        _ifft = [new BluesteinInverse(Nfft[0]), new BluesteinInverse(Nfft[1])];
        _window = [Hann(Nfft[0]), Hann(Nfft[1])];
    }

    /// <summary>True when the projector carries the ZAC codec decoder and the codebook head.</summary>
    public static bool IsZacProjector(GgufTensorReader reader)
    {
        try
        {
            return reader.TryGetInfo("a.gen.code.head.weight", out _) &&
                   reader.TryGetInfo("a.gen.wav.sem.table", out _) &&
                   reader.TryGetInfo("a.gen.wav.head.out.weight", out _);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Decodes frame-major codes (<c>[frame * 10 + codebook]</c>, 0..4095) into 44.1 kHz
    /// mono samples. Frames containing EOA/pad are masked by the caller.
    /// </summary>
    public float[] Decode(int[] codes, int frameCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(codes);
        if (frameCount <= 0) return [];
        if (codes.Length < frameCount * Codebooks)
            throw new ArgumentException("Code buffer is shorter than the frame count.", nameof(codes));

        // 1. Semantic codebook + causal ConvNeXt head -> latent [T, 1024].
        var latent = Semantics(codes, frameCount);

        // 2. Residual codebooks and the summed residual bias.
        for (int t = 0; t < frameCount; t++)
        {
            int rowBase = t * LatentDim;
            for (int d = 0; d < LatentDim; d++)
                latent[rowBase + d] += _resBias[d];
            for (int c = 1; c < Codebooks; c++)
            {
                int row = (c - 1) * CodebookSize + codes[t * Codebooks + c];
                int tableBase = row * LatentDim;
                for (int d = 0; d < LatentDim; d++)
                    latent[rowBase + d] += _resTable[tableBase + d];
            }
        }

        // 3. MOSS stages with patch upsampling.
        EnsureRope(frameCount * HeadFramesPerFrame);
        int framesPerColumn = 1;
        int width = LatentDim;
        float[] x = latent;
        for (int s = 0; s < _stages.Length; s++)
        {
            Stage stage = _stages[s];
            int rows = frameCount * framesPerColumn;
            if (stage.InDim != width)
                throw new InvalidDataException(
                    $"ZAC stage {s} expects {stage.InDim} latent channels, got {width}.");
            var projected = new float[(long)rows * stage.OutDim];
            Lfm3AudioCodeGenerator.MatMulBatch(x, stage.InW, width, stage.OutDim, projected, rows);
            x = projected;
            width = stage.OutDim;

            foreach (MossLayer layer in stage.Layers)
                x = MossForward(x, rows, width, layer, stage.Context);

            if (stage.OutW is not null)
            {
                var outProjected = new float[(long)rows * width];
                Lfm3AudioCodeGenerator.MatMulBatch(x, stage.OutW, width, width, outProjected, rows);
                x = outProjected;
            }

            if (s + 1 < _stages.Length)
            {
                x = PatchUp(x, rows, width);
                width /= 2;
                framesPerColumn *= 2;
            }
        }

        // 4. Head ConvNeXt trunk, then the iSTFT head.
        int columns = frameCount * HeadFramesPerFrame;
        if (frameCount * framesPerColumn != columns)
            throw new InvalidDataException("ZAC stage upsampling did not reach 16 columns per frame.");

        foreach (ConvNext block in _headBlocks)
            x = ConvNextForward(x, columns, HeadDim, block);
        x = LayerNorm(x, columns, HeadDim, _headNormW, _headNormB, ConvEps, copy: false);

        return Synthesize(x, columns);
    }

    private float[] Semantics(int[] codes, int frameCount)
    {
        var x = new float[frameCount * SemDim];
        for (int t = 0; t < frameCount; t++)
        {
            int rowBase = t * SemDim;
            int tableBase = codes[t * Codebooks] * SemDim;
            for (int d = 0; d < SemDim; d++)
                x[rowBase + d] = _semTable[tableBase + d] + _semBias[d];
        }

        foreach (ConvNext block in _semBlocks)
            x = ConvNextForward(x, frameCount, SemDim, block);

        var latent = new float[frameCount * LatentDim];
        Lfm3AudioCodeGenerator.MatMulBatch(x, _semOutW, SemDim, LatentDim, latent, frameCount);
        for (int t = 0; t < frameCount; t++)
        {
            int rowBase = t * LatentDim;
            for (int d = 0; d < LatentDim; d++)
                latent[rowBase + d] += _semOutB[d];
        }
        return latent;
    }

    private static float[] ConvNextForward(float[] x, int rows, int channels, ConvNext block)
    {
        int kernel = block.Kernel;
        var h = new float[x.Length];

        // Causal depthwise conv (kernel 7) with zero history.
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * channels;
            for (int c = 0; c < channels; c++)
            {
                float acc = 0f;
                for (int k = 0; k < kernel; k++)
                {
                    int src = t - (kernel - 1) + k;
                    if (src >= 0) acc += block.DwW[k * channels + c] * x[src * channels + c];
                }
                h[rowBase + c] = acc + block.DwB[c];
            }
        }

        h = LayerNorm(h, rows, channels, block.NormW, block.NormB, ConvEps, copy: false);

        var wide = new float[(long)rows * block.Pw1Out];
        Lfm3AudioCodeGenerator.MatMulBatch(h, block.Pw1W, channels, block.Pw1Out, wide, rows);
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * block.Pw1Out;
            for (int o = 0; o < block.Pw1Out; o++)
                wide[rowBase + o] = GeluErf(wide[rowBase + o] + block.Pw1B[o]);
        }

        var y = new float[x.Length];
        Lfm3AudioCodeGenerator.MatMulBatch(wide, block.Pw2W, block.Pw1Out, channels, y, rows);
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * channels;
            for (int c = 0; c < channels; c++)
                y[rowBase + c] = x[rowBase + c] + y[rowBase + c] + block.Pw2B[c];
        }
        return y;
    }

    private float[] MossForward(float[] x, int rows, int dim, MossLayer layer, int context)
    {
        int heads = dim / AttentionHeadDim;
        var h = LayerNorm(x, rows, dim, layer.Norm1W, layer.Norm1B, LnEps, copy: true);

        var qkv = new float[(long)rows * (3 * dim)];
        Lfm3AudioCodeGenerator.MatMulBatch(h, layer.QkvW, dim, 3 * dim, qkv, rows);

        Rope(qkv, rows, dim, heads, qOffset: 0);
        Rope(qkv, rows, dim, heads, qOffset: dim);

        var attention = new float[(long)rows * dim];
        Attention(qkv, attention, rows, dim, heads, context);

        var projected = new float[(long)rows * dim];
        Lfm3AudioCodeGenerator.MatMulBatch(attention, layer.WoW, dim, dim, projected, rows);
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * dim;
            for (int c = 0; c < dim; c++)
                x[rowBase + c] += projected[rowBase + c] * layer.AttnScale[c];
        }

        h = LayerNorm(x, rows, dim, layer.Norm2W, layer.Norm2B, LnEps, copy: true);
        var ffn = new float[(long)rows * layer.FfnDim];
        Lfm3AudioCodeGenerator.MatMulBatch(h, layer.Ffn1W, dim, layer.FfnDim, ffn, rows);
        for (int i = 0; i < ffn.Length; i++) ffn[i] = GeluErf(ffn[i]);
        var ffnOut = new float[(long)rows * dim];
        Lfm3AudioCodeGenerator.MatMulBatch(ffn, layer.Ffn2W, layer.FfnDim, dim, ffnOut, rows);
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * dim;
            for (int c = 0; c < dim; c++)
                x[rowBase + c] += ffnOut[rowBase + c] * layer.FfnScale[c];
        }
        return x;
    }

    private void Rope(float[] qkv, int rows, int dim, int heads, int qOffset)
    {
        int width = AttentionHeadDim / 2;
        var cos = _ropeCos!;
        var sin = _ropeSin!;
        for (int t = 0; t < rows; t++)
        {
            int table = t * width;
            for (int h = 0; h < heads; h++)
            {
                int at = t * (3 * dim) + qOffset + h * AttentionHeadDim;
                for (int p = 0; p < width; p++)
                {
                    int i = at + 2 * p;
                    float a = qkv[i];
                    float b = qkv[i + 1];
                    float c = cos[table + p];
                    float s = sin[table + p];
                    qkv[i] = a * c - b * s;
                    qkv[i + 1] = a * s + b * c;
                }
            }
        }
    }

    private static void Attention(
        float[] qkv, float[] output, int rows, int dim, int heads, int context)
    {
        int width = 3 * dim;
        float scale = 1f / MathF.Sqrt(AttentionHeadDim);
        var pool = ArrayPool<float>.Shared;

        Parallel.For(0, rows, () => pool.Rent(context), (t, _, scores) =>
        {
            int lo = Math.Max(0, t - context + 1);
            for (int h = 0; h < heads; h++)
            {
                int qBase = t * width + h * AttentionHeadDim;
                int kBaseOff = dim + h * AttentionHeadDim;
                int vBaseOff = 2 * dim + h * AttentionHeadDim;

                float max = float.NegativeInfinity;
                int n = t - lo + 1;
                for (int j = 0; j < n; j++)
                {
                    int kBase = (lo + j) * width + kBaseOff;
                    float dot = 0f;
                    for (int d = 0; d < AttentionHeadDim; d++)
                        dot += qkv[qBase + d] * qkv[kBase + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > max) max = dot;
                }

                float sum = 0f;
                for (int j = 0; j < n; j++)
                {
                    float e = MathF.Exp(scores[j] - max);
                    scores[j] = e;
                    sum += e;
                }
                float inv = 1f / sum;

                int outBase = t * dim + h * AttentionHeadDim;
                for (int j = 0; j < n; j++)
                {
                    float w = scores[j] * inv;
                    int vBase = (lo + j) * width + vBaseOff;
                    for (int d = 0; d < AttentionHeadDim; d++)
                        output[outBase + d] += w * qkv[vBase + d];
                }
            }
            return scores;
        }, scores => pool.Return(scores));
    }

    private static float[] PatchUp(float[] x, int rows, int channels)
    {
        int outChannels = channels / 2;
        var y = new float[2 * rows * outChannels];
        for (int t = 0; t < rows; t++)
        {
            int srcBase = t * channels;
            for (int ch = 0; ch < outChannels; ch++)
            {
                y[(2 * t) * outChannels + ch] = x[srcBase + 2 * ch];
                y[(2 * t + 1) * outChannels + ch] = x[srcBase + 2 * ch + 1];
            }
        }
        return y;
    }

    private float[] Synthesize(float[] trunk, int columns)
    {
        EnsureScratch();
        int hopLength = Hop;
        long length = (long)columns * hopLength;
        var ola = new float[Nfft.Length][];
        var env = new float[Nfft.Length][];
        for (int i = 0; i < Nfft.Length; i++)
        {
            long size = (long)(columns - 1) * hopLength + Nfft[i];
            ola[i] = new float[size];
            env[i] = new float[size];
        }

        var spectrum = new Complex[Nfft[1]];
        var frame = new double[Nfft[1]];
        const int chunk = 512;
        var slice = new float[(long)chunk * HeadDim];
        for (int start = 0; start < columns; start += chunk)
        {
            int n = Math.Min(chunk, columns - start);
            var spec = new float[(long)n * HeadOutDim];
            Array.Copy(trunk, (long)start * HeadDim, slice, 0, (long)n * HeadDim);
            Lfm3AudioCodeGenerator.MatMulBatch(slice, _headOutW, HeadDim, HeadOutDim, spec, n);
            for (int c = 0; c < n; c++)
            {
                int rowBase = c * HeadOutDim;
                for (int r = 0; r < HeadOutDim; r++)
                    spec[rowBase + r] += _headOutB[r];
                AddColumn(spec, rowBase, start + c, ola, env);
            }
        }

        // The mixer is a centered FIR over both branches; outside [0, L) it reads zeros.
        var mixed = new float[length];
        int half = (FirTaps - 1) / 2;
        for (int i = 0; i < Nfft.Length; i++)
        {
            int pad = Nfft[i] / 2;
            var samples = new float[length];
            for (int m = 0; m < length; m++)
            {
                long at = (long)m + pad;
                samples[m] = env[i][at] > 1e-11f ? ola[i][at] / env[i][at] : 0f;
            }
            for (int m = 0; m < length; m++)
            {
                float acc = 0f;
                for (int k = 0; k < FirTaps; k++)
                {
                    int idx = m + k - half;
                    if ((uint)idx < (uint)length)
                        acc += _mixer[i * FirTaps + k] * samples[idx];
                }
                mixed[m] += acc;
            }
        }

        return Resample(mixed);
    }

    private void AddColumn(float[] spec, int rowBase, int columnIndex, float[][] ola, float[][] env)
    {
        var window = _window;
        long at = (long)columnIndex * Hop;
        int[] offsets = [0, Nfft[0] + 2];
        for (int i = 0; i < Nfft.Length; i++)
        {
            int n = Nfft[i];
            int bins = n / 2 + 1;
            int block = rowBase + offsets[i];
            double maxLogMag = Math.Log(100.0);
            Complex[] spectrum = _spectrumCache![i];
            spectrum[0] = new Complex(Mag(spec[block], maxLogMag) * Math.Cos(spec[block + bins]), 0);
            for (int k = 1; k < n / 2; k++)
            {
                double a = Mag(spec[block + k], maxLogMag);
                double phase = spec[block + bins + k];
                spectrum[k] = new Complex(a * Math.Cos(phase), a * Math.Sin(phase));
            }
            spectrum[n / 2] = new Complex(
                Mag(spec[block + n / 2], maxLogMag) * Math.Cos(spec[block + bins + n / 2]), 0);
            for (int k = 1; k < n / 2; k++)
                spectrum[n - k] = Complex.Conjugate(spectrum[k]);

            double[] frame = _frameCache![i];
            _ifft[i].Inverse(spectrum, frame);
            float[] win = window[i];
            long baseIndex = at;
            float[] o = ola[i];
            float[] e = env[i];
            for (int j = 0; j < n; j++)
            {
                o[baseIndex + j] += win[j] * (float)frame[j];
                e[baseIndex + j] += win[j] * win[j];
            }
        }
    }

    private static double Mag(float logMag, double maxLogMag) => Math.Exp(Math.Min(logMag, maxLogMag));

    private float[] Resample(float[] input)
    {
        int kernelLength = 2 * 17 + RsOrig; // 770
        int width = (kernelLength - RsOrig) / 2; // 17
        long total = ((long)RsNew * input.Length + RsOrig - 1) / RsOrig;
        var output = new float[total];
        long produced = 0;
        for (long block = 0; produced < total; block++)
        {
            long x0 = block * RsOrig - width;
            int emit = (int)Math.Min(RsNew, total - produced);
            for (int i = 0; i < emit; i++)
            {
                int kernelBase = i * kernelLength;
                double acc = 0;
                for (int k = 0; k < kernelLength; k++)
                {
                    long j = x0 + k;
                    if ((ulong)j < (ulong)input.Length)
                        acc += _resample[kernelBase + k] * input[j];
                }
                output[produced + i] = (float)acc;
            }
            produced += emit;
        }
        return output;
    }

    private Complex[][]? _spectrumCache;
    private double[][]? _frameCache;
    private float[]? _ropeCos;
    private float[]? _ropeSin;

    private void EnsureRope(int rows)
    {
        if (_ropeCos is not null && _ropeCos.Length >= rows * (AttentionHeadDim / 2)) return;
        int width = AttentionHeadDim / 2;
        _ropeCos = new float[rows * width];
        _ropeSin = new float[rows * width];
        for (int t = 0; t < rows; t++)
        {
            for (int p = 0; p < width; p++)
            {
                double angle = t * Math.Pow(10000.0, -2.0 * p / AttentionHeadDim);
                _ropeCos[t * width + p] = (float)Math.Cos(angle);
                _ropeSin[t * width + p] = (float)Math.Sin(angle);
            }
        }
    }

    /// <summary>Pre-allocates the per-branch iSTFT scratch (call once per Decode).</summary>
    private void EnsureScratch()
    {
        if (_spectrumCache is not null) return;
        _spectrumCache = [new Complex[Nfft[0]], new Complex[Nfft[1]]];
        _frameCache = [new double[Nfft[0]], new double[Nfft[1]]];
    }

    private static float[] LayerNorm(
        float[] x, int rows, int channels, float[] weight, float[] bias, float eps, bool copy)
    {
        var y = copy ? new float[x.Length] : x;
        for (int t = 0; t < rows; t++)
        {
            int rowBase = t * channels;
            double sum = 0, sumSq = 0;
            for (int c = 0; c < channels; c++)
            {
                double v = x[rowBase + c];
                sum += v;
                sumSq += v * v;
            }
            double mean = sum / channels;
            double variance = sumSq / channels - mean * mean;
            float inv = (float)(1.0 / Math.Sqrt(variance + eps));
            for (int c = 0; c < channels; c++)
                y[rowBase + c] = (float)((x[rowBase + c] - mean) * inv) * weight[c] + bias[c];
        }
        return y;
    }

    private static float GeluErf(float x)
    {
        double z = x * 0.7071067811865476;
        double az = Math.Abs(z);
        double t = 1.0 / (1.0 + 0.3275911 * az);
        double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        double erf = 1.0 - poly * Math.Exp(-az * az);
        if (z < 0) erf = -erf;
        return (float)(0.5 * x * (1.0 + erf));
    }

    private static float[] Hann(int n)
    {
        var window = new float[n];
        for (int i = 0; i < n; i++)
            window[i] = (float)(0.5 * (1.0 - Math.Cos(2.0 * Math.PI * i / n)));
        return window;
    }

    private static ConvNext[] LoadConvNexts(GgufTensorReader reader, string prefix, int count)
    {
        var blocks = new ConvNext[count];
        for (int i = 0; i < count; i++)
        {
            string p = prefix + i + ".";
            float[] dw = reader.ReadFloat32(p + "dw.weight");
            float[] pw1 = reader.ReadFloat32(p + "pw1.weight");
            reader.TryGetInfo(p + "dw.bias", out TensorInfo dwInfo);
            reader.TryGetInfo(p + "pw1.bias", out TensorInfo pw1Info);
            int channels = dwInfo.Shape[0];
            int pw1Out = pw1Info.Shape[0];
            blocks[i] = new ConvNext
            {
                DwW = dw,
                DwB = reader.ReadFloat32(p + "dw.bias"),
                NormW = reader.ReadFloat32(p + "norm.weight"),
                NormB = reader.ReadFloat32(p + "norm.bias"),
                Pw1W = pw1,
                Pw1B = reader.ReadFloat32(p + "pw1.bias"),
                Pw2W = reader.ReadFloat32(p + "pw2.weight"),
                Pw2B = reader.ReadFloat32(p + "pw2.bias"),
                Kernel = dw.Length / Math.Max(1, channels),
                Pw1Out = pw1Out,
            };
        }
        return blocks;
    }

    private static Stage LoadStage(GgufTensorReader reader, int index)
    {
        string p = $"a.gen.wav.stage.{index}.";
        reader.TryGetInfo(p + "in.weight", out TensorInfo inInfo);
        int inDim = inInfo.Shape[0];
        int outDim = inInfo.Shape[1];
        float[]? outW = reader.TryGetInfo(p + "out.weight", out _)
            ? reader.ReadFloat32(p + "out.weight")
            : null;

        var layers = new List<MossLayer>();
        for (int l = 0; ; l++)
        {
            string b = p + "blk." + l + ".";
            if (!reader.TryGetInfo(b + "qkv.weight", out _)) break;
            reader.TryGetInfo(b + "norm1.weight", out TensorInfo n1);
            reader.TryGetInfo(b + "ffn1.weight", out TensorInfo f1);
            int dim = n1.Shape[0];
            int ffnDim = f1.Shape[1];
            layers.Add(new MossLayer
            {
                Norm1W = reader.ReadFloat32(b + "norm1.weight"),
                Norm1B = reader.ReadFloat32(b + "norm1.bias"),
                QkvW = reader.ReadFloat32(b + "qkv.weight"),
                WoW = reader.ReadFloat32(b + "wo.weight"),
                AttnScale = reader.ReadFloat32(b + "attn_scale"),
                Norm2W = reader.ReadFloat32(b + "norm2.weight"),
                Norm2B = reader.ReadFloat32(b + "norm2.bias"),
                Ffn1W = reader.ReadFloat32(b + "ffn1.weight"),
                Ffn2W = reader.ReadFloat32(b + "ffn2.weight"),
                FfnDim = ffnDim,
                FfnScale = reader.ReadFloat32(b + "ffn_scale"),
            });
        }

        return new Stage
        {
            InW = reader.ReadFloat32(p + "in.weight"),
            InDim = inDim,
            OutDim = outDim,
            OutW = outW,
            Layers = [.. layers],
            Context = StageContext[index],
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }

    private sealed class ConvNext
    {
        public required float[] DwW;
        public required float[] DwB;
        public required float[] NormW;
        public required float[] NormB;
        public required float[] Pw1W;
        public required float[] Pw1B;
        public required float[] Pw2W;
        public required float[] Pw2B;
        public required int Kernel;
        public required int Pw1Out;
    }

    private sealed class MossLayer
    {
        public required float[] Norm1W;
        public required float[] Norm1B;
        public required float[] QkvW;
        public required float[] WoW;
        public required float[] AttnScale;
        public required float[] Norm2W;
        public required float[] Norm2B;
        public required float[] Ffn1W;
        public required float[] Ffn2W;
        public required float[] FfnScale;
        public required int FfnDim;
    }

    private sealed class Stage
    {
        public required float[] InW;
        public required int InDim;
        public required int OutDim;
        public required float[]? OutW;
        public required MossLayer[] Layers;
        public required int Context;
    }

    /// <summary>
    /// Allocation-free Bluestein inverse FFT (same algorithm as <see cref="Fft.Inverse"/>)
    /// for the two non power-of-two iSTFT sizes.
    /// </summary>
    internal sealed class BluesteinInverse
    {
        private readonly int _n;
        private readonly int _m;
        private readonly Complex[] _chirp;
        private readonly Complex[] _kernel;
        private readonly Complex[] _work;

        public BluesteinInverse(int n)
        {
            _n = n;
            _m = NextPowerOfTwo(2 * n - 1);
            _chirp = new Complex[n];
            long period = 2L * n;
            for (int k = 0; k < n; k++)
            {
                long reduced = (long)k * k % period;
                double angle = -Math.PI * reduced / n;
                _chirp[k] = new Complex(Math.Cos(angle), Math.Sin(angle));
            }

            _kernel = new Complex[_m];
            for (int i = 0; i < n; i++) _kernel[i] = Complex.Conjugate(_chirp[i]);
            for (int i = 1; i < n; i++) _kernel[_m - i] = _kernel[i];
            Radix2(_kernel, inverse: false);
            _work = new Complex[_m];
        }

        public void Inverse(ReadOnlySpan<Complex> input, Span<double> output)
        {
            // IFFT(x) = conj(DFT(conj(x))) / n, where DFT is the Bluestein forward.
            Array.Clear(_work);
            for (int i = 0; i < _n; i++)
                _work[i] = Complex.Conjugate(input[i]) * _chirp[i];

            Radix2(_work, inverse: false);
            for (int i = 0; i < _m; i++) _work[i] *= _kernel[i];
            Radix2(_work, inverse: true);

            double scale = 1.0 / _n;
            for (int i = 0; i < _n; i++)
            {
                Complex v = Complex.Conjugate(_work[i] * _chirp[i]) * scale;
                output[i] = v.Real;
            }
        }

        private void Radix2(Complex[] data, bool inverse)
        {
            int n = data.Length;
            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) (data[i], data[j]) = (data[j], data[i]);
            }

            for (int length = 2; length <= n; length <<= 1)
            {
                double angle = (inverse ? 2 : -2) * Math.PI / length;
                Complex root = Complex.FromPolarCoordinates(1, angle);
                for (int start = 0; start < n; start += length)
                {
                    Complex w = Complex.One;
                    int half = length >> 1;
                    for (int k = 0; k < half; k++)
                    {
                        Complex even = data[start + k];
                        Complex odd = data[start + k + half] * w;
                        data[start + k] = even + odd;
                        data[start + k + half] = even - odd;
                        w *= root;
                    }
                }
            }

            if (!inverse) return;
            double scale = 1.0 / n;
            for (int i = 0; i < n; i++) data[i] *= scale;
        }

        private static int NextPowerOfTwo(int n)
        {
            int value = 1;
            while (value < n) value <<= 1;
            return value;
        }
    }
}
