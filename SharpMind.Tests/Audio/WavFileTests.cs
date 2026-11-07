using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class WavFileTests
{
    private static string TempPath() =>
        Path.Combine(Path.GetTempPath(), $"sharpmind-{Guid.NewGuid():N}.wav");

    [Fact]
    public void Pcm16_RoundTrips()
    {
        string path = TempPath();
        try
        {
            var samples = new float[1600];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = (float)(Math.Sin(2 * Math.PI * 440 * i / 16000.0) * 0.5);

            WavFile.Write(path, samples, 16000);
            WavData data = WavFile.Read(path);

            Assert.Equal(16000, data.SampleRate);
            Assert.Equal(1, data.Channels);
            Assert.Equal(samples.Length, data.Samples.Length);
            for (int i = 0; i < samples.Length; i++)
                Assert.InRange((double)Math.Abs(data.Samples[i] - samples[i]), 0, 1.0 / 32767 + 1e-6);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Float32_RoundTrips()
    {
        string path = TempPath();
        try
        {
            float[] samples = [0f, 0.125f, -0.5f, 0.999f, -1f];
            WavFile.Write(path, samples, 24000, format: WavSampleFormat.Float32);
            WavData data = WavFile.Read(path);

            Assert.Equal(24000, data.SampleRate);
            Assert.Equal(samples.Length, data.Samples.Length);
            for (int i = 0; i < samples.Length; i++) Assert.Equal(samples[i], data.Samples[i]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Stereo_IsDownmixedByMean()
    {
        string path = TempPath();
        try
        {
            // Frames: (0.5, -0.5) -> 0, (0.25, 0.25) -> 0.25, (-1, 0) -> -0.5
            float[] interleaved = [0.5f, -0.5f, 0.25f, 0.25f, -1f, 0f];
            WavFile.Write(path, interleaved, 16000, channels: 2);
            WavData data = WavFile.Read(path);

            Assert.Equal(2, data.Channels);
            Assert.Equal(3, data.Samples.Length);
            Assert.InRange((double)data.Samples[0], -1e-4, 1e-4);
            Assert.InRange((double)data.Samples[1], 0.25 - 1e-4, 0.25 + 1e-4);
            Assert.InRange((double)data.Samples[2], -0.5 - 1e-4, -0.5 + 1e-4);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Resample_UpsamplesAndKeepsTheTone()
    {
        int sourceRate = 8000;
        int targetRate = 16000;
        int length = 8000; // one second
        var samples = new float[length];
        for (int i = 0; i < length; i++)
            samples[i] = (float)Math.Sin(2 * Math.PI * 440 * i / sourceRate);

        float[] resampled = WavFile.Resample(samples, sourceRate, targetRate);

        Assert.Equal(16000, resampled.Length);
        int crossings = 0;
        for (int i = 1; i < resampled.Length; i++)
            if (resampled[i - 1] < 0 && resampled[i] >= 0) crossings++;
        Assert.InRange(crossings, 438, 442);
    }

    [Fact]
    public void Resample_SameRateReturnsACopy()
    {
        float[] samples = [0.1f, 0.2f, 0.3f];
        float[] copy = WavFile.Resample(samples, 16000, 16000);
        Assert.Equal(samples, copy);
        Assert.NotSame(samples, copy);
    }
}
