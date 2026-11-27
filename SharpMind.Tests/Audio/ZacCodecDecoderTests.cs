using System.Numerics;
using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class ZacCodecDecoderTests
{
    [Fact]
    public void CodecConstants_MatchTheReleasedZacDecoder()
    {
        Assert.Equal(44_100, ZacCodecDecoder.SampleRate);
        Assert.Equal(2_940, ZacCodecDecoder.SamplesPerFrame);
        Assert.Equal(10, ZacCodecDecoder.Codebooks);
        Assert.Equal(4_096, ZacCodecDecoder.CodebookSize);
    }

    [Theory]
    [InlineData(736)]
    [InlineData(2208)]
    public void BluesteinInverse_MatchesTheSharedComplexInverse(int n)
    {
        var random = new Random(n);
        var input = new Complex[n];
        for (int k = 0; k < n; k++)
            input[k] = new Complex(random.NextDouble() * 2 - 1, random.NextDouble() * 2 - 1);

        var expected = new double[n];
        Fft.Inverse(input, expected);

        var actual = new double[n];
        new ZacCodecDecoder.BluesteinInverse(n).Inverse(input, actual);

        for (int i = 0; i < n; i++)
        {
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-6,
                $"sample {i}: expected {expected[i]}, got {actual[i]}");
        }
    }

    [Fact]
    public void IsZacProjector_ReturnsFalseForAMissingFile()
    {
        Assert.False(Zonos3TtsRuntime.IsTtsProjector(
            Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.gguf")));
    }

    [Fact]
    public void CollapseResidualBias_AcceptsTheSummedLayout()
    {
        var summed = new float[1024];
        for (int i = 0; i < summed.Length; i++) summed[i] = i * 0.5f;

        Assert.Same(summed, ZacCodecDecoder.CollapseResidualBias(summed));
    }

    [Fact]
    public void CollapseResidualBias_SumsTheNinePerCodebookColumns()
    {
        const int latent = 1024;
        var raw = new float[latent * 9];
        for (int c = 0; c < 9; c++)
            for (int d = 0; d < latent; d++)
                raw[c * latent + d] = c + d * 0.001f;

        float[] collapsed = ZacCodecDecoder.CollapseResidualBias(raw);

        Assert.Equal(latent, collapsed.Length);
        for (int d = 0; d < latent; d++)
            Assert.Equal(36f + 9 * d * 0.001f, collapsed[d], 4);
    }

    [Fact]
    public void CollapseResidualBias_RejectsAMalformedSize()
    {
        Assert.Throws<InvalidDataException>(
            () => ZacCodecDecoder.CollapseResidualBias(new float[1025]));
    }

    [Fact]
    public void ProjectorFinder_PrefersTheFullPrecisionFile()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"zonos3-finder-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            File.WriteAllBytes(Path.Combine(directory, "mmproj-ZONOS3-Mini-F16.gguf"), []);
            string f32 = Path.Combine(directory, "mmproj-ZONOS3-Mini-F32.gguf");
            File.WriteAllBytes(f32, []);

            string? found = TtsProjector.Find(
                Path.Combine(directory, "ZONOS3-Mini-Q8_0.gguf"), static _ => true);

            Assert.Equal(f32, found);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Zonos3Constants_MatchTheFrameFormat()
    {
        Assert.Equal(44_100, Zonos3TtsRuntime.SampleRate);
        Assert.Equal(2_940, Zonos3TtsRuntime.SamplesPerFrame);
        Assert.Equal("Savannah_en-American_F_1", Zonos3TtsRuntime.DefaultVoice);
    }
}
