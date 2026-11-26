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
    public void Zonos3Constants_MatchTheFrameFormat()
    {
        Assert.Equal(44_100, Zonos3TtsRuntime.SampleRate);
        Assert.Equal(2_940, Zonos3TtsRuntime.SamplesPerFrame);
        Assert.Equal("Savannah_en-American_F_1", Zonos3TtsRuntime.DefaultVoice);
    }
}
