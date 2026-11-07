using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class LogMelSpectrogramTests
{
    [Theory]
    [InlineData(0, 400, 1)]
    [InlineData(400, 400, 1)]
    [InlineData(401, 560, 2)]
    [InlineData(1000, 1040, 5)]
    [InlineData(1600, 1680, 9)]
    public void Padding_MatchesTheReferenceFormula(int samples, int padded, int frames)
    {
        Assert.Equal(padded, LogMelSpectrogram.PaddedLength(samples));
        Assert.Equal(frames, LogMelSpectrogram.FrameCount(samples));
    }

    [Fact]
    public void Silence_IsTheNormalizedFloor()
    {
        float[,] mel = LogMelSpectrogram.Compute(new float[1600]);

        Assert.Equal(9, mel.GetLength(0));
        Assert.Equal(128, mel.GetLength(1));
        for (int f = 0; f < mel.GetLength(0); f++)
            for (int m = 0; m < mel.GetLength(1); m++)
                Assert.Equal(-2f, mel[f, m], precision: 5);
    }

    [Fact]
    public void Tone_PeaksInTheFilterAroundOneKilohertz()
    {
        const int rate = 16000;
        const int frequency = 1000;
        var samples = new float[8000];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(Math.Sin(2 * Math.PI * frequency * i / rate) * 0.5);

        float[,] mel = LogMelSpectrogram.Compute(samples);
        int frames = mel.GetLength(0);

        // Skip the first window: the Hann taper and the frame edges colour the spectrum.
        int frame = frames / 2;
        int peak = 0;
        for (int m = 1; m < mel.GetLength(1); m++)
            if (mel[frame, m] > mel[frame, peak]) peak = m;

        Assert.InRange(CenterFrequency(peak), frequency - 150.0, frequency + 150.0);
        Assert.True(mel[frame, peak] > -2f);
    }

    [Fact]
    public void Filters_AreSlaneyNormalizedTriangles()
    {
        double[][] filters = LogMelSpectrogram.BuildSlaneyFilters();
        Assert.Equal(128, filters.Length);
        Assert.All(filters, filter => Assert.Equal(201, filter.Length));

        // Each filter has one contiguous positive region and a positive area.
        foreach (double[] filter in filters)
        {
            int first = Array.FindIndex(filter, value => value > 0);
            int last = Array.FindLastIndex(filter, value => value > 0);
            double area = 0;
            for (int k = first; k <= last; k++)
            {
                Assert.True(filter[k] >= 0);
                area += filter[k];
            }
            Assert.True(area > 0);
        }
    }

    [Fact]
    public void SlaneyMelScale_MatchesKnownAnchors()
    {
        Assert.Equal(0, LogMelSpectrogram.HertzToMel(0), precision: 10);
        Assert.Equal(15, LogMelSpectrogram.HertzToMel(1000), precision: 10);
        Assert.Equal(1000, LogMelSpectrogram.MelToHertz(15), precision: 6);
        Assert.Equal(6.4 * 1000, LogMelSpectrogram.MelToHertz(42), precision: 3);
    }

    private static double CenterFrequency(int filter)
    {
        double low = LogMelSpectrogram.HertzToMel(0);
        double high = LogMelSpectrogram.HertzToMel(8000);
        double step = (high - low) / 129;
        return LogMelSpectrogram.MelToHertz(low + (filter + 1) * step);
    }
}
