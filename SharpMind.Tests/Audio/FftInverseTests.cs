using System.Numerics;
using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

/// <summary>
/// Extra inverse-transform coverage for the sizes LiquidCodec uses: 1280 is not a power
/// of two, so it exercises the Bluestein path that the ISTFT depends on.
/// </summary>
public sealed class FftInverseTests
{
    [Theory]
    [InlineData(1280)]  // LiquidCodec STFT size (Bluestein)
    [InlineData(400)]   // LFM3-Audio frontend (Bluestein)
    [InlineData(1024)]  // radix-2
    public void InverseReconstructsTheForwardTransform(int n)
    {
        var random = new Random(1234);
        var signal = new double[n];
        for (int i = 0; i < n; i++) signal[i] = random.NextDouble() * 2 - 1;

        var spectrum = new Complex[n];
        Fft.Forward(signal, spectrum);
        var restored = new double[n];
        Fft.Inverse(spectrum, restored);

        double maxError = 0;
        for (int i = 0; i < n; i++) maxError = Math.Max(maxError, Math.Abs(signal[i] - restored[i]));
        Assert.True(maxError < 1e-6, $"n={n} max error {maxError}");
    }

    [Fact]
    public void InverseOfAHermitianSpectrumStaysFinite()
    {
        const int n = 1280;
        const int bins = n / 2 + 1;
        var random = new Random(99);
        var spectrum = new Complex[n];
        for (int k = 0; k < bins; k++)
        {
            double magnitude = random.NextDouble();
            double angle = random.NextDouble() * Math.PI * 2;
            spectrum[k] = new Complex(magnitude * Math.Cos(angle), magnitude * Math.Sin(angle));
        }
        spectrum[0] = new Complex(spectrum[0].Real, 0);
        spectrum[bins - 1] = new Complex(spectrum[bins - 1].Real, 0);
        for (int k = 1; k < bins - 1; k++) spectrum[n - k] = Complex.Conjugate(spectrum[k]);

        var restored = new double[n];
        Fft.Inverse(spectrum, restored);

        for (int i = 0; i < n; i++) Assert.True(double.IsFinite(restored[i]));
        Assert.All(restored, sample => Assert.InRange(sample, -4.0, 4.0));
    }
}
