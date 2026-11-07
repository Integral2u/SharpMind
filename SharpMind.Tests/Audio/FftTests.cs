using System.Numerics;
using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class FftTests
{
    [Theory]
    [InlineData(7)]
    [InlineData(12)]
    [InlineData(400)]
    public void PowerSpectrum_MatchesNaiveDft(int length)
    {
        var random = new Random(1234 + length);
        var input = new float[length];
        for (int i = 0; i < length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);

        var power = new float[length / 2 + 1];
        Fft.PowerSpectrum(input, power);

        for (int k = 0; k <= length / 2; k++)
        {
            double real = 0, imaginary = 0;
            for (int j = 0; j < length; j++)
            {
                double angle = -2 * Math.PI * k * j / length;
                real += input[j] * Math.Cos(angle);
                imaginary += input[j] * Math.Sin(angle);
            }

            double expected = real * real + imaginary * imaginary;
            Assert.True(Math.Abs(power[k] - expected) <= 1e-3 * Math.Max(1, expected),
                $"bin {k}: expected {expected}, got {power[k]}");
        }
    }

    [Fact]
    public void PowerSpectrum_Impulse_IsFlat()
    {
        var input = new float[400];
        input[0] = 1;
        var power = new float[201];
        Fft.PowerSpectrum(input, power);

        foreach (float value in power) Assert.InRange(value, 0.999f, 1.001f);
    }

    [Fact]
    public void Inverse_RoundTrips()
    {
        int length = 400;
        var real = new double[length];
        for (int i = 0; i < length; i++) real[i] = Math.Sin(2 * Math.PI * 37 * i / length) + 0.25;

        var spectrum = new Complex[length];
        Fft.Forward(real, spectrum);
        var roundTrip = new double[length];
        Fft.Inverse(spectrum, roundTrip);

        for (int i = 0; i < length; i++) Assert.InRange(Math.Abs(roundTrip[i] - real[i]), 0, 1e-9);
    }
}
