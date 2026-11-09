using System.Numerics;
using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class LiquidCodecInverseStftTests
{
    [Fact]
    public void InverseStft_ReconstructsAnAnalysisStft()
    {
        const int positions = 12;
        const int n = LiquidCodecDecoder.NFft;
        const int hop = LiquidCodecDecoder.Hop;
        const int bins = LiquidCodecDecoder.Bins;
        const int length = (positions - 1) * hop + n;

        // A two-tone signal, so both magnitude and phase carry information.
        var signal = new double[length];
        for (int i = 0; i < length; i++)
            signal[i] = 0.4 * Math.Sin(2 * Math.PI * 440 * i / 24000.0) +
                        0.2 * Math.Sin(2 * Math.PI * 1234 * i / 24000.0);

        var window = new double[n];
        for (int i = 0; i < n; i++) window[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / n);

        var spectrum = new float[positions * 2 * bins];
        var frame = new double[n];
        var fft = new Complex[n];
        for (int p = 0; p < positions; p++)
        {
            for (int i = 0; i < n; i++) frame[i] = signal[p * hop + i] * window[i];
            Fft.Forward(frame, fft);
            int offset = p * 2 * bins;
            for (int k = 0; k < bins; k++)
            {
                double magnitude = fft[k].Magnitude;
                spectrum[offset + k] = (float)Math.Log(Math.Max(magnitude, 1e-8));
                spectrum[offset + bins + k] = (float)Math.Atan2(fft[k].Imaginary, fft[k].Real);
            }
        }

        float[] restored = LiquidCodecDecoder.InverseStft(spectrum, positions);

        int trim = (n - hop) / 2;
        int expectedLength = positions * hop;
        Assert.Equal(expectedLength, restored.Length);

        // The analysis/synthesis window pair reconstructs the interior; the first and last
        // windows taper to zero, so compare one window away from each edge.
        double maxError = 0;
        double energy = 0;
        for (int i = trim; i < restored.Length - trim; i++)
        {
            double expected = signal[trim + i];
            maxError = Math.Max(maxError, Math.Abs(expected - restored[i]));
            energy = Math.Max(energy, Math.Abs(expected));
        }

        Assert.True(maxError < 1e-3, $"max error {maxError} for signal peak {energy}");
    }
}
