using System.Collections.Concurrent;
using System.Numerics;

namespace SharpMind.Model.Audio;

/// <summary>
/// FFT helpers for audio preprocessing and codec synthesis. Arbitrary transform sizes are
/// supported: power-of-two sizes use an iterative radix-2 transform, everything else uses
/// Bluestein's chirp-z algorithm on top of the same radix-2 kernel. The LFM3-Audio frontend
/// needs 400-point transforms and the LiquidCodec needs 1280-point ones, so neither is a
/// power of two.
/// </summary>
public static class Fft
{
    private static readonly ConcurrentDictionary<int, Complex[]> ChirpCache = new();

    /// <summary>Smallest power of two greater than or equal to <paramref name="n"/>.</summary>
    public static int NextPowerOfTwo(int n)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        int value = 1;
        while (value < n) value <<= 1;
        return value;
    }

    /// <summary>
    /// Power spectrum of a real signal, length <c>input.Length / 2 + 1</c>.
    /// </summary>
    public static void PowerSpectrum(ReadOnlySpan<float> input, Span<float> power)
    {
        int n = input.Length;
        if (n == 0) throw new ArgumentException("FFT input must not be empty.", nameof(input));
        if (power.Length < n / 2 + 1)
            throw new ArgumentException($"Power spectrum needs {n / 2 + 1} values, got {power.Length}.", nameof(power));

        var spectrum = new Complex[n];
        for (int i = 0; i < n; i++) spectrum[i] = new Complex(input[i], 0);

        Transform(spectrum, inverse: false);

        for (int i = 0; i <= n / 2; i++)
        {
            Complex c = spectrum[i];
            power[i] = (float)(c.Real * c.Real + c.Imaginary * c.Imaginary);
        }
    }

    /// <summary>Forward complex transform of <paramref name="real"/> (imaginary part zero).</summary>
    public static void Forward(ReadOnlySpan<double> real, Span<Complex> spectrum)
    {
        int n = real.Length;
        if (n == 0) throw new ArgumentException("FFT input must not be empty.", nameof(real));
        if (spectrum.Length < n)
            throw new ArgumentException($"Spectrum needs {n} values, got {spectrum.Length}.", nameof(spectrum));

        var buffer = new Complex[n];
        for (int i = 0; i < n; i++) buffer[i] = new Complex(real[i], 0);
        Transform(buffer, inverse: false);
        buffer.CopyTo(spectrum);
    }

    /// <summary>Inverse complex transform; the real part is written to <paramref name="output"/>.</summary>
    public static void Inverse(ReadOnlySpan<Complex> spectrum, Span<double> output)
    {
        int n = spectrum.Length;
        if (n == 0) throw new ArgumentException("FFT input must not be empty.", nameof(spectrum));
        if (output.Length < n)
            throw new ArgumentException($"Output needs {n} values, got {output.Length}.", nameof(output));

        var buffer = spectrum.ToArray();
        Transform(buffer, inverse: true);
        for (int i = 0; i < n; i++) output[i] = buffer[i].Real;
    }

    private static void Transform(Complex[] data, bool inverse)
    {
        int n = data.Length;
        if (n == 1) return;

        if (IsPowerOfTwo(n))
        {
            Radix2(data, inverse);
            return;
        }

        Bluestein(data, inverse);
    }

    private static bool IsPowerOfTwo(int n) => (n & (n - 1)) == 0;

    private static void Radix2(Complex[] data, bool inverse)
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

    /// <summary>
    /// Bluestein's algorithm: reduce a length-n DFT to a power-of-two convolution.
    /// X[k] = w[k] * sum_j (x[j] w[j]) conj(w[k-j]), with w[j] = exp(-i pi j^2 / n).
    /// The inverse is conj(DFT(conj(x))) / n.
    /// </summary>
    private static void Bluestein(Complex[] data, bool inverse)
    {
        int n = data.Length;
        if (inverse)
        {
            for (int i = 0; i < n; i++) data[i] = Complex.Conjugate(data[i]);
            BluesteinForward(data);
            double scale = 1.0 / n;
            for (int i = 0; i < n; i++) data[i] = Complex.Conjugate(data[i]) * scale;
            return;
        }

        BluesteinForward(data);
    }

    private static void BluesteinForward(Complex[] data)
    {
        int n = data.Length;
        int m = NextPowerOfTwo(2 * n - 1);
        Complex[] chirp = Chirp(n);

        var a = new Complex[m];
        for (int i = 0; i < n; i++) a[i] = data[i] * chirp[i];

        Complex[] b = BuildConvolutionKernel(n, m, chirp);
        Radix2(a, inverse: false);
        Radix2(b, inverse: false);
        for (int i = 0; i < m; i++) a[i] *= b[i];
        Radix2(a, inverse: true);

        for (int i = 0; i < n; i++) data[i] = a[i] * chirp[i];
    }

    private static Complex[] BuildConvolutionKernel(int n, int m, Complex[] chirp)
    {
        var b = new Complex[m];
        for (int i = 0; i < n; i++) b[i] = Complex.Conjugate(chirp[i]);
        for (int i = 1; i < n; i++) b[m - i] = b[i];
        return b;
    }

    private static Complex[] Chirp(int n) => ChirpCache.GetOrAdd(n, static length =>
    {
        var chirp = new Complex[length];
        long period = 2L * length;
        for (int k = 0; k < length; k++)
        {
            long reduced = (long)k * k % period;
            double angle = -Math.PI * reduced / length;
            chirp[k] = new Complex(Math.Cos(angle), Math.Sin(angle));
        }
        return chirp;
    });
}
