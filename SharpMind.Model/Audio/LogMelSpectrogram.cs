namespace SharpMind.Model.Audio;

/// <summary>
/// Log-mel frontend settings. The defaults are the LFM3-Audio values: 16 kHz mono,
/// 25 ms Hann window every 10 ms without centering, 128 Slaney mel filters, base-10
/// logarithm clamped at -12 and an affine (x + 4) / 4 normalization.
/// </summary>
public sealed record LogMelOptions
{
    public int SampleRate { get; init; } = 16_000;
    public int MelBins { get; init; } = 128;
    public int NFft { get; init; } = 400;
    public int HopLength { get; init; } = 160;
    public int WinLength { get; init; } = 400;
    public float FMin { get; init; } = 0f;
    /// <summary>Upper filter frequency; 0 or less means the Nyquist frequency.</summary>
    public float FMax { get; init; } = 0f;
    public float LogFloor { get; init; } = -12f;
    public float NormalizationOffset { get; init; } = 4f;
    public float NormalizationDivisor { get; init; } = 4f;

    internal void Validate()
    {
        if (SampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(SampleRate));
        if (MelBins <= 0) throw new ArgumentOutOfRangeException(nameof(MelBins));
        if (NFft <= 0) throw new ArgumentOutOfRangeException(nameof(NFft));
        if (HopLength <= 0) throw new ArgumentOutOfRangeException(nameof(HopLength));
        if (WinLength <= 0 || WinLength > NFft)
            throw new ArgumentOutOfRangeException(nameof(WinLength), "Window must fit inside the FFT.");
        if (NormalizationDivisor == 0) throw new ArgumentOutOfRangeException(nameof(NormalizationDivisor));
    }
}

/// <summary>
/// The LFM3-Audio log-mel spectrogram. Output is <c>[frames, melBins]</c> (time-major);
/// the encoder treats each group of eight frames as one token.
/// </summary>
public static class LogMelSpectrogram
{
    /// <summary>Number of spectrogram frames for a clip, after right-padding to a whole hop.</summary>
    public static int FrameCount(int numSamples, LogMelOptions? options = null)
    {
        options ??= new LogMelOptions();
        options.Validate();
        return 1 + (PaddedLength(numSamples, options) - options.NFft) / options.HopLength;
    }

    /// <summary>
    /// Right-pads so that no trailing samples are dropped: clips shorter than one window
    /// become one window, longer clips are padded to a whole hop after the first window.
    /// </summary>
    public static int PaddedLength(int numSamples, LogMelOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(numSamples);
        options ??= new LogMelOptions();
        options.Validate();
        if (numSamples < options.NFft) return options.NFft;
        int remainder = (numSamples - options.NFft) % options.HopLength;
        return remainder == 0 ? numSamples : numSamples + options.HopLength - remainder;
    }

    /// <summary>Computes the normalized log-mel spectrogram, <c>[frames, melBins]</c>.</summary>
    public static float[,] Compute(ReadOnlySpan<float> samples, LogMelOptions? options = null)
    {
        options ??= new LogMelOptions();
        options.Validate();

        int paddedLength = PaddedLength(samples.Length, options);
        var padded = new float[paddedLength];
        samples.CopyTo(padded);

        int frames = 1 + (paddedLength - options.NFft) / options.HopLength;
        double[] window = HannWindow(options.WinLength);
        double[][] filters = BuildSlaneyFilters(options);
        int bins = options.NFft / 2 + 1;
        var power = new float[bins];
        var frame = new float[options.NFft];
        var result = new float[frames, options.MelBins];
        int windowOffset = (options.NFft - options.WinLength) / 2;

        for (int f = 0; f < frames; f++)
        {
            Array.Clear(frame);
            int start = f * options.HopLength;
            for (int i = 0; i < options.WinLength; i++)
                frame[windowOffset + i] = (float)(padded[start + i] * window[i]);

            Fft.PowerSpectrum(frame, power);

            for (int m = 0; m < options.MelBins; m++)
            {
                double[] filter = filters[m];
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += filter[k] * power[k];

                double log = sum > 0 ? Math.Log10(sum) : double.NegativeInfinity;
                if (log < options.LogFloor) log = options.LogFloor;
                result[f, m] = (float)((log + options.NormalizationOffset) / options.NormalizationDivisor);
            }
        }

        return result;
    }

    /// <summary>Slaney-normalized triangular mel filterbank, <c>[melBins][nFft / 2 + 1]</c>.</summary>
    public static double[][] BuildSlaneyFilters(LogMelOptions? options = null)
    {
        options ??= new LogMelOptions();
        options.Validate();

        double fMax = options.FMax > 0 ? options.FMax : options.SampleRate / 2.0;
        int bins = options.NFft / 2 + 1;
        double[] fftFreqs = Linspace(0, options.SampleRate / 2.0, bins);
        double[] melPoints = Linspace(HertzToMel(options.FMin), HertzToMel(fMax), options.MelBins + 2);
        double[] hz = new double[melPoints.Length];
        for (int i = 0; i < melPoints.Length; i++) hz[i] = MelToHertz(melPoints[i]);

        var filters = new double[options.MelBins][];
        for (int m = 0; m < options.MelBins; m++)
        {
            double left = hz[m];
            double center = hz[m + 1];
            double right = hz[m + 2];
            var filter = new double[bins];
            for (int k = 0; k < bins; k++)
            {
                double frequency = fftFreqs[k];
                double down = (frequency - left) / (center - left);
                double up = (right - frequency) / (right - center);
                double weight = Math.Max(0, Math.Min(down, up));
                if (weight > 0) weight *= 2.0 / (right - left);
                filter[k] = weight;
            }
            filters[m] = filter;
        }

        return filters;
    }

    private static double[] HannWindow(int length)
    {
        var window = new double[length];
        for (int i = 0; i < length; i++)
            window[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / length);
        return window;
    }

    private static double[] Linspace(double start, double end, int count)
    {
        var values = new double[count];
        if (count == 1)
        {
            values[0] = start;
            return values;
        }
        double step = (end - start) / (count - 1);
        for (int i = 0; i < count; i++) values[i] = start + i * step;
        return values;
    }

    private const double SlaneyFSp = 200.0 / 3.0;
    private const double SlaneyMinLogHz = 1000.0;
    private const double SlaneyMinLogMel = SlaneyMinLogHz / SlaneyFSp;
    private static readonly double SlaneyLogStep = Math.Log(6.4) / 27.0;

    /// <summary>Slaney mel scale: linear below 1 kHz, logarithmic above it.</summary>
    public static double HertzToMel(double hertz) => hertz < SlaneyMinLogHz
        ? hertz / SlaneyFSp
        : SlaneyMinLogMel + Math.Log(hertz / SlaneyMinLogHz) / SlaneyLogStep;

    /// <summary>Inverse of <see cref="HertzToMel"/>.</summary>
    public static double MelToHertz(double mel) => mel < SlaneyMinLogMel
        ? mel * SlaneyFSp
        : SlaneyMinLogHz * Math.Exp((mel - SlaneyMinLogMel) * SlaneyLogStep);
}
