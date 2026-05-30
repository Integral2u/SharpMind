namespace SharpMind.Core.Quantization;

public sealed record QuantizationConfig
{
    public const string KeyVecDotQ3K   = "vecdot_q3k";
    public const string KeyVecDotQ4K   = "vecdot_q4k";
    public const string KeyVecDotQ5K   = "vecdot_q5k";
    public const string KeyVecDotQ6K   = "vecdot_q6k";
    public const string KeyVecDotQ8_0  = "vecdot_q8_0";
    public const string KeyVecDotQ4_0  = "vecdot_q4_0";
    public const string KeyVecDotQ4_1  = "vecdot_q4_1";
    public const string KeyVecDotQ5_0  = "vecdot_q5_0";
    public const string KeyVecDotQ5_1  = "vecdot_q5_1";
    public const string KeyVecDotQ8_1  = "vecdot_q8_1";
    public const string KeyVecDotQ2K   = "vecdot_q2k";
    public const string KeyVecDotQ8K   = "vecdot_q8k";
    public const string KeyHSum256     = "hsum256";
    public const string KeyHalfToFloat = "halftofloat";
    public const string KeyGetScaleMinK4_Scale = "getscalemink4_scale";
    public const string KeyGetScaleMinK4_Min   = "getscalemink4_min";

    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;

    public Dictionary<string, string> ToJigSawMapping()
    {
        // VecDot methods that have FMA / AVX2 variants — use the requested tier.
        // Methods that only have Scalar always map to _scalar regardless of tier.
        string vecSuffix = Hardware switch
        {
            HardwareTier.FMA  => "_fma",
            HardwareTier.AVX2 => "_avx2",
            _                 => "_scalar"
        };

        return new Dictionary<string, string>
        {
            // Has FMA, AVX2, Scalar
            [KeyVecDotQ3K]   = $"q3k{vecSuffix}",
            [KeyVecDotQ4K]   = $"q4k{vecSuffix}",
            [KeyVecDotQ5K]   = $"q5k{vecSuffix}",
            [KeyVecDotQ6K]   = $"q6k{vecSuffix}",
            [KeyVecDotQ2K]   = $"q2k{vecSuffix}",
            // Has AVX2, SSE, Scalar (FMA falls back)
            [KeyVecDotQ4_0]  = Hardware == HardwareTier.FMA ? "q4_0_scalar"
                             : $"q4_0{vecSuffix}",
            [KeyVecDotQ4_1]  = Hardware == HardwareTier.FMA ? "q4_1_scalar"
                             : $"q4_1{vecSuffix}",
            [KeyVecDotQ8_0]  = Hardware == HardwareTier.FMA ? "q8_0_fma"
                             : Hardware == HardwareTier.AVX2 ? "q8_0_avx2"
                             : $"q8_0{vecSuffix}",
            [KeyVecDotQ8_1]  = Hardware == HardwareTier.FMA ? "q8_1_fma"
                             : Hardware == HardwareTier.AVX2 ? "q8_1_avx2"
                             : $"q8_1{vecSuffix}",
            [KeyVecDotQ8K]   = Hardware == HardwareTier.FMA ? "q8k_fma"
                             : Hardware == HardwareTier.AVX2 ? "q8k_avx2"
                             : $"q8k{vecSuffix}",
            // Scalar only — always scalar regardless of tier
            [KeyVecDotQ5_0]  = "q5_0_scalar",
            [KeyVecDotQ5_1]  = "q5_1_scalar",
            // Helpers
            [KeyHSum256]     = Hardware is HardwareTier.FMA or HardwareTier.AVX2 ? "avx" : "scalar",
            [KeyHalfToFloat] = Hardware is HardwareTier.FMA or HardwareTier.AVX2 ? "f16c" : "scalar",
            [KeyGetScaleMinK4_Scale] = "scalar",
            [KeyGetScaleMinK4_Min]   = "scalar",
        };
    }
}
