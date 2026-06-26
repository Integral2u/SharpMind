namespace SharpMind.Core.Quantization;

public sealed record QuantizationConfig
{
    public const string KeyVecDotQ3K   = "vecdot_q3k";
    public const string KeyVecDotQ4K   = "vecdot_q4k";
    public const string KeyVecDotQ5K   = "vecdot_q5k";
    public const string KeyVecDotQ6K   = "vecdot_q6k";
	public const string KeyVecDotQ8_0  = "vecdot_q8_0";
	public const string KeyQuantizedMatMulQ8_0 = "qmatmul_q8_0";
	public const string KeyVecDotQ4_NL = "vecdot_q4_nl";
	public const string KeyVecDotQ4_0  = "vecdot_q4_0";
    public const string KeyVecDotQ4_1  = "vecdot_q4_1";
    public const string KeyVecDotQ5_0  = "vecdot_q5_0";
    public const string KeyVecDotQ5_1  = "vecdot_q5_1";
    public const string KeyVecDotQ8_1  = "vecdot_q8_1";
    public const string KeyVecDotQ2K   = "vecdot_q2k";
    public const string KeyVecDotQ8K   = "vecdot_q8k";
    public const string KeyHSum256     = "hsum256";
    public const string KeyHalfToFloat = "halftofloat";
    public const string KeyFloatToHalf = "floattohalf";
    public const string KeyGetScaleMinK4_Scale = "getscalemink4_scale";
    public const string KeyGetScaleMinK4_Min   = "getscalemink4_min";

    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;

    public Dictionary<string, string> ToJigSawMapping()
    {
        string suffix = Hardware switch
        {
            HardwareTier.FMA  => "_fma",
            HardwareTier.AVX2 => "_avx2",
            HardwareTier.SSE  => "_sse",
            _                 => "_scalar"
        };

        return new Dictionary<string, string>
        {
            [KeyVecDotQ3K]   = $"q3k{suffix}",
            [KeyVecDotQ4K]   = $"q4k{suffix}",
            [KeyVecDotQ5K]   = $"q5k{suffix}",
            [KeyVecDotQ6K]   = $"q6k{suffix}",
            [KeyVecDotQ8_0]  = $"q8_0{suffix}",
            [KeyQuantizedMatMulQ8_0] = suffix == "_sse" ? "qmatmul_q8_0_scalar" : $"qmatmul_q8_0{suffix}",
			[KeyVecDotQ4_NL] = $"q4_nl{suffix}",
			[KeyVecDotQ4_0]  = $"q4_0{suffix}",
            [KeyVecDotQ4_1]  = $"q4_1{suffix}",
            [KeyVecDotQ5_0]  = $"q5_0{suffix}",
            [KeyVecDotQ5_1]  = $"q5_1{suffix}",
            [KeyVecDotQ8_1]  = $"q8_1{suffix}",
            [KeyVecDotQ2K]   = $"q2k{suffix}",
            [KeyVecDotQ8K]   = $"q8k{suffix}",
            [KeyHSum256]     = $"hsum{suffix}",
            [KeyHalfToFloat] = $"halftofloat{suffix}",
            [KeyFloatToHalf] = $"floattohalf{suffix}",
            [KeyGetScaleMinK4_Scale] = $"getscalemink4_scale{suffix}",
            [KeyGetScaleMinK4_Min]   = $"getscalemink4_min{suffix}",
        };
    }
}
