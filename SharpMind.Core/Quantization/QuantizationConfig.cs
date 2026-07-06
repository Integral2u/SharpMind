namespace SharpMind.Core.Quantization;

public sealed record QuantizationConfig
{
    public const string KeyVecDotQ3K   = "vecdot_q3k";
    public const string KeyVecDotQ4K   = "vecdot_q4k";
    public const string KeyVecDotQ5K   = "vecdot_q5k";
    public const string KeyVecDotQ6K   = "vecdot_q6k";
	public const string KeyVecDotQ8_0  = "vecdot_q8_0";
	public const string KeyQuantizedMatMulQ8_0 = "qmatmul_q8_0";
	public const string KeyQuantizedMatMulQ5_0 = "qmatmul_q5_0";
	public const string KeyQuantizedMatMulQ6K = "qmatmul_q6k";
	public const string KeyVecDotQ4_NL = "vecdot_q4_nl";
    public const string KeyVecDotQ4_0  = "vecdot_q4_0";
	public const string KeyQuantizedMatMulQ4_0 = "qmatmul_q4_0";
    public const string KeyVecDotQ4_1  = "vecdot_q4_1";
	public const string KeyQuantizedMatMulQ4_1 = "qmatmul_q4_1";
    public const string KeyVecDotQ5_0  = "vecdot_q5_0";
    public const string KeyVecDotQ5_1  = "vecdot_q5_1";
    public const string KeyVecDotQ8_1  = "vecdot_q8_1";
    public const string KeyVecDotQ2K   = "vecdot_q2k";
    public const string KeyVecDotQ8K   = "vecdot_q8k";

    // Missing QuantizedMatMul keys (types now have their own matmul instead of WrapVecDotAsMatMul)
    public const string KeyQuantizedMatMulQ2K  = "qmatmul_q2k";
    public const string KeyQuantizedMatMulQ3K  = "qmatmul_q3k";
    public const string KeyQuantizedMatMulQ4K  = "qmatmul_q4k";
    public const string KeyQuantizedMatMulQ5K  = "qmatmul_q5k";
    public const string KeyQuantizedMatMulQ8K  = "qmatmul_q8k";
    public const string KeyQuantizedMatMulQ8_1 = "qmatmul_q8_1";
    public const string KeyQuantizedMatMulQ5_1 = "qmatmul_q5_1";
    public const string KeyQuantizedMatMulQ4_NL = "qmatmul_q4_nl";

    // ReadQ* keys (dequantization, moved from GgufLoader)
    public const string KeyReadQ8_0 = "read_q8_0";
    public const string KeyReadQ4_0 = "read_q4_0";
    public const string KeyReadQ4_1 = "read_q4_1";
    public const string KeyReadQ5_0 = "read_q5_0";
    public const string KeyReadQ5_1 = "read_q5_1";
    public const string KeyReadQ8_1 = "read_q8_1";
    public const string KeyReadQ4_NL = "read_q4_nl";
    public const string KeyReadQ2K  = "read_q2k";
    public const string KeyReadQ3K  = "read_q3k";
    public const string KeyReadQ4K  = "read_q4k";
    public const string KeyReadQ5K  = "read_q5k";
    public const string KeyReadQ6K  = "read_q6k";
    public const string KeyReadQ8K  = "read_q8k";

    // F32/F16 QuantizedMatMul keys
    public const string KeyQuantizedMatMulF32 = "qmatmul_f32";
    public const string KeyQuantizedMatMulF16 = "qmatmul_f16";

    public const string KeyHSum256     = "hsum256";
    public const string KeyHalfToFloat = "halftofloat";
    public const string KeyFloatToHalf = "floattohalf";
    public const string KeyGetScaleMinK4_Scale = "getscalemink4_scale";
    public const string KeyGetScaleMinK4_Min   = "getscalemink4_min";

    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;
    public bool Parallel { get; init; } = false;

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
            [KeyQuantizedMatMulQ5_0] = suffix == "_sse" ? "qmatmul_q5_0_scalar" : $"qmatmul_q5_0{suffix}",
            [KeyQuantizedMatMulQ6K] = suffix == "_sse" ? "qmatmul_q6k_scalar" : $"qmatmul_q6k{suffix}",
			[KeyVecDotQ4_NL] = $"q4_nl{suffix}",
			[KeyVecDotQ4_0]  = $"q4_0{suffix}",
			[KeyQuantizedMatMulQ4_0] = suffix == "_sse" ? "qmatmul_q4_0_scalar" : $"qmatmul_q4_0{suffix}",
            [KeyVecDotQ4_1]  = $"q4_1{suffix}",
			[KeyQuantizedMatMulQ4_1] = suffix == "_sse" ? "qmatmul_q4_1_scalar" : $"qmatmul_q4_1{suffix}",
            [KeyVecDotQ5_0]  = $"q5_0{suffix}",
            [KeyVecDotQ5_1]  = $"q5_1{suffix}",
            [KeyVecDotQ8_1]  = $"q8_1{suffix}",
            [KeyVecDotQ2K]   = $"q2k{suffix}",
            [KeyVecDotQ8K]   = $"q8k{suffix}",
            // Missing QuantizedMatMul — all map to _scalar for now
            [KeyQuantizedMatMulQ2K]  = "qmatmul_q2k_scalar",
            [KeyQuantizedMatMulQ3K]  = "qmatmul_q3k_scalar",
            [KeyQuantizedMatMulQ4K]  = "qmatmul_q4k_scalar",
            [KeyQuantizedMatMulQ5K]  = "qmatmul_q5k_scalar",
            [KeyQuantizedMatMulQ8K]  = "qmatmul_q8k_scalar",
            [KeyQuantizedMatMulQ8_1] = "qmatmul_q8_1_scalar",
            [KeyQuantizedMatMulQ5_1] = "qmatmul_q5_1_scalar",
            [KeyQuantizedMatMulQ4_NL]= "qmatmul_q4_nl_scalar",
            // ReadQ* — all map to _scalar for now
            [KeyReadQ8_0] = "read_q8_0_scalar",
            [KeyReadQ4_0] = "read_q4_0_scalar",
            [KeyReadQ4_1] = "read_q4_1_scalar",
            [KeyReadQ5_0] = "read_q5_0_scalar",
            [KeyReadQ5_1] = "read_q5_1_scalar",
            [KeyReadQ8_1] = "read_q8_1_scalar",
            [KeyReadQ4_NL] = "read_q4_nl_scalar",
            [KeyReadQ2K]  = "read_q2k_scalar",
            [KeyReadQ3K]  = "read_q3k_scalar",
            [KeyReadQ4K]  = "read_q4k_scalar",
            [KeyReadQ5K]  = "read_q5k_scalar",
            [KeyReadQ6K]  = "read_q6k_scalar",
            [KeyReadQ8K]  = "read_q8k_scalar",
            // F32/F16 QuantizedMatMul
            [KeyQuantizedMatMulF32] = "qmatmul_f32_scalar",
            [KeyQuantizedMatMulF16] = "qmatmul_f16_scalar",
            [KeyHSum256]     = $"hsum{suffix}",
            [KeyHalfToFloat] = $"halftofloat{suffix}",
            [KeyFloatToHalf] = $"floattohalf{suffix}",
            [KeyGetScaleMinK4_Scale] = $"getscalemink4_scale{suffix}",
            [KeyGetScaleMinK4_Min]   = $"getscalemink4_min{suffix}",
        };
    }
}
