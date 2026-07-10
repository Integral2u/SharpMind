using System.Runtime.Intrinsics.X86;

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
    public const string KeyVecDotF32   = "vecdot_f32";
    public const string KeyVecDotF16   = "vecdot_f16";

    public const string KeyQuantizedMatMulQ2K  = "qmatmul_q2k";
    public const string KeyQuantizedMatMulQ3K  = "qmatmul_q3k";
    public const string KeyQuantizedMatMulQ4K  = "qmatmul_q4k";
    public const string KeyQuantizedMatMulQ5K  = "qmatmul_q5k";
    public const string KeyQuantizedMatMulQ8K  = "qmatmul_q8k";
    public const string KeyQuantizedMatMulQ8_1 = "qmatmul_q8_1";
    public const string KeyQuantizedMatMulQ5_1 = "qmatmul_q5_1";
    public const string KeyQuantizedMatMulQ4_NL = "qmatmul_q4_nl";

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
    public const string KeyReadF32  = "read_f32";
    public const string KeyReadF16  = "read_f16";

    public const string KeyQuantizedMatMulF32 = "qmatmul_f32";
    public const string KeyQuantizedMatMulF16 = "qmatmul_f16";

    public const string KeyHSum256     = "hsum256";
    public const string KeyHalfToFloat = "halftofloat";
    public const string KeyFloatToHalf = "floattohalf";
    public const string KeyGetScaleMinK4_Scale = "getscalemink4_scale";
    public const string KeyGetScaleMinK4_Min   = "getscalemink4_min";

    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;
    public HardwareTier ResolvedHardware => Hardware switch
    {
        HardwareTier.Auto => Fma.IsSupported ? HardwareTier.FMA :
                             Avx2.IsSupported ? HardwareTier.AVX2 :
                             Sse3.IsSupported ? HardwareTier.SSE :
                                                 HardwareTier.Scalar,
        _ => Hardware
    };
    public bool Parallel { get; init; } = true;

    public Dictionary<string, string> ToJigSawMapping()
    {
        string mode = Parallel ? "_parallel" : "_serial";

        string hwSuffix = ResolvedHardware switch
        {
            HardwareTier.FMA  => "_fma",
            HardwareTier.AVX2 => "_avx2",
            HardwareTier.SSE  => "_sse",
            _                 => "_scalar"
        };

        string qmmSuffix = ResolvedHardware switch
        {
            HardwareTier.FMA  => $"{mode}_fma",
            HardwareTier.AVX2 => $"{mode}_avx2",
            HardwareTier.SSE  => $"{mode}_sse",
            _                 => $"{mode}_scalar"
        };

        return new Dictionary<string, string>
        {
            [KeyVecDotQ3K]   = $"q3k{hwSuffix}",
            [KeyVecDotQ4K]   = $"q4k{hwSuffix}",
            [KeyVecDotQ5K]   = $"q5k{hwSuffix}",
            [KeyVecDotQ6K]   = $"q6k{hwSuffix}",
            [KeyVecDotQ8_0]  = $"q8_0{hwSuffix}",
            [KeyQuantizedMatMulQ8_0] = qmmSuffix == "_serial_sse" || qmmSuffix == "_parallel_sse"
                ? $"qmatmul_q8_0{mode}_scalar"
                : $"qmatmul_q8_0{qmmSuffix}",
            [KeyQuantizedMatMulQ5_0] = qmmSuffix == "_serial_sse" || qmmSuffix == "_parallel_sse"
                ? $"qmatmul_q5_0{mode}_scalar"
                : $"qmatmul_q5_0{qmmSuffix}",
            [KeyQuantizedMatMulQ6K] = qmmSuffix == "_serial_sse" || qmmSuffix == "_parallel_sse"
                ? $"qmatmul_q6k{mode}_scalar"
                : $"qmatmul_q6k{qmmSuffix}",
			[KeyVecDotQ4_NL] = $"q4_nl{hwSuffix}",
			[KeyVecDotQ4_0]  = $"q4_0{hwSuffix}",
			[KeyQuantizedMatMulQ4_0] = $"qmatmul_q4_0{qmmSuffix}",
            [KeyVecDotQ4_1]  = $"q4_1{hwSuffix}",
			[KeyQuantizedMatMulQ4_1] = $"qmatmul_q4_1{qmmSuffix}",
            [KeyVecDotQ5_0]  = $"q5_0{hwSuffix}",
            [KeyVecDotQ5_1]  = $"q5_1{hwSuffix}",
            [KeyVecDotQ8_1]  = $"q8_1{hwSuffix}",
            [KeyVecDotQ2K]   = $"q2k{hwSuffix}",
            [KeyVecDotQ8K]   = $"q8k{hwSuffix}",
            [KeyVecDotF32]   = $"f32{hwSuffix}",
            [KeyVecDotF16]   = $"f16{hwSuffix}",
            [KeyQuantizedMatMulQ2K]  = $"qmatmul_q2k{qmmSuffix}",
            [KeyQuantizedMatMulQ3K]  = $"qmatmul_q3k{qmmSuffix}",
            [KeyQuantizedMatMulQ4K]  = $"qmatmul_q4k{qmmSuffix}",
            [KeyQuantizedMatMulQ5K]  = $"qmatmul_q5k{qmmSuffix}",
            [KeyQuantizedMatMulQ8K]  = $"qmatmul_q8k{qmmSuffix}",
            [KeyQuantizedMatMulQ8_1] = $"qmatmul_q8_1{qmmSuffix}",
            [KeyQuantizedMatMulQ5_1] = $"qmatmul_q5_1{qmmSuffix}",
            [KeyQuantizedMatMulQ4_NL]= $"qmatmul_q4_nl{qmmSuffix}",
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
            [KeyReadF32]  = "read_f32_scalar",
            [KeyReadF16]  = "read_f16_scalar",
            [KeyQuantizedMatMulF32] = $"qmatmul_f32{qmmSuffix}",
            [KeyQuantizedMatMulF16] = $"qmatmul_f16{qmmSuffix}",
            [KeyHSum256]     = $"hsum{hwSuffix}",
            [KeyHalfToFloat] = $"halftofloat{hwSuffix}",
            [KeyFloatToHalf] = $"floattohalf{hwSuffix}",
            [KeyGetScaleMinK4_Scale] = $"getscalemink4_scale{hwSuffix}",
            [KeyGetScaleMinK4_Min]   = $"getscalemink4_min{hwSuffix}",
        };
    }
}
