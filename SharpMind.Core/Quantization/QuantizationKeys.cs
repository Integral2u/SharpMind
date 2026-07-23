namespace SharpMind.Core.Quantization;

public static class QuantizationKeys
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
    public const string KeyVecDotI8    = "vecdot_i8";
    public const string KeyVecDotI16   = "vecdot_i16";
    public const string KeyVecDotI32   = "vecdot_i32";

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
    public const string KeyReadI8   = "read_i8";
    public const string KeyReadI16  = "read_i16";
    public const string KeyReadI32  = "read_i32";

    public const string KeyQuantizedMatMulF32 = "qmatmul_f32";
    public const string KeyQuantizedMatMulF16 = "qmatmul_f16";
    public const string KeyQuantizedMatMulI8  = "qmatmul_i8";
    public const string KeyQuantizedMatMulI16 = "qmatmul_i16";
    public const string KeyQuantizedMatMulI32 = "qmatmul_i32";

    public const string KeyHSum256     = "hsum256";
    public const string KeyHalfToFloat = "halftofloat";
    public const string KeyFloatToHalf = "floattohalf";
    public const string KeyGetScaleMinK4_Scale = "getscalemink4_scale";
    public const string KeyGetScaleMinK4_Min   = "getscalemink4_min";
}
