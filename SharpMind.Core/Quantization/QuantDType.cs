namespace SharpMind.Core.Quantization;
public enum QuantDType : uint
{
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
    Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ4_NL = 20, // 4-bit non-linear (block_iq4_nl: d[2]+qs[16]=18B, 32 el)
    // Integer raw types (unquantized, element-is-own-block)
    I8 = 16, I16 = 17, I32 = 18,
    // K-quant variant aliases (not real GGML types; identical block format to base)
    Q2_K_S = 100, Q3_K_S = 101, Q3_K_M = 102, Q3_K_L = 103,
    Q4_K_S = 104, Q4_K_M = 105, Q5_K_S = 106, Q5_K_M = 107,
    Q6_K_S = 108,
}
