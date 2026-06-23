namespace SharpMind.Model.Format;
public enum GgufDtype : uint
{
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9, Q2_K = 10, Q3_K = 11, Q4_K = 12,
    Q5_K = 13, Q6_K = 14, Q8_K = 15,
    // Medium K-quant variants (identical block format to base types)
    Q2_K_S = 16, Q3_K_S = 17, Q3_K_M = 18, Q3_K_L = 19,
    Q4_K_S = 20, Q4_K_M = 21, Q5_K_S = 22, Q5_K_M = 23,
    Q6_K_S = 24,
}
