using SharpMind.Core.Quantization;

namespace SharpMind.Model;

public static class QuantHelper
{
    public static string DtypeToCompound(QuantDType dtype) => dtype switch
    {
        QuantDType.Q8_0 => "q8_0",
        QuantDType.Q4_0 => "q4_0",
        QuantDType.Q4_1 => "q4_1",
        QuantDType.Q5_0 => "q5_0",
        QuantDType.Q5_1 => "q5_1",
        QuantDType.Q8_1 => "q8_1",
        QuantDType.Q2_K or QuantDType.Q2_K_S => "q2k",
        QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => "q3k",
        QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => "q4k",
        QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M => "q5k",
        QuantDType.Q6_K or QuantDType.Q6_K_S => "q6k",
        QuantDType.Q8_K => "q8k",
        QuantDType.IQ4_NL => "q4_nl",
        QuantDType.F16 => "f16",
        _ => "f32"
    };
}
