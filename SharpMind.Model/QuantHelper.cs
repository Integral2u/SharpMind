using SharpMind.Core.Quantization;

namespace SharpMind.Model;

public static class QuantHelper
{
    /// <summary>
    /// Kernel-name prefix for a dtype. Every dtype a GGUF file can carry a
    /// linear layer in must have its own prefix: falling back to "f32" (as an
    /// unlisted IQ1_S/TQ1_0 layer used to) hands those raw block bytes to the
    /// float kernel, which decodes them as garbage instead of quantized weights.
    /// </summary>
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
        QuantDType.MXFP4 => "mxfp4",
        QuantDType.Q1_0 => "q1_0",
        QuantDType.TQ2_0 => "tq2_0",
        QuantDType.F16 => "f16",
        QuantDType.BF16 => "bf16",
        // IQ1_S, IQ1_M and TQ1_0 have decode kernels but no linear-layer
        // compound, so a file made of them used to fall through to "f32" and run
        // its raw block bytes through the float kernel. Refuse it instead of
        // emitting garbage; a message beats a fluent answer built on noise.
        QuantDType.IQ1_S or QuantDType.IQ1_M or QuantDType.TQ1_0 =>
            throw new NotSupportedException(
                $"{dtype} weights are not wired to a linear-layer kernel. SharpMind can decode " +
                "these tensors but has no matmul target for them yet; convert the model to " +
                "Q8_0, Q4_K_M or MXFP4."),
        _ => "f32"
    };

    /// <summary>
    /// The compound kernel name for <paramref name="dtype"/>, i.e. prefix plus
    /// the serial/parallel and tier suffix.
    ///
    /// The suffix is read out of the mapping's own <c>qmatmul_&lt;prefix&gt;</c>
    /// entry rather than guessed from the ISA. Guessing meant asking
    /// <c>Fma.IsSupported</c>, which is false on Arm64: every linear layer then
    /// selected <c>_scalar</c>, so the NEON Q4_K/Q6_K/Q8_0 matmuls were only
    /// reachable from <c>kbench</c> and a real model decoded on one core's
    /// scalar path. The mapping is where the tier decision already lives.
    /// </summary>
    public static string GetCompound(QuantDType dtype, Dictionary<string, string>? mapping)
    {
        string prefix = DtypeToCompound(dtype);
        string? suffix = ExtractQmatMulSuffix($"qmatmul_{prefix}", mapping)
            ?? ExtractQmatMulSuffix("qmatmul_f32", mapping)
            ?? IsaSuffix();
        return $"{prefix}{suffix}";
    }

    private static string? ExtractQmatMulSuffix(string key, Dictionary<string, string>? mapping)
    {
        if (mapping == null || !mapping.TryGetValue(key, out string? value)) return null;

        int serialIdx = value.IndexOf("_serial_", StringComparison.Ordinal);
        int parallelIdx = value.IndexOf("_parallel_", StringComparison.Ordinal);
        int idx = serialIdx >= 0 ? serialIdx : parallelIdx;
        return idx < 0 ? null : value[idx..];
    }

    /// <summary>Suffix used when no mapping is available at all (training-only paths).</summary>
    private static string IsaSuffix()
    {
        string hw = System.Runtime.Intrinsics.X86.Fma.IsSupported ? "fma" :
                    System.Runtime.Intrinsics.X86.Avx2.IsSupported ? "avx2" :
                    System.Runtime.Intrinsics.X86.Sse3.IsSupported ? "sse" : "scalar";
        return $"_serial_{hw}";
    }
}
