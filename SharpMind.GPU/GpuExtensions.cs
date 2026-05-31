namespace SharpMind.GPU;

public static class MappingBuilderExtensions
{
    /// <summary>
    /// Augments the mapping with GPU-accelerated kernels.
    /// This should be called after <see cref="MappingBuilder.ApplyPreset"/>.
    /// </summary>
    public static MappingBuilder WithGpu(this MappingBuilder builder)
    {
        // Override standard keys with GPU equivalents
        builder.Override(SharpMindConfig.KeyPointWise, "gpu_pointwise");
        builder.Override(SharpMindConfig.KeyGate, "gpu_gate");
        builder.Override(SharpMindConfig.KeySoftmax, "gpu_softmax");
        builder.Override(SharpMindConfig.KeyRMSNorm, "gpu_rmsnorm");
        builder.Override(SharpMindConfig.KeyMatMul, "gpu_matmul");
        builder.Override(SharpMindConfig.KeyAttention, "gpu_attention");
        builder.Override(SharpMindConfig.KeyFfn, "gpu_ffn");
        builder.Override(SharpMindConfig.KeyNorm, "gpu_norm");
        builder.Override(SharpMindConfig.KeyArch, "gpu_arch");
        builder.Override(SharpMindConfig.KeyAdamW, "gpu_adamw");
        builder.Override(SharpMindConfig.KeyGradNorm, "gpu_gradnorm");

        builder.Override("vecdot_q3k", "q3k_gpu");
        builder.Override("vecdot_q4k", "q4k_gpu");
        builder.Override("vecdot_q5k", "q5k_gpu");
        builder.Override("vecdot_q6k", "q6k_gpu");
        builder.Override("vecdot_q8_0", "q8_0_gpu");
        builder.Override("vecdot_q4_0", "q4_0_gpu");
        builder.Override("vecdot_q4_1", "q4_1_gpu");
        builder.Override("vecdot_q5_0", "q5_0_gpu");
        builder.Override("vecdot_q5_1", "q5_1_gpu");
        builder.Override("vecdot_q8_1", "q8_1_gpu");
        builder.Override("vecdot_q2k", "q2k_gpu");
        builder.Override("vecdot_q8k", "q8k_gpu");
        builder.Override("hsum256", "hsum_gpu");
        builder.Override("halftofloat", "halftofloat_gpu");
        builder.Override("getscalemink4_scale", "getscalemink4_scale_gpu");
        builder.Override("getscalemink4_min", "getscalemink4_min_gpu");

        return builder;
    }
}
