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

        return builder;
    }
}
