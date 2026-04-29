using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Model;
/// <summary>
/// Assembles a <see cref="Transformer"/> from a <see cref="ModelConfig"/>
/// and a <see cref="SharpMindConfig"/>.
///
/// All JigSaw-assembled layers (attention, FFN, norm) are created here
/// using the unified mapping from <see cref="SharpMindConfig.ToJigSawMapping"/>.
/// JigSaw caches assembled types — creating two models with the same config
/// reuses the compiled types for free.
///
/// Usage:
/// <code>
/// var model = ModelFactory.Create(ModelConfig.Llama3_8B, SharpMindConfig.Llama);
/// </code>
/// </summary>
public static class ModelFactory
{
    public static Transformer Create(ModelConfig modelConfig, SharpMindConfig sharpConfig)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();

        var mapping = sharpConfig.ToJigSawMapping();

        // ── Assembled singleton ops ───────────────────────────────────────
        var acts = Assembler.CreateInstance<ActivationOps>(mapping);
        var ops = TensorOpsFactory.Create(sharpConfig);

        // ── Embedding table ───────────────────────────────────────────────
        var embedding = new EmbeddingTable(modelConfig.VocabSize, modelConfig.HiddenDim);
        embedding.InitNormal(std: 0.02f);

        // ── Transformer blocks ────────────────────────────────────────────
        var blocks = Enumerable.Range(0, modelConfig.NumLayers)
            .Select(_ => BuildBlock(modelConfig, sharpConfig, mapping, acts, ops));

        // ── Architecture wrapper ──────────────────────────────────────────
        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException(
                                    $"Unknown ArchKind: {sharpConfig.Arch}")
        };

        // ── Final norm ────────────────────────────────────────────────────
        var finalNorm = BuildNorm(modelConfig.HiddenDim, sharpConfig, mapping);

        return new Transformer(modelConfig, embedding, arch, finalNorm, ops);
    }

    // ── Block construction ────────────────────────────────────────────────

    private static TransformerBlock BuildBlock(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        Dictionary<string, string> mapping,
        ActivationOps acts,
        TensorOps ops)
    {
        var attention = Assembler.CreateInstance<AttentionLayer>(mapping, modelConfig);
        var ffn = BuildFfn(modelConfig, sharpConfig, acts, ops);
        var norm1 = BuildNorm(modelConfig.HiddenDim, sharpConfig, mapping);
        var norm2 = BuildNorm(modelConfig.HiddenDim, sharpConfig, mapping);

        return new TransformerBlock(attention, ffn, norm1, norm2, ops);
    }

    // ── FFN construction ──────────────────────────────────────────────────

    private static FfnLayer BuildFfn(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        ActivationOps acts,
        TensorOps ops)
        => sharpConfig.Ffn switch
        {
            FfnKind.Dense => new DenseFfnLayer(modelConfig, acts, ops),
            FfnKind.Gated => new GatedFfnLayer(modelConfig, acts, ops),
            FfnKind.MoE => new MoEFfnLayer(modelConfig, acts, ops),
            _ => throw new NotSupportedException($"Unknown FfnKind: {sharpConfig.Ffn}")
        };

    // ── Norm construction ─────────────────────────────────────────────────

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig, Dictionary<string, string> mapping)
        => sharpConfig.Norm switch
        {
            NormKind.RMSNorm => (NormLayer)Activator.CreateInstance(
                                     Assembler.Assemble<RmsNormLayer>(mapping), dim)!,
            NormKind.LayerNorm => (NormLayer)Activator.CreateInstance(
                                     Assembler.Assemble<LayerNormLayer>(mapping), dim)!,
            _ => throw new NotSupportedException(
                                     $"Unknown NormKind: {sharpConfig.Norm}")
        };

}
