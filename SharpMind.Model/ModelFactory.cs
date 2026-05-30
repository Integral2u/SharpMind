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

public static class ModelFactory
{
    public static Transformer Create(ModelConfig modelConfig, SharpMindConfig sharpConfig)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();

        var mapping = sharpConfig.ToJigSawMapping();

        var acts = Assembler.CreateInstance<ActivationOps>(mapping);
        var ops = TensorOpsFactory.Create(sharpConfig);

        var embedding = new EmbeddingTable(modelConfig.VocabSize, modelConfig.HiddenDim);
        embedding.InitNormal(std: 0.02f);

        var blocks = Enumerable.Range(0, modelConfig.NumLayers)
            .Select(i => BuildBlock(i, modelConfig, sharpConfig, mapping, acts, ops));

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(modelConfig.HiddenDim, sharpConfig, modelConfig.NormEps);

        return new Transformer(modelConfig, embedding, arch, finalNorm, ops);
    }

    private static TransformerBlock BuildBlock(
        int layerIdx,
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        Dictionary<string, string> mapping,
        ActivationOps acts,
        TensorOps ops)
    {
        var attention = Assembler.CreateInstance<AttentionLayer>(mapping, modelConfig);
        var ffn = BuildFfn(modelConfig, sharpConfig, acts, ops);
        float eps = modelConfig.NormEps;
        var norm1 = BuildNorm(modelConfig.HiddenDim, sharpConfig, eps);
        var norm2 = BuildNorm(modelConfig.HiddenDim, sharpConfig, eps);

        return new TransformerBlock(layerIdx, attention, ffn, norm1, norm2, ops);
    }

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

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig, float eps = 1e-5f)
    {
        return sharpConfig.Norm switch
        {
            NormKind.RMSNorm => new RmsNormLayer(dim, eps),
            NormKind.LayerNorm => new LayerNormLayer(dim, eps),
            _ => throw new NotSupportedException($"Unknown NormKind: {sharpConfig.Norm}")
        };
    }
}