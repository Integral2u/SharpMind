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
            .Select(_ => BuildBlock(modelConfig, sharpConfig, mapping, acts, ops));

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(modelConfig.HiddenDim, sharpConfig);

        return new Transformer(modelConfig, embedding, arch, finalNorm, ops);
    }

    public static Transformer CreateSafe(
        ModelConfig config, 
        SharpMindConfig sharpConfig, 
        SharpMind.Tokenization.Tokenizer tokenizer, 
        int trainingSeqLen = 128)
    {
        bool vocabMismatch = config.VocabSize < tokenizer.VocabSize;
        bool seqMismatch = config.MaxSeqLen < trainingSeqLen;

        if (vocabMismatch || seqMismatch)
        {
            Console.WriteLine("[ModelFactory] Config mismatch - auto-adjusting...");
            if (vocabMismatch) 
                Console.WriteLine($"  VocabSize: {config.VocabSize} -> {tokenizer.VocabSize}");
            if (seqMismatch) 
                Console.WriteLine($"  MaxSeqLen: {config.MaxSeqLen} -> {trainingSeqLen}");

            config = config with 
            {
                VocabSize = Math.Max(config.VocabSize, tokenizer.VocabSize),
                MaxSeqLen = Math.Max(config.MaxSeqLen, trainingSeqLen)
            };
        }

        return Create(config, sharpConfig);
    }

    private static TransformerBlock BuildBlock(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        Dictionary<string, string> mapping,
        ActivationOps acts,
        TensorOps ops)
    {
        var attention = Assembler.CreateInstance<AttentionLayer>(mapping, modelConfig);
        var ffn = BuildFfn(modelConfig, sharpConfig, acts, ops);
        var norm1 = BuildNorm(modelConfig.HiddenDim, sharpConfig);
        var norm2 = BuildNorm(modelConfig.HiddenDim, sharpConfig);

        return new TransformerBlock(attention, ffn, norm1, norm2, ops);
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

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig)
    {
        return sharpConfig.Norm switch
        {
            NormKind.RMSNorm => new RmsNormLayer(dim),
            NormKind.LayerNorm => new LayerNormLayer(dim),
            _ => throw new NotSupportedException($"Unknown NormKind: {sharpConfig.Norm}")
        };
    }
}