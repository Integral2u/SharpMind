using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;


namespace SharpMind.Model;

public static class ModelFactory
{
    public static TransformerWeights CreateWeights(ModelConfig modelConfig, SharpMindConfig sharpConfig)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();

        var embedding = new Tensor<float>(modelConfig.VocabSize, modelConfig.HiddenDim);
        Tensor<float>? lmHead = null;
        var finalNormW = Tensor<float>.Ones(modelConfig.HiddenDim);
        Tensor<float>? finalNormB = null; // Default RMSNorm has no bias

        var blockWeights = new TransformerWeights.BlockWeights[modelConfig.NumLayers];
        int ffnDim = modelConfig.FfnDim;
        int wf1Dim = sharpConfig.Ffn == FfnKind.Gated ? 2 * ffnDim : ffnDim;

        for (int i = 0; i < modelConfig.NumLayers; i++)
        {
            int kvDim = modelConfig.NumKvHeads * modelConfig.HeadDim;
            blockWeights[i] = new TransformerWeights.BlockWeights(
                new Tensor<float>(modelConfig.HiddenDim, modelConfig.HiddenDim), // Wq
                new Tensor<float>(modelConfig.HiddenDim, kvDim),               // Wk
                new Tensor<float>(modelConfig.HiddenDim, kvDim),               // Wv
                new Tensor<float>(modelConfig.HiddenDim, modelConfig.HiddenDim), // Wo
                new Tensor<float>(modelConfig.HiddenDim),                       // WqB
                new Tensor<float>(kvDim),                                      // WkB
                new Tensor<float>(kvDim),                                      // WvB
                new Tensor<float>(modelConfig.HiddenDim),                       // WoB
                new Tensor<float>(modelConfig.HiddenDim, wf1Dim),              // Wf1
                new Tensor<float>(ffnDim, modelConfig.HiddenDim),              // Wf2
                new Tensor<float>(wf1Dim),                                     // Wf1B
                new Tensor<float>(modelConfig.HiddenDim),                       // Wf2B
                new Tensor<float>(modelConfig.HiddenDim),                       // Norm1W
                null,                                                           // Norm1B
                new Tensor<float>(modelConfig.HiddenDim),                       // Norm2W
                null                                                            // Norm2B
            );
        }
        return new TransformerWeights(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights);
    }

    public static Transformer CreateSession(TransformerWeights weights, SharpMindConfig sharpConfig, Dictionary<string, string>? mapping = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        var actualMapping = mapping ?? sharpConfig.ToJigSawMapping();
        
        var acts = Assembler.CreateInstance<ActivationOps>(actualMapping);
        var ops = TensorOpsFactory.Create(sharpConfig);
        var qOps = QuantizationFactory.CreateForSystem();

        var embedding = new EmbeddingTable(weights.Config.VocabSize, weights.Config.HiddenDim, weights.EmbeddingWeight, false);
        
        var blocks = Enumerable.Range(0, weights.Config.NumLayers)
            .Select(i => BuildBlock(i, weights, sharpConfig, actualMapping, acts, ops, qOps)).ToArray();

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        return new Transformer(weights, embedding, arch, finalNorm, ops, weights.LmHeadWeight);
    }

    private static TransformerBlock BuildBlock(
        int layerIdx,
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        Dictionary<string, string> mapping,
        ActivationOps acts,
        TensorOps ops,
        QuantizationOps qOps)
    {
        var blockWeights = weights.Blocks[layerIdx];
        
        var attn = Assembler.CreateInstance<AttentionLayer>(mapping, weights.Config, qOps);
        attn.SetWeights(blockWeights);
        
        var ffn = BuildFfn(weights, sharpConfig, acts, ops, qOps);
        ffn.SetWeights(blockWeights);
        
        float eps = weights.Config.NormEps;
        var norm1 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.Norm1W, blockWeights.Norm1B);
        var norm2 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.Norm2W, blockWeights.Norm2B);

        return new TransformerBlock(layerIdx, attn, ffn, norm1, norm2, ops);
    }

    private static FfnLayer BuildFfn(
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        ActivationOps acts,
        TensorOps ops,
        QuantizationOps qOps)
    {
        var blockWeights = weights.Blocks[0]; // Weight shapes are same across blocks for FFN
        return sharpConfig.Ffn switch
        {
            FfnKind.Dense => new DenseFfnLayer(weights.Config, acts, ops, qOps, blockWeights),
            FfnKind.Gated => new GatedFfnLayer(weights.Config, acts, ops, qOps, blockWeights),
            FfnKind.MoE => new MoEFfnLayer(weights.Config, acts, ops, qOps, blockWeights),
            _ => throw new NotSupportedException($"Unknown FfnKind: {sharpConfig.Ffn}")
        };
    }

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig, float eps, Tensor<float> w, Tensor<float>? b)
    {
        return sharpConfig.Norm switch
        {
            NormKind.RMSNorm => new RmsNormLayer(dim, eps, w, b),
            NormKind.LayerNorm => new LayerNormLayer(dim, eps, w, b),
            _ => throw new NotSupportedException($"Unknown NormKind: {sharpConfig.Norm}")
        };
    }
}