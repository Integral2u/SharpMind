using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;
using System.Collections.Concurrent;

namespace SharpMind.Model;

public static class ModelFactory
{
    private static readonly ConcurrentDictionary<int, Type> _attnCache = [];

    /// <summary>Creates empty trainable weights (no GGUF loader). All float tensors
    /// are allocated and will be populated by training. Does NOT call
    /// <see cref="TransformerWeights.InitializeWeights"/>.</summary>
    public static TransformerWeights CreateForTraining(ModelConfig modelConfig, SharpMindConfig sharpConfig)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();

        var embedding = new Tensor<float>(modelConfig.VocabSize, modelConfig.HiddenDim);
        Tensor<float>? lmHead = null;
        var finalNormW = Tensor<float>.Ones(modelConfig.HiddenDim);
        Tensor<float>? finalNormB = null;
        var blockWeights = AllocateBlockWeights(modelConfig, sharpConfig);
        return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, null!);
    }

    /// <summary>Creates a <see cref="TransformerWeights"/> instance with a
    /// <see cref="GgufLoader"/> for the given path. Call <c>weights.InitializeWeights(progress)</c>
    /// after creation to populate weights from the GGUF file.</summary>
    public static TransformerWeights CreateWeights(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        QuantizationOps qOps,
        string path)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();
        var fmt = ModelFormatHelpers.GetFormatForExtension(path);
        if (fmt == null) throw new FileLoadException($"File type not supported: {path}", path);

        var embedding = new Tensor<float>(modelConfig.VocabSize, modelConfig.HiddenDim);
        Tensor<float>? lmHead = null;
        var finalNormW = Tensor<float>.Ones(modelConfig.HiddenDim);
        Tensor<float>? finalNormB = null;

        var blockWeights = AllocateBlockWeights(modelConfig, sharpConfig);
        var loader = ModelFormatHelpers.GetModelLoaderFor((ModelFormat)fmt, qOps, path, modelConfig);

        return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader);
    }

    private static TransformerWeights.BlockWeights[] AllocateBlockWeights(ModelConfig config, SharpMindConfig sharpConfig)
    {
        int ffnDim = config.FfnDim;
        int wf1Dim = sharpConfig.Ffn == FfnKind.Gated ? 2 * ffnDim : ffnDim;
        var blocks = new TransformerWeights.BlockWeights[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            int qDim = config.NumHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            blocks[i] = new TransformerWeights.BlockWeights(
                new Tensor<float>(config.HiddenDim, qDim),          // Wq
                new Tensor<float>(config.HiddenDim, kvDim),          // Wk
                new Tensor<float>(config.HiddenDim, kvDim),          // Wv
                new Tensor<float>(qDim, config.HiddenDim),           // Wo
                new Tensor<float>(qDim),                              // WqB
                new Tensor<float>(kvDim),                            // WkB
                new Tensor<float>(kvDim),                            // WvB
                new Tensor<float>(config.HiddenDim),                  // WoB
                new Tensor<float>(config.HiddenDim, wf1Dim),         // Wf1
                new Tensor<float>(ffnDim, config.HiddenDim),         // Wf2
                new Tensor<float>(wf1Dim),                            // Wf1B
                new Tensor<float>(config.HiddenDim),                  // Wf2B
                new Tensor<float>(config.HiddenDim),                  // Norm1W
                null,                                                 // Norm1B
                new Tensor<float>(config.HiddenDim),                  // Norm2W
                null,                                                 // Norm2B
                null,                                                 // QNormW
                null,                                                 // KNormW
                null,                                                 // PostNorm1W
                null                                                  // PostNorm2W
            );
        }
        return blocks;
    }

    public static Transformer CreateTransformer(TransformerWeights weights, SharpMindConfig sharpConfig, Dictionary<string, string>? mapping = null, bool optimizeMemory = true)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(sharpConfig);

        var fullMapping = mapping ?? sharpConfig.ToJigSawMapping();
        var acts = ActivationFactory.Create(sharpConfig, fullMapping);
        var qOps = QuantizationFactory.Create(fullMapping);
        NormOpsFactory.SetDefault(sharpConfig);

        var embedding = new EmbeddingTable(weights.Config.VocabSize, weights.Config.HiddenDim, weights.EmbeddingWeight, false);
        var blocks = Enumerable.Range(0, weights.Config.NumLayers)
            .Select(i => BuildBlock(i, weights, sharpConfig, mapping, acts, qOps, sharpConfig.UseHooks)).ToArray();

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        var transformer = new Transformer(weights, embedding, arch, finalNorm, weights.LmHeadWeight, qOps, fullMapping);

        if (optimizeMemory)
            transformer.FreeFloatWeights();
        return transformer;         
    }

    private static TransformerBlock BuildBlock(
        int layerIdx,
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        Dictionary<string, string>? overrides,
        ActivationOps acts,
        QuantizationOps qOps,
        bool useHooks = false)
    {
        var cfg = sharpConfig.ToJigSawMapping();
        if (overrides != null)
        {
            foreach (var m in overrides)
            {
                if (cfg.TryGetValue(m.Key, out string? value)) cfg[m.Key] = value;
                else cfg.Add(m.Key, m.Value);
            }
        }
        var blockWeights = weights.Blocks[layerIdx];
        var t = _attnCache.GetOrAdd(MappingHash.Compute(cfg), (h) =>
        {
            return Assembler.Assemble<AttentionLayer>(cfg);
        });
        var attn = Activator.CreateInstance(t, weights.Config/*, qOps*/, blockWeights, cfg) as AttentionLayer;
        ArgumentNullException.ThrowIfNull(attn);
        attn.SetWeights(blockWeights);
        
        var ffn = BuildFfn(layerIdx, weights, sharpConfig, acts, qOps, cfg);
        ffn.SetWeights(blockWeights);
        
        float eps = weights.Config.NormEps;

        // Norm tensors may be null in Cached mode; use placeholder if needed
        var n1w = blockWeights.Norm1W ?? Tensor<float>.Ones(weights.Config.HiddenDim);
        var n2w = blockWeights.Norm2W ?? Tensor<float>.Ones(weights.Config.HiddenDim);
        var norm1 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, n1w, blockWeights.Norm1B);
        var norm2 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, n2w, blockWeights.Norm2B);
        if (blockWeights.Norm1W == null) n1w.Dispose();
        if (blockWeights.Norm2W == null) n2w.Dispose();

        // Post-attention and post-FFN norms (Gemma-3)
        NormLayer? postAttnNorm = null;
        NormLayer? postFfnNorm = null;
        if (blockWeights.PostNorm1W != null)
            postAttnNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.PostNorm1W, null);
        if (blockWeights.PostNorm2W != null)
            postFfnNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.PostNorm2W, null);

        return useHooks
            ? new HookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2, postAttnNorm, postFfnNorm)
            : new UnhookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2, postAttnNorm, postFfnNorm);
    }

    private static FfnLayer BuildFfn(
        int layerIdx,
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        ActivationOps acts,
        QuantizationOps qOps,
        Dictionary<string, string>? cfg = null)
    {
        var blockWeights = weights.Blocks[layerIdx];
        return sharpConfig.Ffn switch
        {
            FfnKind.Dense => new DenseFfnLayer(weights.Config, acts, qOps, blockWeights, cfg),
            FfnKind.Gated => new GatedFfnLayer(weights.Config, acts, qOps, blockWeights, cfg),
            FfnKind.MoE => new MoEFfnLayer(weights.Config, acts, qOps, blockWeights, cfg),
            _ => throw new NotSupportedException($"Unknown FfnKind: {sharpConfig.Ffn}")
        };
    }

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig, float eps, Tensor<float> w, Tensor<float>? b) => sharpConfig.Norm switch
    {
        NormKind.RMSNorm => new RmsNormLayer(dim, eps, w, b),
        NormKind.LayerNorm => new LayerNormLayer(dim, eps, w, b),
        _ => throw new NotSupportedException($"Unknown NormKind: {sharpConfig.Norm}")
    };
}
