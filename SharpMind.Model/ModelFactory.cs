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
        var blockWeights = AllocateBlockWeights(modelConfig, sharpConfig, allocateFloatTensors: true);
        return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, null!);
    }

    /// <summary>Creates a <see cref="TransformerWeights"/> instance (Full or Cached subclass)
    /// with a <see cref="GgufLoader"/> for the given path. Call <c>weights.InitializeWeights(progress)</c>
    /// after creation to populate weights from the GGUF file.</summary>
    public static TransformerWeights Create(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        QuantizationOps qOps,
        string path,
        ModelMetaData meta,
        LoadMode mode,
        int cacheDepth = 2)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();

        bool fullMode = mode == LoadMode.Full;
        var embedding = new Tensor<float>(modelConfig.VocabSize, modelConfig.HiddenDim);
        Tensor<float>? lmHead = null;
        var finalNormW = Tensor<float>.Ones(modelConfig.HiddenDim);
        Tensor<float>? finalNormB = null;

        var blockWeights = AllocateBlockWeights(modelConfig, sharpConfig, fullMode);
        var loader = new GgufLoader(qOps, path, modelConfig);

        if (fullMode)
        {
            return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader);
        }
        else
        {
            return new TransformerWeightsCached(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader, path, meta, cacheDepth);
        }
    }

    private static TransformerWeights.BlockWeights[] AllocateBlockWeights(ModelConfig config, SharpMindConfig sharpConfig, bool allocateFloatTensors)
    {
        int ffnDim = config.FfnDim;
        int wf1Dim = sharpConfig.Ffn == FfnKind.Gated ? 2 * ffnDim : ffnDim;
        var blocks = new TransformerWeights.BlockWeights[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            int qDim = config.NumHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            if (allocateFloatTensors)
            {
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
                    null                                                  // KNormW
                );
            }
            else
            {
                // Pre-allocate small float tensors (norm weights, biases) in cached mode.
                // Large weight tensors (Wq, Wk, Wv, Wo, Wf1, Wf2) remain null and are
                // loaded on-demand.
                blocks[i] = new TransformerWeights.BlockWeights(
                    null,                                 // Wq (large, on-demand)
                    null,                                 // Wk
                    null,                                 // Wv
                    null,                                 // Wo
                    new Tensor<float>(qDim),              // WqB (small)
                    new Tensor<float>(kvDim),             // WkB (small)
                    new Tensor<float>(kvDim),             // WvB (small)
                    new Tensor<float>(config.HiddenDim),  // WoB (small)
                    null,                                 // Wf1 (large, on-demand)
                    null,                                 // Wf2 (large, on-demand)
                    new Tensor<float>(wf1Dim),            // Wf1B (small)
                    new Tensor<float>(config.HiddenDim),  // Wf2B (small)
                    new Tensor<float>(config.HiddenDim),  // Norm1W (small, always loaded)
                    null,                                 // Norm1B
                    new Tensor<float>(config.HiddenDim),  // Norm2W (small, always loaded)
                    null,                                 // Norm2B
                    null,                                 // QNormW
                    null                                  // KNormW
                );
            }
        }
        return blocks;
    }

    public static Transformer CreateSession(TransformerWeights weights, SharpMindConfig sharpConfig, Dictionary<string, string>? mapping = null, bool optimizeMemory = true)
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

        // In Cached mode, subscribe to OnLayerLoaded to push raw data into running layers
        if (weights is TransformerWeightsCached cached)
        {
            cached.OnLayerLoaded += layerIdx =>
            {
                if (layerIdx < blocks.Length)
                    blocks[layerIdx].UpdateRawWeights(weights.Blocks[layerIdx]);
            };
            // Push raw data for layers that were pre-loaded by InitializeWeights
            // before the subscription was set up
            for (int i = 0; i < blocks.Length; i++)
            {
                if (cached.IsLayerLoaded(i))
                    blocks[i].UpdateRawWeights(weights.Blocks[i]);
            }
        }

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks, weights as TransformerWeightsCached),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        var transformer = new Transformer(weights, embedding, arch, finalNorm, weights.LmHeadWeight, qOps);
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
        
        var ffn = BuildFfn(weights, sharpConfig, acts, qOps, cfg);
        ffn.SetWeights(blockWeights);
        
        float eps = weights.Config.NormEps;

        // Norm tensors may be null in Cached mode; use placeholder if needed
        var n1w = blockWeights.Norm1W ?? Tensor<float>.Ones(weights.Config.HiddenDim);
        var n2w = blockWeights.Norm2W ?? Tensor<float>.Ones(weights.Config.HiddenDim);
        var norm1 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, n1w, blockWeights.Norm1B);
        var norm2 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, n2w, blockWeights.Norm2B);
        if (blockWeights.Norm1W == null) n1w.Dispose();
        if (blockWeights.Norm2W == null) n2w.Dispose();

        return useHooks
            ? new HookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2)
            : new UnhookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2);
    }

    private static FfnLayer BuildFfn(
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        ActivationOps acts,
        QuantizationOps qOps,
        Dictionary<string, string>? cfg = null)
    {
        var blockWeights = weights.Blocks[0]; // Weight shapes are same across blocks for FFN
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
