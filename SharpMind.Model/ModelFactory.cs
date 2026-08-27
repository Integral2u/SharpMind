using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Encoders;
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
        return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, null!,
            positionEmbedding: AllocatePositionEmbedding(modelConfig));
    }

    /// <summary>Creates a <see cref="TransformerWeights"/> instance with a
    /// <see cref="GgufLoader"/> for the given path. Call <c>weights.InitializeWeights(progress)</c>
    /// after creation to populate weights from the GGUF file.</summary>
    /// <param name="quantizedResident">
    /// Skip allocating the per-layer F32 attention/FFN weights and keep only the
    /// raw quantized bytes, which is all <see cref="InferenceLinearLayer"/>'s
    /// forward reads. Roughly halves resident memory for a chat/inference load.
    /// Leave false when the float tensors are needed after loading — SMM export
    /// and format conversion read them back.
    /// </param>
    public static TransformerWeights CreateWeights(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        QuantizationOps qOps,
        string path,
        LoadMode loadMode = LoadMode.Full,
        bool quantizedResident = false, 
        bool useSafeIo = false)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();
        ThrowIfArchitectureUnsupported(modelConfig.Architecture);
        var fmt = ModelFormatHelpers.GetFormatForExtension(path) ?? throw new FileLoadException($"File type not supported: {path}", path);
        var embedding = new Tensor<float>(modelConfig.VocabSize, modelConfig.HiddenDim);
        Tensor<float>? lmHead = null;
        var finalNormW = Tensor<float>.Ones(modelConfig.HiddenDim);
        Tensor<float>? finalNormB = null;

        // Streaming has always left the 2D weights null; quantizedResident opts the
        // Full path into the same thing. CreateTransformer builds
        // InferenceLinearLayer, whose forward reads the raw quantized bytes and
        // never touches the dequantized F32 duplicates, so for pure inference they
        // cost ~4x the file size for nothing (3.57 GB before a byte was read, on a
        // 0.49 GB model). It is opt-in because reading weights back as floats is a
        // real use — SMM export and the conversion round-trip do exactly that.
        var blockWeights = loadMode == LoadMode.Full && !quantizedResident
            ? AllocateBlockWeights(modelConfig, sharpConfig)
            : AllocateInferenceBlockWeights(modelConfig);

        var loader = ModelFormatHelpers.GetModelLoaderFor((ModelFormat)fmt, qOps, path, modelConfig, useSafeIo);

        if (loadMode == LoadMode.Full)
            return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader,
                positionEmbedding: AllocatePositionEmbedding(modelConfig));
        else
            return new TransformerWeightsStreaming(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader,
                positionEmbedding: AllocatePositionEmbedding(modelConfig)) { GgufPath = path };
    }

    private static Tensor<float>? AllocatePositionEmbedding(ModelConfig config)
        => config.PositionalEncoding == Config.PositionalEncoding.Learned
            ? new Tensor<float>(config.MaxSeqLen, config.HiddenDim)
            : null;

    /// <summary>
    /// Architectures known NOT to work, with the reason. A denylist rather than an
    /// allowlist on purpose: only entries actually observed to fail belong here, so
    /// this can never reject a model that would have loaded.
    ///
    /// Without it these fail late and misleadingly — the generic decoder derives
    /// layer shapes from the config, those shapes disagree with the file, and the
    /// first matmul reports a byte-count mismatch as if the file were corrupt.
    /// </summary>
    private static readonly Dictionary<string, string> UnsupportedArchitectures = new(StringComparer.OrdinalIgnoreCase)
    {
        // Verified against gemma-4-{E2B,e4b,26B-A4B}-it-Q4_K_M. The family is
        // gemma-3n-style and departs from a standard decoder in several ways at
        // once, so correcting any single one only moves the failure along.
        ["gemma4"] = "per-layer input embeddings (per_layer_token_embd, inp_gate, proj), "
                   + "per-layer FFN widths (feed_forward_length is an array), "
                   + "KV sharing across layers (attention.shared_kv_layers), and "
                   + "per-layer output scaling",
        // Same family under the name llama.cpp uses for the 3n conversions.
        ["gemma3n"] = "per-layer input embeddings, per-layer FFN widths and shared KV layers "
                    + "(same architecture family as gemma4)",
    };

    /// <summary>
    /// Fails a model whose architecture is known not to work, before anything is
    /// allocated, so the message names the architecture instead of a tensor size.
    /// </summary>
    private static void ThrowIfArchitectureUnsupported(string? architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture)) return;
        if (!UnsupportedArchitectures.TryGetValue(architecture, out string? reason)) return;

        throw new NotSupportedException(
            $"Model architecture '{architecture}' is not supported. It uses {reason}, " +
            "none of which the decoder implements. The file is fine — SharpMind cannot run it. " +
            "Loading it anyway would derive the wrong layer shapes and produce garbage.");
    }

    /// <summary>
    /// Per-block tensors for inference: norms and biases only. The 2D weights are
    /// left null so the quantized forward reads the raw bytes directly instead of
    /// a dequantized F32 duplicate.
    /// </summary>
    private static TransformerWeights.BlockWeights[] AllocateInferenceBlockWeights(ModelConfig config)
    {
        int qDim = config.NumHeads * config.HeadDim;
        int kvDim = config.NumKvHeads * config.HeadDim;
        var blocks = new TransformerWeights.BlockWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            blocks[i] = new TransformerWeights.BlockWeights
            {
                Norm1W = new Tensor<float>(config.HiddenDim),
                Norm2W = new Tensor<float>(config.HiddenDim),
                WqBias = new Tensor<float>(qDim),
                WkBias = new Tensor<float>(kvDim),
                WvBias = new Tensor<float>(kvDim),
                WoBias = new Tensor<float>(config.HiddenDim)
            };
        }
        return blocks;
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
                Tensor<float>.Ones(config.HiddenDim),                 // Norm1W
                null,                                                 // Norm1B
                Tensor<float>.Ones(config.HiddenDim),                 // Norm2W
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
            ArchKind.Decoder => new DecoderArch(blocks, weights.Config.SlidingWindowSize),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        // Wire up streaming weight loading if applicable
        if (weights is TransformerWeightsStreaming sw && arch is DecoderArch da)
        {
            sw.BlockRefs = blocks;

            // Preload layer 0 in the background after BlockRefs is set,
            // so the first forward pass doesn't wait for synchronous I/O.
            // Pushing into LinearLayers happens later in EnsureLayerLoadedSync.
            sw.PreloadLayerAsync(0);

            da.BeforeBlock = (layerIndex) =>
            {
                // Ensure current layer is loaded (wait for async preload if needed)
                sw.EnsureLayerLoadedSync(layerIndex);

                // Free the layer two steps behind (keep current + next resident)
                if (layerIndex > 0)
                    sw.FreeLayer(layerIndex - 1);

                // Fire async preload for the next layer (overlaps I/O with compute)
                if (layerIndex + 1 < blocks.Length)
                    sw.PreloadLayerAsync(layerIndex + 1);
            };
        }

        // Free float LM head in streaming mode — quantized projection uses RawLmHead
        // and never accesses the float tensor. Avoiding this ~544 MB allocation
        // is the single biggest memory reduction after block float weights.
        if (weights is TransformerWeightsStreaming && weights.LmHeadWeight != null)
        {
            weights.LmHeadWeight.Dispose();
            weights.SetLmHead(new Tensor<float>(1, 1)); // tiny dummy placeholder
        }

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        bool gemmaScale = sharpConfig.Activation == ActivationKind.GELU && sharpConfig.Gate == GateKind.GeGLU;
        var (visionEncoder, audioEncoder) = BuildEncoders(weights.Config);
        var transformer = new Transformer(weights, embedding, arch, finalNorm, weights.LmHeadWeight, qOps, fullMapping, gemmaEmbeddingScale: gemmaScale, visionEncoder: visionEncoder, audioEncoder: audioEncoder);

        // Free pre-allocated (zero-filled) float tensors from BuildBlock.
        // In streaming mode the cyclic load/unload manages raw quantized data;
        // the float tensors are never used (quantized forward uses RawQuantizedData).
        // In Full mode they were populated by LoadAllWeights and are no longer
        // needed once raw quantized data is available.
        if (optimizeMemory)
            transformer.FreeFloatWeights();
        return transformer;         
    }

    /// <summary>
    /// Creates a transformer whose linear layers are float <see cref="TrainingLinearLayer"/>s,
    /// suitable for training forward/backward. Unlike <see cref="CreateTransformer"/>, this
    /// never wires quantized inference layers (which require raw quantized data), never frees
    /// float weight memory, and does not support streaming weight loading.
    /// </summary>
    public static Transformer CreateTrainingTransformer(TransformerWeights weights, SharpMindConfig sharpConfig)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(sharpConfig);

        var fullMapping = sharpConfig.ToJigSawMapping();
        var acts = ActivationFactory.Create(sharpConfig, fullMapping);
        var qOps = QuantizationFactory.Create(fullMapping);
        NormOpsFactory.SetDefault(sharpConfig);

        var embedding = new EmbeddingTable(weights.Config.VocabSize, weights.Config.HiddenDim, weights.EmbeddingWeight, false);
        var blocks = Enumerable.Range(0, weights.Config.NumLayers)
            .Select(i => BuildBlock(i, weights, sharpConfig, null, acts, qOps, useHooks: false, floatLayers: true)).ToArray();

        IArchitecture arch = sharpConfig.Arch switch
        {
            ArchKind.Decoder => new DecoderArch(blocks, weights.Config.SlidingWindowSize),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        bool gemmaScale = sharpConfig.Activation == ActivationKind.GELU && sharpConfig.Gate == GateKind.GeGLU;
        var (visionEncoder, audioEncoder) = BuildEncoders(weights.Config);
        return new Transformer(weights, embedding, arch, finalNorm, weights.LmHeadWeight, qOps, fullMapping, gemmaEmbeddingScale: gemmaScale, visionEncoder: visionEncoder, audioEncoder: audioEncoder);
    }

    /// <summary>Builds the multimodal encoders declared by the model config (null when absent).</summary>
    private static (VisionEncoder? Vision, AudioEncoder? Audio) BuildEncoders(ModelConfig config)
    {
        VisionEncoder? vision = config.HasVision ? new VisionEncoder(config) : null;
        AudioEncoder? audio = config.HasAudio ? new AudioEncoder(config) : null;
        return (vision, audio);
    }

    private static TransformerBlock BuildBlock(
        int layerIdx,
        TransformerWeights weights,
        SharpMindConfig sharpConfig,
        Dictionary<string, string>? overrides,
        ActivationOps acts,
        QuantizationOps qOps,
        bool useHooks = false,
        bool floatLayers = false)
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
        // The assembled type still resolves attention/FFN kernels from cfg. When
        // floatLayers is set (training), the layer constructors receive a null
        // mapping so they build float TrainingLinearLayers (forward/backward);
        // otherwise they receive the resolved cfg and build quantized inference
        // layers exactly as before.
        var layerMapping = floatLayers ? null : cfg;
        var blockWeights = weights.Blocks[layerIdx];
        var t = _attnCache.GetOrAdd(MappingHash.Compute(cfg), (h) =>
        {
            return Assembler.Assemble<AttentionLayer>(cfg);
        });
        var attn = Activator.CreateInstance(t, weights.Config, blockWeights, layerMapping) as AttentionLayer;
        ArgumentNullException.ThrowIfNull(attn);
        attn.SetWeights(blockWeights);
        
        var ffn = BuildFfn(layerIdx, weights, sharpConfig, acts, qOps, layerMapping);
        ffn.SetWeights(blockWeights);
        
        float eps = weights.Config.NormEps;

        // Norm tensors may be null in streaming/Cached mode. When null,
        // NormLayer creates and owns a placeholder Ones tensor internally.
        var norm1 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.Norm1W, blockWeights.Norm1B);
        var norm2 = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.Norm2W, blockWeights.Norm2B);

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

    private static NormLayer BuildNorm(int dim, SharpMindConfig sharpConfig, float eps, Tensor<float>? w, Tensor<float>? b) => sharpConfig.Norm switch
    {
        NormKind.RMSNorm => new RmsNormLayer(dim, eps, w, b),
        NormKind.LayerNorm => new LayerNormLayer(dim, eps, w, b),
        _ => throw new NotSupportedException($"Unknown NormKind: {sharpConfig.Norm}")
    };
}
