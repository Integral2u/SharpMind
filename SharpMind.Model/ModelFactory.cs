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
using SharpMind.Model.Layers.ShortConv;
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
            positionEmbedding: AllocatePositionEmbedding(modelConfig),
            tokenTypeEmbedding: AllocateTokenTypeEmbedding(modelConfig));
    }

    /// <summary>Creates a <see cref="TransformerWeights"/> instance with a
    /// <see cref="GgufLoader"/> for the given path. Call <c>weights.InitializeWeights(progress)</c>
    /// after creation to populate weights from the GGUF file.</summary>
    /// <param name="quantizedResident">
    /// Skip allocating the per-layer F32 attention/FFN weights and keep only the
    /// raw quantized bytes, which is all <see cref="InferenceLinearLayer"/>'s
    /// forward reads. Roughly halves resident memory for a chat/inference load.
    /// Leave false when the float tensors are needed after loading - SMM export
    /// and format conversion read them back.
    /// </param>
    /// <param name="maxParallelLoadDegree">
    /// Threads a <see cref="LoadMode.Full"/> load may fan out over. 0 (default)
    /// means one per core; higher values cap it, and 1 restores the original
    /// single-threaded file-order loop. The default overlaps the per-tensor reads
    /// and dequantization, partitioned so each thread owns a disjoint set of
    /// transformer blocks -- about 2x faster to load on a 4-core box, at the cost
    /// of peak transient memory scaling with the degree. Pass 1 on a memory-
    /// starved host, or to reproduce sequential behaviour exactly. A single-core
    /// host and safe I/O (WASM) ignore the request and load sequentially;
    /// Streaming loads are unaffected (they already load one layer at a time).
    /// </param>
    public static TransformerWeights CreateWeights(
        ModelConfig modelConfig,
        SharpMindConfig sharpConfig,
        QuantizationOps qOps,
        string path,
        LoadMode loadMode = LoadMode.Full,
        bool quantizedResident = false, 
        bool useSafeIo = false,
        int maxParallelLoadDegree = 0,
        int desiredResidentExperts = 0)
    {
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(sharpConfig);
        modelConfig.Validate();
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
        // real use � SMM export and the conversion round-trip do exactly that.
        var blockWeights = loadMode == LoadMode.Full && !quantizedResident
            ? AllocateBlockWeights(modelConfig, sharpConfig)
            : AllocateInferenceBlockWeights(modelConfig, sharpConfig);

        var loader = ModelFormatHelpers.GetModelLoaderFor((ModelFormat)fmt, qOps, path, modelConfig, useSafeIo, maxParallelLoadDegree);

        if (loadMode == LoadMode.Full)
            return new TransformerWeightsFull(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader,
                positionEmbedding: AllocatePositionEmbedding(modelConfig),
                tokenTypeEmbedding: AllocateTokenTypeEmbedding(modelConfig));
        else
            return new TransformerWeightsStreaming(modelConfig, embedding, lmHead, finalNormW, finalNormB, blockWeights, loader,
                positionEmbedding: AllocatePositionEmbedding(modelConfig),
                tokenTypeEmbedding: AllocateTokenTypeEmbedding(modelConfig))
            {
                // Streaming MoE expert residency: > 0 pins that many most-used routed experts
                // per layer across unloads and streams the rest on demand; the streaming weights
                // clamp it to what memory allows. 0 keeps the existing whole-layer streaming
                // behaviour. Dense models and loaders without expert slicing ignore it.
                GgufPath = path,
                DesiredResidentExperts = Math.Max(0, desiredResidentExperts)
            };
    }

    private static Tensor<float>? AllocatePositionEmbedding(ModelConfig config)
        => config.PositionalEncoding == Config.PositionalEncoding.Learned
            ? new Tensor<float>(config.MaxSeqLen, config.HiddenDim)
            : null;

    private static Tensor<float>? AllocateTokenTypeEmbedding(ModelConfig config)
        => config.NumTokenTypes > 0
            ? new Tensor<float>(config.NumTokenTypes, config.HiddenDim)
            : null;

    /// <summary>
    /// Per-block tensors for inference: norms and biases only. The 2D weights are
    /// left null so the quantized forward reads the raw bytes directly instead of
    /// a dequantized F32 duplicate.
    /// </summary>
    private static TransformerWeights.BlockWeights[] AllocateInferenceBlockWeights(ModelConfig config, SharpMindConfig sharpConfig)
    {
        // LayerNorm carries a bias; allocate placeholder tensors so the loader
        // fills them and BuildNorm wires them into the NormLayer (BERT-family
        // attn_output_norm/layer_output_norm biases depend on this). RMSNorm has
        // no bias, so no tensor is allocated and NormBias stays null.
        bool layerNorm = sharpConfig.Norm == NormKind.LayerNorm;
        // Dense FFNs (BERT, GPT-2) carry ffn_up/ffn_down biases. They have no
        // raw field, so the streaming metadata scan never allocates them and
        // FfnLayer.SetWeights only applied a bias alongside a float weight —
        // a streaming load silently dropped them and diverged from the full
        // load. Allocate placeholders so SetWeights can push the bias alone.
        bool denseFfn = sharpConfig.Ffn == FfnKind.Dense;
        var blocks = new TransformerWeights.BlockWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            // Attention shapes are per-layer (gemma4 uses 256 on SWA layers and
            // 512 on full layers), so the bias placeholders must match the layer.
            int headDim = config.HeadDimForLayer(i);
            int qDim = config.NumHeads * headDim;
            int kvDim = config.NumKvHeads * headDim;
            blocks[i] = new TransformerWeights.BlockWeights
            {
                LayerIndex = i,
                Norm1W = Tensor<float>.Ones(config.HiddenDim),
                Norm1B = layerNorm ? new Tensor<float>(config.HiddenDim) : null,
                Norm2W = Tensor<float>.Ones(config.HiddenDim),
                Norm2B = layerNorm ? new Tensor<float>(config.HiddenDim) : null,
                WqBias = new Tensor<float>(qDim),
                WkBias = new Tensor<float>(kvDim),
                WvBias = new Tensor<float>(kvDim),
                WoBias = new Tensor<float>(config.HiddenDim),
                Wf1Bias = denseFfn ? new Tensor<float>(config.FfnDim) : null,
                Wf2Bias = denseFfn ? new Tensor<float>(config.HiddenDim) : null,
                // Gemma-4 PLE block tensors. Pre-allocate so BuildBlock wires the PLE
                // layers before any weight data is read; the loader fills these in place
                // and ReleaseLayerData keeps them resident (like the gemma post-norms).
                PerLayerInpGate = config.HasPerLayerEmbeddings ? new Tensor<float>(config.HiddenDim, config.PerLayerEmbdDim) : null,
                PerLayerProj = config.HasPerLayerEmbeddings ? new Tensor<float>(config.PerLayerEmbdDim, config.HiddenDim) : null,
                PerLayerPostNorm = config.HasPerLayerEmbeddings ? new Tensor<float>(config.HiddenDim) : null,
                OutScale = config.HasPerLayerEmbeddings ? new Tensor<float>(1) : null
            };
        }
        return blocks;
    }

    private static TransformerWeights.BlockWeights[] AllocateBlockWeights(ModelConfig config, SharpMindConfig sharpConfig)
    {
        int ffnDim = config.FfnDim;
        int wf1Dim = sharpConfig.Ffn == FfnKind.Gated ? 2 * ffnDim : ffnDim;
        bool layerNorm = sharpConfig.Norm == NormKind.LayerNorm;
        var blocks = new TransformerWeights.BlockWeights[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            // Attention shapes are per-layer (gemma4: 256 on SWA layers, 512 on
            // full layers), so a single global head dim mis-sizes the F32 planes
            // on the full-load path.
            int headDim = config.HeadDimForLayer(i);
            int qDim = config.NumHeads * headDim;
            int kvDim = config.NumKvHeads * headDim;

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
                layerNorm ? new Tensor<float>(config.HiddenDim) : null, // Norm1B
                Tensor<float>.Ones(config.HiddenDim),                 // Norm2W
                layerNorm ? new Tensor<float>(config.HiddenDim) : null, // Norm2B
                null,                                                 // QNormW
                null,                                                 // KNormW
                null,                                                 // PostNorm1W
                null,                                                 // PostNorm2W
                config.HasPerLayerEmbeddings ? new Tensor<float>(config.HiddenDim, config.PerLayerEmbdDim) : null, // PerLayerInpGate
                config.HasPerLayerEmbeddings ? new Tensor<float>(config.PerLayerEmbdDim, config.HiddenDim) : null, // PerLayerProj
                config.HasPerLayerEmbeddings ? new Tensor<float>(config.HiddenDim) : null,                          // PerLayerPostNorm
                config.HasPerLayerEmbeddings ? new Tensor<float>(1) : null                                          // OutScale
            )
            {
                LayerIndex = i,
            };
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
            ArchKind.Decoder => new DecoderArch(blocks, weights.Config),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        // Wire up streaming weight loading if applicable
        if (weights is TransformerWeightsStreaming sw)
        {
            sw.BlockRefs = blocks;

            // Wire MoE expert residency (if configured on the streaming weights) before any
            // preload fires, so both the synchronous and async loads already know to skip the
            // routed experts. No-op when disabled or unsupported.
            sw.AttachExpertResidency(blocks);

            // Preload layer 0 in the background after BlockRefs is set,
            // so the first forward pass doesn't wait for synchronous I/O.
            // Pushing into LinearLayers happens later in EnsureLayerLoadedSync.
            sw.PreloadLayerAsync(0);

            arch.BeforeBlock = (layerIndex) =>
            {
                // Ensure current layer is loaded (wait for async preload if needed)
                sw.EnsureLayerLoadedSync(layerIndex);

                // Keep a resident window of sw.ResidentWindow layers: unload the layer that
                // far behind once the loop is past it, and preload the layer that far ahead.
                if (layerIndex >= sw.ResidentWindow)
                    sw.FreeLayer(layerIndex - sw.ResidentWindow);

                // Fire async preload for the next layer (overlaps I/O with compute)
                if (layerIndex + sw.ResidentWindow < blocks.Length)
                    sw.PreloadLayerAsync(layerIndex + sw.ResidentWindow);
            };
        }

        // Streaming keeps only the raw head bytes: the quantized projection reads
        // RawLmHead and never accesses the float tensor. The head is now created
        // lazily (PR #55), so accessing LmHeadWeight here would dequantize the
        // whole head (~544 MB for Qwen2.5-1.5B) just to dispose it. Free a copy a
        // consumer already materialized, and leave an un-materialized one alone —
        // not materializing in the first place is the entire memory win.
        if (weights is TransformerWeightsStreaming && weights.HasLmHead)
            weights.DisposeMaterializedLmHead();

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        bool gemmaScale = sharpConfig.Activation == ActivationKind.GELU && sharpConfig.Gate == GateKind.GeGLU;

        // Gemma-4 per-layer-embedding builder. The model projection and its norm are
        // F32 globals; the token table stays quantized (raw bytes).
        PerLayerInputs? perLayerInputs = null;
        if (weights.Config.HasPerLayerEmbeddings
            && weights.PerLayerModelProj is not null
            && weights.RawPerLayerTokEmbd is not null
            && weights.RawPerLayerTokEmbdDtype is { } tokDtype)
        {
            var modelProj = LinearLayerFactory.Create(
                "per_layer_model_proj", weights.Config.HiddenDim, weights.Config.PerLayerTotalDim,
                false, weights.PerLayerModelProj, null, QuantDType.F32, baseMapping: null);
            var projNorm = BuildNorm(weights.Config.PerLayerEmbdDim, sharpConfig, weights.Config.NormEps, weights.PerLayerProjNormW, null);
            perLayerInputs = new PerLayerInputs(
                modelProj, projNorm, weights.RawPerLayerTokEmbd, tokDtype, qOps,
                weights.Config.NumLayers, weights.Config.PerLayerEmbdDim, weights.Config.HiddenDim, weights.Config.VocabSize);
        }

        var (visionEncoder, audioEncoder) = BuildEncoders(weights.Config);
        var transformer = new Transformer(weights, embedding, arch, finalNorm, qOps, fullMapping, gemmaEmbeddingScale: gemmaScale, visionEncoder: visionEncoder, audioEncoder: audioEncoder, perLayerInputs: perLayerInputs);

        // Streaming reloads a layer's raw buffers every token, and the Q8_0 wide repack is
        // keyed on the raw array instance, so it would rebuild several MiB of pinned LOH per
        // tensor per reload. Turn it off for streaming; full/resident mode builds the copy
        // once and keeps it, so it stays enabled there.
        if (weights is TransformerWeightsStreaming)
            transformer.SuppressWide();

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
            ArchKind.Decoder => new DecoderArch(blocks, weights.Config),
            ArchKind.Encoder => new EncoderArch(blocks),
            _ => throw new NotSupportedException($"Unknown ArchKind: {sharpConfig.Arch}")
        };

        var finalNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, weights.Config.NormEps, weights.FinalNormWeight, weights.FinalNormBias);

        bool gemmaScale = sharpConfig.Activation == ActivationKind.GELU && sharpConfig.Gate == GateKind.GeGLU;
        var (visionEncoder, audioEncoder) = BuildEncoders(weights.Config);
        return new Transformer(weights, embedding, arch, finalNorm, qOps, fullMapping, gemmaEmbeddingScale: gemmaScale, visionEncoder: visionEncoder, audioEncoder: audioEncoder);
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

        var ffn = BuildFfn(layerIdx, weights, sharpConfig, acts, qOps, layerMapping);
        ffn.SetWeights(blockWeights);

        // Gemma-4 per-layer-embedding tail. The layers are F32, so they are built as
        // float linear layers (null mapping) whose weights the loader holds by reference
        // and refills in place across streaming reloads.
        PerLayerEmbedding? perLayer = null;
        if (weights.Config.HasPerLayerEmbeddings && blockWeights.PerLayerInpGate is not null && blockWeights.PerLayerProj is not null)
        {
            var inpGate = LinearLayerFactory.Create(
                $"blk.{layerIdx}.inp_gate", weights.Config.HiddenDim, weights.Config.PerLayerEmbdDim,
                false, blockWeights.PerLayerInpGate, null, QuantDType.F32, baseMapping: null);
            var proj = LinearLayerFactory.Create(
                $"blk.{layerIdx}.proj", weights.Config.PerLayerEmbdDim, weights.Config.HiddenDim,
                false, blockWeights.PerLayerProj, null, QuantDType.F32, baseMapping: null);
            var perLayerPostNorm = BuildNorm(weights.Config.HiddenDim, sharpConfig, eps, blockWeights.PerLayerPostNorm, null);
            perLayer = new PerLayerEmbedding(layerIdx, inpGate, proj, perLayerPostNorm);
        }

        // LFM2 short-conv blocks have no attention tensors; the ShortConvLayer
        // replaces the KV-cache branch entirely (its state lives in the
        // per-layer ShortConvCache, never in the KVCache).
        if (weights.Config.IsShortConvLayer(layerIdx))
        {
            var shortConv = new ShortConvLayer(weights.Config, qOps, blockWeights, layerMapping);
            shortConv.SetWeights(blockWeights);
            bool shortConvPostNorm = weights.Config.PostNormResidual;
            return useHooks
                ? new HookedTransformerBlock(layerIdx, null, ffn, norm1, norm2, postAttnNorm, postFfnNorm, shortConv, shortConvPostNorm, perLayer, blockWeights.OutScale)
                : new UnhookedTransformerBlock(layerIdx, null, ffn, norm1, norm2, postAttnNorm, postFfnNorm, shortConv, shortConvPostNorm, perLayer, blockWeights.OutScale);
        }

        var t = _attnCache.GetOrAdd(MappingHash.Compute(cfg), (h) =>
        {
            return Assembler.Assemble<AttentionLayer>(cfg);
        });
        var attn = Activator.CreateInstance(t, weights.Config, blockWeights, layerMapping) as AttentionLayer;
        ArgumentNullException.ThrowIfNull(attn);
        attn.SetWeights(blockWeights);

        bool postNormResidual = weights.Config.PostNormResidual;
        return useHooks
            ? new HookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2, postAttnNorm, postFfnNorm, postNormResidual: postNormResidual, perLayer: perLayer, outScale: blockWeights.OutScale)
            : new UnhookedTransformerBlock(layerIdx, attn, ffn, norm1, norm2, postAttnNorm, postFfnNorm, postNormResidual: postNormResidual, perLayer: perLayer, outScale: blockWeights.OutScale);
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
