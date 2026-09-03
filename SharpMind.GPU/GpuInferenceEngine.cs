using ILGPU;
using SharpMind.Core;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.GPU;

/// <summary>
/// GPU inference engine: the whole forward for a conversation's autoregressive pass runs on the
/// device. The first prefill of a conversation (empty cache) runs the prompt forward once, and
/// every subsequent step — a continued turn's incremental prefill and each single-token
/// <see cref="DecodeStep"/> — runs the new tokens forward on the device too, reading and
/// appending a persistent device K/V cache (<see cref="_kCache"/>/<see cref="_vCache"/>) rather
/// than replaying the prefix. Decode is KV-cache-efficient: one fresh query row at absolute
/// position <c>c</c> attends only the <c>c+1</c> cached rows (O(S) per token, no O(S²) replay).
///
/// The forward reuses <see cref="GpuBackpropEngine"/>'s op sequence (embed → per-block
/// norm/QKV/RoPE/attention/FFN → final norm → LM head) and is genuinely correct and tested the
/// same way (<c>GpuTestDevice</c>, no real hardware required). The positioned attention is the
/// flash-style <see cref="Kernels.FlashAttentionKernels.FwdKvLen"/> (independent query vs.
/// key/value length with a position offset), so no probabilities are materialised and continued
/// prefill / decode share one kernel with <c>pos0</c> = the cache length at that point.
///
/// The host <see cref="IKVCache"/>[] (<see cref="_cpuCaches"/>) is kept as a mirror that each
/// GPU step appends to. It remains the source of <see cref="CachedLength"/> and of the
/// trim/reset/snapshot/import operations, and is what falls back to the CPU path uses, so the
/// engine's multi-turn and session contracts are unchanged; the device cache is the authority
/// the GPU attention actually reads.
///
/// <see cref="_maxPromptTokens"/> bounds the arena for one GPU <see cref="Prefill"/> call's new
/// tokens: a fresh prompt longer than it, or an over-long conversation step, falls back to the
/// CPU <see cref="Transformer"/> instead of throwing. The persistent device KV cache is sized by
/// <see cref="MaxCacheLength"/>, so continued prefill and decode stay on the device for any
/// context up to that cache cap — the two gates are independent and over-long conversations always
/// work, just without GPU acceleration past the cap.
///
/// Model constraints mirror <see cref="GpuBackpropEngine.ValidateSupported"/> minus the
/// LoRA/trainable-parameter checks, which don't apply here: RMSNorm final norm and block
/// norms, RoPE or NoPE, gated FFN, no MoE. Quantized GGUF weights run on the device for the
/// dtypes in <see cref="SupportedDtypes"/>; a block linear in any other quant (e.g. Q8_K) falls
/// back per-tensor to the host and the rest of the model stays on the device.
/// </summary>
public sealed class GpuInferenceEngine : IInferenceEngine
{
    private readonly GpuDevice _dev;
    private readonly Transformer _model;
    private readonly ModelConfig _cfg;
    private readonly bool _gelu, _gemmaScale, _rope, _neox;
    private readonly int _ropeDim;
    private readonly int _maxPromptTokens;
    private readonly GpuBlock[] _blocks;
    private readonly DeviceBuffer? _embedding;
    private readonly DeviceBuffer _finalNormW;
    private readonly DeviceByteBuffer? _embeddingRaw;
    private readonly QuantDType? _embeddingDtype;
    private readonly DeviceBuffer? _cos, _sin;

    /// <summary>
    /// Persistent per-layer device K/V storage, [MaxCacheLength, NumKvHeads·HeadDim]. Written by
    /// every GPU forward (prefill, continued prefill, decode) and read by the positioned GPU
    /// attention <see cref="GpuKernels.AttnFwdKvLen"/> — this is the cache the device attention
    /// actually uses. Row <c>p</c> is position <c>p</c>'s post-RoPE K/V, so the live length is
    /// always <see cref="_deviceCacheLength"/>.
    /// </summary>
    private readonly DeviceBuffer[] _kCache, _vCache;

    /// <summary>Number of valid rows in <see cref="_kCache"/>/<see cref="_vCache"/>. Kept equal to
    /// <see cref="CachedLength"/> after every GPU forward; a host cache mutation that changes which
    /// positions are valid (trim/import) re-syncs the device cache from the host so the two never
    /// diverge when the next GPU step runs.</summary>
    private int _deviceCacheLength;

    /// <summary>
    /// Host CPU KV caches, one per layer. Each GPU forward appends its new rows here too, so the
    /// host stays a mirror of the device cache: <see cref="CachedLength"/> and the
    /// trim/reset/snapshot/import operations all forward here, and the CPU fallback path (over-long
    /// or continued prompts) reads them. The device cache remains the authority for GPU attention.
    /// </summary>
    private readonly IKVCache[] _cpuCaches;

    /// <summary>Scratch arena for one GPU Prefill call's activations. Sized for <see cref="_maxPromptTokens"/> at construction — see the class doc about chunking (long/continued prompts go to the CPU path).</summary>
    private readonly DeviceArena _arena;

    private readonly IWorkspace _cpuWorkspace;
    private readonly float[] _logitsHost;
    private readonly int[] _hostIds;
    private bool _disposed;

    public int CachedLength => _cpuCaches[0].Length;
    public int MaxCacheLength { get; }
    public bool IsCacheFull => CachedLength >= MaxCacheLength;

    /// <summary>
    /// The backend this engine actually runs on, for the UI. A real CUDA/OpenCL device shows
    /// <see cref="GpuDevice.Description"/> (e.g. <c>"[Cuda] ..., cuBLAS 12.8"</c>); an ILGPU CPU
    /// fallback device reports <c>"CPU"</c> so the display never claims GPU acceleration that
    /// isn't happening.
    /// </summary>
    public string Description => _dev.IsCpuFallback ? "CPU" : _dev.Description;

    /// <param name="maxPromptTokens">
    /// Requested upper bound on one GPU <see cref="Prefill"/> call's token count — sizes the scratch
    /// arena. It is clamped down at construction to the largest bound whose flash-only arena fits
    /// the device's reported memory, so engine creation never fails on a small GPU. A first prompt
    /// over the (clamped) bound, or a cache that is no longer empty, routes through the CPU path
    /// instead of throwing, so over-long or continuation prompts always work, just without GPU
    /// acceleration.
    /// </param>
    public GpuInferenceEngine(GpuDevice device, Transformer model, SharpMindConfig config, int maxCacheLength, int maxPromptTokens)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxCacheLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxPromptTokens);
        if (maxPromptTokens > maxCacheLength)
            throw new ArgumentException($"maxPromptTokens ({maxPromptTokens}) cannot exceed maxCacheLength ({maxCacheLength}).");
        ValidateSupported(model, config);

        _dev = device; _model = model; _cfg = model.Config; MaxCacheLength = maxCacheLength; _maxPromptTokens = maxPromptTokens;
        _gelu = config.Gate == GateKind.GeGLU;
        _gemmaScale = config.Activation == ActivationKind.GELU && config.Gate == GateKind.GeGLU;
        int kvDim = _cfg.NumKvHeads * _cfg.HeadDim;

        var owned = new List<IDisposable>();
        try
        {
            _blocks = new GpuBlock[_cfg.NumLayers];
            for (int l = 0; l < _cfg.NumLayers; l++) { _blocks[l] = new GpuBlock(device, model.GetBlock(l)!, static _ => null); owned.Add(_blocks[l]); }
            var rawEmbDtype = model.RawEmbeddingDtype;
            bool blockQuantEmb = rawEmbDtype is { } ed && IsOnDeviceDequant(ed);
            _embedding = blockQuantEmb
                ? null
                : DeviceBuffer.From(device, model.EmbeddingWeight);
            if (_embedding is not null) owned.Add(_embedding);
            // Quantized-resident embedding: upload the GGUF raw bytes and dequant on the device
            // (compact ~ a third of the F32 size), reading them for both the gather and the
            // weight-tied LM head. F32 embedding follows the ordinary float buffer above.
            if (blockQuantEmb)
            {
                if (model.RawEmbedding is null) throw new InvalidOperationException($"Model reports a {rawEmbDtype} embedding but carries no raw bytes.");
                _embeddingDtype = rawEmbDtype;
                _embeddingRaw = new DeviceByteBuffer(device.Accelerator, model.RawEmbedding); owned.Add(_embeddingRaw);
            }
            _finalNormW = DeviceBuffer.From(device, model.FinalNorm.NormWeight); owned.Add(_finalNormW);
            if (model.GetBlock(0)!.Attention.PositionalEncoder is RoPE rope)
            {
                _rope = true; _neox = rope.NeoxStyle; _ropeDim = rope.RopeDim;
                _cos = new DeviceBuffer(device, rope.CosTable.Length / (_ropeDim / 2), _ropeDim / 2); owned.Add(_cos); _cos.Tensor.Upload(rope.CosTable);
                _sin = new DeviceBuffer(device, rope.SinTable.Length / (_ropeDim / 2), _ropeDim / 2); owned.Add(_sin); _sin.Tensor.Upload(rope.SinTable);
            }
            _kCache = new DeviceBuffer[_cfg.NumLayers];
            _vCache = new DeviceBuffer[_cfg.NumLayers];
            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                _kCache[l] = new DeviceBuffer(device, maxCacheLength, kvDim); owned.Add(_kCache[l]);
                _vCache[l] = new DeviceBuffer(device, maxCacheLength, kvDim); owned.Add(_vCache[l]);
            }
            // Requests like the CLI's default (maxPromptTokens = min(4096, maxCache)) assume the arena
            // holds a 4096-token flash prefill. On a limited device that is far larger than available
            // memory (the old materialised-probs sizing asked for ~122 GB for a 7B model at 4096 rows),
            // so clamp the continuous GPU bound to whatever the flash-only arena can actually fit.
            // Over-bound prompts still work — the CPU path handles them, see the class doc.
            _maxPromptTokens = ClampRowsToDeviceBudget(maxPromptTokens);
            _arena = new DeviceArena(device, ArenaFloats(_cfg, _maxPromptTokens)); owned.Add(_arena);

            _cpuCaches = new IKVCache[_cfg.NumLayers];
            var cpuCacheBuilder = new KVCacherBuilder();
            for (int l = 0; l < _cfg.NumLayers; l++)
                _cpuCaches[l] = cpuCacheBuilder.CreateKVCache(1, _cfg.NumKvHeads, maxCacheLength, _cfg.HeadDim);

            _cpuWorkspace = MemoryHelpers.CreateWorkspace(Workspace.CalculateRequiredSize(model.Config.HiddenDim, model.Config.FfnDim, model.Config.VocabSize, model.Config.NumLayers, model.Config.MaxSeqLen));
        }
        catch
        {
            foreach (var d in owned) d.Dispose();
            throw;
        }

        _hostIds = new int[maxPromptTokens];
        _logitsHost = new float[_cfg.VocabSize];
    }

    /// <summary>Forward-only version of <see cref="GpuBackpropEngine.ArenaFloats"/>: no backward terms, no LoRA rank slots (inference weights are frozen/merged). Sized flash-only — the positioned attention materialises no S² probabilities, so the arena needs no <c>rows²·heads</c> term.</summary>
    private static long ArenaFloats(ModelConfig c, int rows)
    {
        long H = c.HiddenDim, qDim = (long)c.NumHeads * c.HeadDim, kvDim = (long)c.NumKvHeads * c.HeadDim, F = c.FfnDim, V = c.VocabSize;
        long entry = (long)rows * H;
        long fwdBlock = (long)rows * (6 * H + 2 + 2 * qDim + 2 * kvDim + 3 * F);   // flash attention: no materialised probs
        long head = (long)rows * (H + 1 + V);
        return (entry + fwdBlock * c.NumLayers + head) * 5 / 4 + 4096;
    }

    /// <summary>Largest rows in [1, requested] whose flash-only arena fits on the device, so arena
    /// allocation at construction never throws <c>CLException</c>/OOM on a small OpenCL/CUDA device.
    /// Uses ~3/4 of reported device memory as budget; when the device reports no usable size it
    /// returns <paramref name="requested"/> (the caller passes a sane CLI-bound default). Always ≥ 1 —
    /// even a single-token prefill needs only a few MB, and any over-bound prompt falls back to CPU.</summary>
    private int ClampRowsToDeviceBudget(int requested)
    {
        if (requested <= 1) return requested;
        long memBytes = _dev.Accelerator.MemorySize;
        if (memBytes <= 0) return requested;
        long budgetFloats = memBytes * 3 / 4 / 4;
        if (ArenaFloats(_cfg, 1) > budgetFloats) return 1;
        int lo = 1, hi = requested;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (ArenaFloats(_cfg, mid) <= budgetFloats) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

    /// <summary>
    /// The raw weight dtypes this engine can run <i>on the device</i>, consulted by the embedding
    /// path, <see cref="ValidateSupported(Transformer, SharpMindConfig)"/>, the metadata-only
    /// <see cref="CheckSupported(ModelMetaData, ModelConfig, SharpMindConfig)"/> gate, and
    /// <see cref="DescribeCpuFallback(ModelMetaData, ModelConfig, SharpMindConfig)"/>. F32 layers use
    /// the ordinary GEMM (the raw bytes are a full float weight); the block quants and K-quants run
    /// the on-device dequant matmul/gather. Dtypes absent here (notably Q8_K, whose block holds an
    /// F32 scale ILGPU device code cannot reinterpret, and the deferred dtypes with no kernel at all)
    /// are not refused: a block linear in such a quant falls back per-tensor to the host
    /// <see cref="SharpMind.Model.Layers.InferenceLinearLayer"/> forward, which handles every quant the
    /// CPU transformer runs.
    /// </summary>
    public static readonly IReadOnlySet<QuantDType> SupportedDtypes = new HashSet<QuantDType>
    {
        QuantDType.F32, QuantDType.Q8_0, QuantDType.Q4_0, QuantDType.Q4_1, QuantDType.Q5_0, QuantDType.Q5_1,
        QuantDType.Q2_K, QuantDType.Q3_K, QuantDType.Q4_K, QuantDType.Q5_K, QuantDType.Q6_K,
    };

    /// <summary>True for the quant dtypes with an on-device dequant matmul/gather kernel
    /// (<see cref="Kernels.QuantMatmulKernels.DequantMatmul"/>/<see cref="Kernels.QuantMatmulKernels.DequantMatmulK"/>)
    /// — i.e. <see cref="SupportedDtypes"/> minus the F32 float path.</summary>
    internal static bool IsOnDeviceDequant(QuantDType q)
        => q is not QuantDType.F32 && SupportedDtypes.Contains(q);

    /// <summary>The quant dtypes in <paramref name="meta"/> that have no on-device kernel, i.e. the
    /// per-tensor CPU fallback set <see cref="DescribeCpuFallback(ModelMetaData, ModelConfig, SharpMindConfig)"/>
    /// reports. Consults the tensors directly so block linears can be told apart from the embedding
    /// and the LM head (which always run on the device via the loader's F32 copies / device dequant).
    private static IEnumerable<QuantDType> CpuFallbackDtypes(ModelMetaData meta)
    {
        foreach (var dtype in meta.GetUsedQuantizations())
            if (!SupportedDtypes.Contains(dtype))
                yield return dtype;
    }

    /// <summary>
    /// Metadata-only (pre-weight-load) compatibility gate. The host calls this as soon as it has
    /// read a GGUF's headers — before paying for the weight load. Mirrors the config-level checks of
    /// <see cref="ValidateSupported(Transformer, SharpMindConfig)"/> that are derivable from
    /// <paramref name="meta"/> and <paramref name="modelConfig"/>. Only architecture limits refuse
    /// here — those are whole-model CPU cases the accelerator genuinely cannot run. Quant dtypes are
    /// <i>not</i> a refusal: unsupported-quant block linears fall back per-tensor to the host (see
    /// <see cref="DescribeCpuFallback(ModelMetaData, ModelConfig, SharpMindConfig)"/>), and the
    /// per-layer gate still runs on the built model as the authoritative check.
    /// </summary>
    /// <returns>true when supported, with <paramref name="reason"/> null; otherwise false.</returns>
    public static bool CheckSupported(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(meta);
        ArgumentNullException.ThrowIfNull(modelConfig);
        ArgumentNullException.ThrowIfNull(config);
        string Why(string s) => $"GPU inference engine (M0) does not support {s}; use CPU inference, which does.";

        if (modelConfig.NumLayers <= 0) { reason = Why("a model without blocks"); return false; }
        if (modelConfig.PositionalEncoding is not (PositionalEncoding.RoPE or PositionalEncoding.NoPE)) { reason = Why($"positional encoding {modelConfig.PositionalEncoding}"); return false; }
        if (config.Ffn == FfnKind.MoE) { reason = Why("MoE"); return false; }
        if (config.Gate == GateKind.None) { reason = Why("dense (ungated) FFN"); return false; }
        reason = null;
        return true;
    }

    /// <summary>
    /// Describes per-tensor CPU fallback for <paramref name="meta"/>: which model content this engine
    /// would run on the host rather than the device. Null means the whole model runs on the device
    /// (no consent needed). Non-null names the block-line quant dtypes that fall back. Embeddings and
    /// the LM head are deliberately excluded — whatever their storage quant, the loader always
    /// materialises their F32 copies (or the engine's device dequant handles them), so they never
    /// fall back. The caller (factory → launcher → CUI) surfaces this once, before loading weights.
    /// </summary>
    public static string? DescribeCpuFallback(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config)
    {
        ArgumentNullException.ThrowIfNull(meta);
        var dtypes = CpuFallbackDtypes(meta).ToArray();
        if (dtypes.Length == 0) return null;

        // Which of the fallen-back dtypes actually land on block linears? The embedding and the LM
        // head store their quants too, but (as above) always run on the device; only block linears
        // genuinely fall back. When a dtype appears on block tensors, it is reported.
        var blockDtypes = meta.Tensors
            .Where(t => t.Name.Contains("blk.", StringComparison.OrdinalIgnoreCase))
            .Select(t => t.Dtype)
            .Where(dtypes.Contains)
            .Distinct()
            .OrderBy(d => d)
            .ToArray();
        if (blockDtypes.Length == 0) return null;

        return $"will run on the CPU: {string.Join(", ", blockDtypes.Select(d => d.ToString()))} " +
            $"weights (block linears); the rest of the model stays on the GPU.";
    }

    public static void ValidateSupported(Transformer model, SharpMindConfig config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);
        var c = model.Config;
        static string Why(string s) => $"GPU inference engine (M0) does not support {s}; use CPU inference, which does.";
        if (model.FinalNorm is not RmsNormLayer) throw new NotSupportedException(Why($"a {model.FinalNorm.GetType().Name} final norm — LayerNorm, only RMSNorm"));
        if (c.PositionalEncoding is not (PositionalEncoding.RoPE or PositionalEncoding.NoPE)) throw new NotSupportedException(Why($"positional encoding {c.PositionalEncoding}"));
        if (config.Ffn == FfnKind.MoE) throw new NotSupportedException(Why("MoE"));
        if (model.QuantAwareTrainingTarget is not null and not Core.Quantization.QuantDType.F32) throw new NotSupportedException(Why("a quantized model — dequantize first (LoadMode.Full)"));
        if (config.Gate == GateKind.None) throw new NotSupportedException(Why("dense (ungated) FFN"));
        if (c.NumLayers <= 0 || model.GetBlock(0) is null) throw new NotSupportedException(Why("a model without blocks"));
        // Quant dtypes are NOT a gate here: a block linear in a quant with no on-device kernel
        // (Q8_K, or any deferred quant) falls back per-tensor to the host InferenceLinearLayer
        // forward, which runs every quant the CPU transformer runs. The embedding and the tied LM
        // head run on the device for every storage quant — the loader always materialises their F32
        // copies, and the device-dequant path handles the dtypes with a kernel. Only architecture
        // limits refuse below (whole-model CPU cases).
        for (int l = 0; l < c.NumLayers; l++)
        {
            var b = model.GetBlock(l) ?? throw new NotSupportedException(Why($"a model missing block {l}"));
            if (b.PostAttnNorm is not null || b.PostFfnNorm is not null) throw new NotSupportedException(Why("Gemma post-attention/post-FFN norms"));
            if (b.Norm1 is not RmsNormLayer || b.Norm2 is not RmsNormLayer) throw new NotSupportedException(Why($"a {b.Norm1.GetType().Name} block norm — LayerNorm, only RMSNorm"));
            if (b.Ffn is not GatedFfnLayer) throw new NotSupportedException(Why($"FFN kind {b.Ffn.GetType().Name}"));
        }
    }

    public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokenIds.Length == 0) throw new ArgumentException("tokenIds must not be empty.", nameof(tokenIds));
        cancellationToken.ThrowIfCancellationRequested();

        // GPU-accelerate any prefill whose new tokens fit the arena. Fresh (empty-cache) prompt
        // starts on the device; a continued conversation's incremental prefill appends to the
        // device cache the prior turns already filled. The arena is sized for one call's new tokens
        // only (_maxPromptTokens) — the persistent device KV cache it reads is a separate,
        // pre-allocated buffer — so a continued prefill stays on the device for any context length
        // up to the cache cap, matching decode. Anything that would overflow the arena or the
        // device cache uses the CPU path through the host caches, never recomputing the prefix.
        if (CachedLength == 0 && tokenIds.Length <= _maxPromptTokens)
            return PrefillGpu(tokenIds, onChunkProgress, cancellationToken);

        if (CachedLength > 0
            && tokenIds.Length <= _maxPromptTokens
            && CachedLength + tokenIds.Length <= MaxCacheLength)
            return PrefillGpuContinued(tokenIds, onChunkProgress, cancellationToken);

        return PrefillCpu(tokenIds, onChunkProgress, cancellationToken);
    }

    /// <summary>Fills <paramref name="x"/> ([m, H]) with the token embeddings for <paramref name="ids"/>,
    /// dequantising on the device when the embedding is a Q8_0 raw table, ordinary gather otherwise.</summary>
    private void EmbedOnce(DeviceTensor x, ArrayView<int> ids)
    {
        if (_embeddingDtype is { } q) _dev.Kernels.DequantGather(x, _embeddingRaw!, ids, _cfg.HiddenDim, q);
        else _dev.Kernels.EmbedGather(x, _embedding!.Tensor, ids);
    }

    /// <summary>Weight-tied LM head: logits = n·Eᵀ. For a block-quantized embedding this is the on-device
    /// dequant matmul (K = Hidden, N = Vocab, raw = <see cref="_embeddingRaw"/>); an F32 embedding
    /// is the same weight via <see cref="GpuDevice.Gemm"/> as before.</summary>
    private void LmHead(DeviceTensor logits, DeviceTensor finalNormed, int m)
    {
        int H = _cfg.HiddenDim, V = _cfg.VocabSize;
        if (_embeddingDtype is { } q) _dev.Kernels.DequantMatmul(logits, finalNormed, _embeddingRaw!, H, V, q);
        else _dev.Gemm(logits, finalNormed, _embedding!.Tensor, m, V, H, saI: H, saK: 1, sbK: 1, sbJ: H);
    }

    /// <summary>The device first-prefill of a fresh conversation: runs the whole prompt forward once
    /// on the device, persists its post-RoPE K/V at cache positions [0, m), and mirrors it into
    /// <see cref="_cpuCaches"/> so <see cref="CachedLength"/> and the session contracts are in step
    /// for the GPU decode / continued-prefill steps that follow.</summary>
    private ReadOnlyMemory<float> PrefillGpu(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress,
        CancellationToken cancellationToken)
    {
        int m = tokenIds.Length;
        tokenIds.CopyTo(_hostIds);
        _arena.Reset();
        using var ids = _dev.UploadInts(_hostIds[..m]);

        var k = _dev.Kernels;
        int H = _cfg.HiddenDim, D = _cfg.HeadDim, nh = _cfg.NumHeads, nkv = _cfg.NumKvHeads;
        int kvDim = nkv * D;
        var x = _arena.Rent(m, H);
        EmbedOnce(x, ids.View);
        if (_gemmaScale) k.Scale(x, MathF.Sqrt(H));

        for (int l = 0; l < _blocks.Length; l++)
        {
            var blk = _blocks[l];
            var norm1Out = _arena.Rent(m, H); var rInv1 = _arena.Rent(m, 1);
            k.RmsNormFwd(norm1Out, rInv1, x, blk.Norm1W.Tensor, _cfg.NormEps);

            var q = _arena.Rent(m, nh * D); var kk = _arena.Rent(m, nkv * D); var v = _arena.Rent(m, nkv * D);
            blk.Wq.Forward(q, norm1Out, _arena); blk.Wk.Forward(kk, norm1Out, _arena); blk.Wv.Forward(v, norm1Out, _arena);
            if (_rope) { k.RopeFwd(q, _cos!.Tensor, _sin!.Tensor, m, nh, D, _ropeDim, _neox); k.RopeFwd(kk, _cos.Tensor, _sin.Tensor, m, nkv, D, _ropeDim, _neox); }

            // Persist post-RoPE K/V at [0, m) on the device cache — the authority the GPU attention
            // reads. This first prefill uses the same positioned flash attention as decode and
            // continued prefill (pos0 = 0, qLen = kvLen = m), so the arena needs no materialised
            // S² probabilities tensor and is sized flash-only by ArenaFloats.
            k.Copy(_kCache[l].Tensor.Slice(0, m), kk);
            k.Copy(_vCache[l].Tensor.Slice(0, m), v);

            var stats = _arena.Rent(nh * m, 3);
            var attnOut = _arena.Rent(m, nh * D);
            k.AttnFwdKvLen(attnOut, stats, q, _kCache[l].Tensor.Slice(0, m), _vCache[l].Tensor.Slice(0, m),
                pos0: 0, qLen: m, kvLen: m, numHeads: nh, numKv: nkv, headDim: D);

            var proj = _arena.Rent(m, H); blk.Wo.Forward(proj, attnOut, _arena);
            k.AddInPlace(x, proj);

            var norm2Out = _arena.Rent(m, H); var rInv2 = _arena.Rent(m, 1);
            k.RmsNormFwd(norm2Out, rInv2, x, blk.Norm2W.Tensor, _cfg.NormEps);
            var fused = _arena.Rent(m, 2 * _cfg.FfnDim); blk.WGated.Forward(fused, norm2Out, _arena);
            var act = _arena.Rent(m, _cfg.FfnDim); k.GateFwd(act, fused, _gelu);
            var down = _arena.Rent(m, H); blk.WDown.Forward(down, act, _arena);
            k.AddInPlace(x, down);

            cancellationToken.ThrowIfCancellationRequested();
            onChunkProgress?.Invoke((double)(l + 1) / _blocks.Length);
        }

        var finalNormed = _arena.Rent(m, H); var finalRInv = _arena.Rent(m, 1);
        k.RmsNormFwd(finalNormed, finalRInv, x, _finalNormW.Tensor, _cfg.NormEps);
        var logits = _arena.Rent(m, _cfg.VocabSize);
        // logits = n·Eᵀ, E [V,H] — same weight-tied LM head as GpuBackpropEngine.Forward.
        LmHead(logits, finalNormed, m);
        _dev.Synchronize();

        // Only the LAST token's logits are wanted (IInferenceEngine.Prefill's contract).
        logits.Slice(m - 1, 1).Download(_logitsHost);

        // Materialise the device K/V into the host mirror cache. Each layer's device tensor is
        // [m, kvDim] position-major [pos][kvHead][headDim], which is exactly the [1, m, kvDim]
        // layout KVCache.Update expects, so a flat download + Update is the bridge.
        var kHost = new float[m * kvDim];
        var vHost = new float[m * kvDim];
        for (int l = 0; l < _blocks.Length; l++)
        {
            _kCache[l].Tensor.Slice(0, m).Download(kHost);
            _vCache[l].Tensor.Slice(0, m).Download(vHost);
            using var kTensor = Tensor<float>.From(kHost, 1, m, kvDim);
            using var vTensor = Tensor<float>.From(vHost, 1, m, kvDim);
            _cpuCaches[l].Update(kTensor, vTensor, nkv, D);
        }
        _deviceCacheLength = m;
        return _logitsHost;
    }

    /// <summary>
    /// Continued GPU prefill: <c>m</c> fresh prompt tokens at absolute positions [c, c+m) on top
    /// of a warm device cache of length <c>c</c>. Like <see cref="PrefillGpu"/> but the new K/V
    /// rows are appended at <c>Slice(c, m)</c> and attention is the positioned
    /// <see cref="GpuKernels.AttnFwdKvLen"/> (flash, no materialised probs) with <c>pos0 = c</c>,
    /// so the already-cached prefix is attended to, never recomputed. The new rows are mirrored
    /// into <see cref="_cpuCaches"/> so the host length stays in step with the device.
    /// </summary>
    private ReadOnlyMemory<float> PrefillGpuContinued(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress,
        CancellationToken cancellationToken)
    {
        int c = CachedLength;
        int m = tokenIds.Length;
        tokenIds.CopyTo(_hostIds);
        _arena.Reset();
        using var ids = _dev.UploadInts(_hostIds[..m]);

        var k = _dev.Kernels;
        int H = _cfg.HiddenDim, D = _cfg.HeadDim, nh = _cfg.NumHeads, nkv = _cfg.NumKvHeads;
        int kvDim = nkv * D;
        var x = _arena.Rent(m, H);
        EmbedOnce(x, ids.View);
        if (_gemmaScale) k.Scale(x, MathF.Sqrt(H));

        for (int l = 0; l < _blocks.Length; l++)
        {
            var blk = _blocks[l];
            var norm1Out = _arena.Rent(m, H); var rInv1 = _arena.Rent(m, 1);
            k.RmsNormFwd(norm1Out, rInv1, x, blk.Norm1W.Tensor, _cfg.NormEps);

            var q = _arena.Rent(m, nh * D); var kk = _arena.Rent(m, nkv * D); var v = _arena.Rent(m, nkv * D);
            blk.Wq.Forward(q, norm1Out, _arena); blk.Wk.Forward(kk, norm1Out, _arena); blk.Wv.Forward(v, norm1Out, _arena);
            if (_rope) { k.RopeFwdPos(q, _cos!.Tensor, _sin!.Tensor, m, c, nh, D, _ropeDim, _neox); k.RopeFwdPos(kk, _cos.Tensor, _sin.Tensor, m, c, nkv, D, _ropeDim, _neox); }

            // Append post-RoPE K/V at positions [c, c+m) on the device cache.
            k.Copy(_kCache[l].Tensor.Slice(c, m), kk);
            k.Copy(_vCache[l].Tensor.Slice(c, m), v);

            var stats = _arena.Rent(nh * m, 3);
            var attnOut = _arena.Rent(m, nh * D);
            k.AttnFwdKvLen(attnOut, stats, q, _kCache[l].Tensor.Slice(0, c + m), _vCache[l].Tensor.Slice(0, c + m),
                pos0: c, qLen: m, kvLen: c + m, numHeads: nh, numKv: nkv, headDim: D);

            var proj = _arena.Rent(m, H); blk.Wo.Forward(proj, attnOut, _arena);
            k.AddInPlace(x, proj);

            var norm2Out = _arena.Rent(m, H); var rInv2 = _arena.Rent(m, 1);
            k.RmsNormFwd(norm2Out, rInv2, x, blk.Norm2W.Tensor, _cfg.NormEps);
            var fused = _arena.Rent(m, 2 * _cfg.FfnDim); blk.WGated.Forward(fused, norm2Out, _arena);
            var act = _arena.Rent(m, _cfg.FfnDim); k.GateFwd(act, fused, _gelu);
            var down = _arena.Rent(m, H); blk.WDown.Forward(down, act, _arena);
            k.AddInPlace(x, down);

            cancellationToken.ThrowIfCancellationRequested();
            onChunkProgress?.Invoke((double)(l + 1) / _blocks.Length);
        }

        var finalNormed = _arena.Rent(m, H); var finalRInv = _arena.Rent(m, 1);
        k.RmsNormFwd(finalNormed, finalRInv, x, _finalNormW.Tensor, _cfg.NormEps);
        var logits = _arena.Rent(m, _cfg.VocabSize);
        LmHead(logits, finalNormed, m);
        _dev.Synchronize();

        logits.Slice(m - 1, 1).Download(_logitsHost);

        // Mirror the newly appended rows [c, c+m) into the host caches so CachedLength stays in
        // step with the device cache (appending keeps position-major [1, m, kvDim] layout).
        UpdateHostMirror(c, m);

        _deviceCacheLength = c + m;
        return _logitsHost;
    }

    /// <summary>CPU prefill for a continued conversation (warm cache) or a prompt beyond the
    /// arena bound. <see cref="Prefill.ForwardLastLogitsChunked"/> starts at the current cache
    /// length, so the GPU-materialised prefix is reused, never recomputed.</summary>
    private ReadOnlyMemory<float> PrefillCpu(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int[] ids = tokenIds.ToArray();
        using var logits = SharpMind.Inference.Prefill.ForwardLastLogitsChunked(_model, _cpuCaches, ids, _cpuWorkspace, onChunkProgress);
        logits.Data[.._cfg.VocabSize].CopyTo(_logitsHost);
        // The CPU pass extended the host caches the device cache can't know about — rewrite the
        // device cache from the host so a later GPU decode/continued-prefill reads the right rows.
        EnsureDeviceCacheInSync();
        return _logitsHost;
    }

    /// <summary>
    /// Downloads the device cache rows [<paramref name="from"/>, <paramref name="from"/> +
    /// <paramref name="count"/>) for every layer and appends them to the matching host
    /// <see cref="_cpuCaches"/> entry. The host cache's own length is <paramref name="from"/>
    /// before this call (the device rows start there), and remains <c>from + count</c> after, so
    /// the two caches never diverge over a GPU forward. Position-major [pos][kvHead][headDim]
    /// maps straight onto KVCache.Update's [1, count, kvDim] expectation.
    /// </summary>
    private void UpdateHostMirror(int from, int count)
    {
        int kvDim = _cfg.NumKvHeads * _cfg.HeadDim;
        if (_cpuCaches[0].Length != from) throw new InvalidOperationException($"Host cache length {_cpuCaches[0].Length} != device append start {from}.");
        var kHost = new float[count * kvDim];
        var vHost = new float[count * kvDim];
        for (int l = 0; l < _blocks.Length; l++)
        {
            _kCache[l].Tensor.Slice(from, count).Download(kHost);
            _vCache[l].Tensor.Slice(from, count).Download(vHost);
            using var kTensor = Tensor<float>.From(kHost, 1, count, kvDim);
            using var vTensor = Tensor<float>.From(vHost, 1, count, kvDim);
            _cpuCaches[l].Update(kTensor, vTensor, _cfg.NumKvHeads, _cfg.HeadDim);
        }
    }

    /// <summary>Re-uploads the full host cache into the device cache, used after a host mutation
    /// (trim-to-last, import) that shifts which positions are valid — those are no longer the
    /// device rows [0, length), so they are rewritten from the host to compress back to [0, length).
    /// Reads each layer's K/V through <see cref="IKVCache.GetKeyPtr"/>/<see cref="IKVCache.GetValuePtr"/>.</summary>
    private unsafe void SyncDeviceCacheFromHost(int length)
    {
        int kvDim = _cfg.NumKvHeads * _cfg.HeadDim;
        var kHost = new float[length * kvDim];
        var vHost = new float[length * kvDim];
        for (int l = 0; l < _blocks.Length; l++)
        {
            var cache = _cpuCaches[l];
            for (int p = 0; p < length; p++)
                for (int h = 0; h < _cfg.NumKvHeads; h++)
                {
                    int off = (p * _cfg.NumKvHeads + h) * _cfg.HeadDim;
                    new Span<float>(cache.GetKeyPtr(0, p, h), _cfg.HeadDim).CopyTo(kHost.AsSpan(off, _cfg.HeadDim));
                    new Span<float>(cache.GetValuePtr(0, p, h), _cfg.HeadDim).CopyTo(vHost.AsSpan(off, _cfg.HeadDim));
                }
            _kCache[l].Tensor.Slice(0, length).Upload(kHost);
            _vCache[l].Tensor.Slice(0, length).Upload(vHost);
        }
        _deviceCacheLength = length;
    }

    /// <summary>Rewrites the device cache from the host cache after a CPU forward extended or
    /// changed it, so the invariant <c>device rows [0, CachedLength) == host rows [0, CachedLength)</c>
    /// holds for the next GPU step. A no-op on an empty cache.</summary>
    private void EnsureDeviceCacheInSync()
    {
        int length = CachedLength;
        if (length > 0 && _deviceCacheLength != length) SyncDeviceCacheFromHost(length);
    }

    /// <summary>
    /// Runs one decode step on the device: the single token at absolute position <c>c</c> is
    /// embedded and forwarded through every block, its post-RoPE K/V is appended to the device
    /// cache at <c>Slice(c, 1)</c>, and positioned attention
    /// <see cref="GpuKernels.AttnFwdKvLen"/> (qLen 1, kvLen c+1) reads the whole <c>c+1</c>-long
    /// cache — KV-cache-efficient, O(S) per token, no prefix replay. Logits are downloaded; the
    /// host mirror is extended so <see cref="CachedLength"/> and the session contracts stay in
    /// step. Falls back to the host <see cref="Transformer"/> for a full cache.
    /// </summary>
    public ReadOnlyMemory<float> DecodeStep(int tokenId, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();

        if (IsCacheFull)
            return DecodeCpu(tokenId, cancellationToken);

        return DecodeGpu(tokenId, cancellationToken);
    }

    private ReadOnlyMemory<float> DecodeCpu(int tokenId, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
        _cpuWorkspace.Reset();
        using var stepInput = _cpuWorkspace.Rent<int>([1, 1]);
        stepInput.Data[0] = tokenId;
        using var logits = _model.ForwardLastLogits(stepInput, _cpuCaches, CachedLength, _cpuWorkspace);
        logits.Data[.._cfg.VocabSize].CopyTo(_logitsHost);
        // The CPU decode appended to the host caches only — sync the device cache so a later GPU
        // decode/continued-prefill reads the updated K/V.
        EnsureDeviceCacheInSync();
        return _logitsHost;
    }

    private ReadOnlyMemory<float> DecodeGpu(int tokenId, CancellationToken cancellationToken)
    {
        int c = CachedLength;
        _arena.Reset();
        using var ids = _dev.UploadInts(new[] { tokenId });

        var k = _dev.Kernels;
        int H = _cfg.HiddenDim, D = _cfg.HeadDim, nh = _cfg.NumHeads, nkv = _cfg.NumKvHeads;
        int kvDim = nkv * D;
        var x = _arena.Rent(1, H);
        EmbedOnce(x, ids.View);
        if (_gemmaScale) k.Scale(x, MathF.Sqrt(H));

        for (int l = 0; l < _blocks.Length; l++)
        {
            var blk = _blocks[l];
            var norm1Out = _arena.Rent(1, H); var rInv1 = _arena.Rent(1, 1);
            k.RmsNormFwd(norm1Out, rInv1, x, blk.Norm1W.Tensor, _cfg.NormEps);

            var q = _arena.Rent(1, nh * D); var kk = _arena.Rent(1, nkv * D); var v = _arena.Rent(1, nkv * D);
            blk.Wq.Forward(q, norm1Out, _arena); blk.Wk.Forward(kk, norm1Out, _arena); blk.Wv.Forward(v, norm1Out, _arena);
            if (_rope) { k.RopeFwdPos(q, _cos!.Tensor, _sin!.Tensor, 1, c, nh, D, _ropeDim, _neox); k.RopeFwdPos(kk, _cos.Tensor, _sin.Tensor, 1, c, nkv, D, _ropeDim, _neox); }

            // Append this token's post-RoPE K/V at position c on the device cache.
            k.Copy(_kCache[l].Tensor.Slice(c, 1), kk);
            k.Copy(_vCache[l].Tensor.Slice(c, 1), v);

            // Decode attention: one query row at position c over the whole c+1-row cache.
            var stats = _arena.Rent(nh, 3);
            var attnOut = _arena.Rent(1, nh * D);
            k.AttnFwdKvLen(attnOut, stats, q, _kCache[l].Tensor.Slice(0, c + 1), _vCache[l].Tensor.Slice(0, c + 1),
                pos0: c, qLen: 1, kvLen: c + 1, numHeads: nh, numKv: nkv, headDim: D);

            var proj = _arena.Rent(1, H); blk.Wo.Forward(proj, attnOut, _arena);
            k.AddInPlace(x, proj);

            var norm2Out = _arena.Rent(1, H); var rInv2 = _arena.Rent(1, 1);
            k.RmsNormFwd(norm2Out, rInv2, x, blk.Norm2W.Tensor, _cfg.NormEps);
            var fused = _arena.Rent(1, 2 * _cfg.FfnDim); blk.WGated.Forward(fused, norm2Out, _arena);
            var act = _arena.Rent(1, _cfg.FfnDim); k.GateFwd(act, fused, _gelu);
            var down = _arena.Rent(1, H); blk.WDown.Forward(down, act, _arena);
            k.AddInPlace(x, down);

            cancellationToken.ThrowIfCancellationRequested();
        }

        var finalNormed = _arena.Rent(1, H); var finalRInv = _arena.Rent(1, 1);
        k.RmsNormFwd(finalNormed, finalRInv, x, _finalNormW.Tensor, _cfg.NormEps);
        var logits = _arena.Rent(1, _cfg.VocabSize);
        LmHead(logits, finalNormed, 1);
        _dev.Synchronize();

        logits.Slice(0, 1).Download(_logitsHost);

        UpdateHostMirror(c, 1);
        _deviceCacheLength = c + 1;
        return _logitsHost;
    }

    public void TruncateCache(int length)
    {
        if (length < 0 || length > CachedLength) throw new ArgumentOutOfRangeException(nameof(length));
        foreach (var c in _cpuCaches) c.Truncate(length);
        // Rows [0, length) are unchanged on the device, so only the live length shrinks.
        _deviceCacheLength = length;
    }

    public void TrimToLast(int keep)
    {
        if (keep < 0 || keep > CachedLength) throw new ArgumentOutOfRangeException(nameof(keep));
        foreach (var c in _cpuCaches) c.TrimToLast(keep);
        // Trim keeps the last `keep` positions — not the device rows [0, keep). Rewrite from host
        // so the next GPU step reads the right rows.
        if (keep > 0) SyncDeviceCacheFromHost(keep);
        else _deviceCacheLength = 0;
    }

    public void ResetCache()
    {
        foreach (var c in _cpuCaches) c.Reset();
        _deviceCacheLength = 0;
    }

    public KVCacheSnapshot ExportCache(int[] promptTokenIds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        var layers = new List<byte[]>(_blocks.Length);
        foreach (var c in _cpuCaches) layers.Add(c.SnapshotBytes()!);
        return new KVCacheSnapshot
        {
            PromptHash = KVCacheSnapshot.HashPromptTokens(promptTokenIds),
            PromptTokenCount = promptTokenIds.Length,
            Layers = layers,
        };
    }

    public void ImportCache(KVCacheSnapshot snapshot)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(snapshot);
        if (snapshot.Layers.Count != _cpuCaches.Length)
            throw new ArgumentException($"Snapshot has {snapshot.Layers.Count} layers, this engine has {_cpuCaches.Length}.", nameof(snapshot));
        for (int l = 0; l < _cpuCaches.Length; l++)
            _cpuCaches[l].RestoreBytes(snapshot.Layers[l]);
        // An import rewrites which rows are valid — sync the device cache from the host so the
        // next GPU decode/continued-prefill reads the imported K/V, not stale device rows.
        int importedLength = _cpuCaches[0].Length;
        if (importedLength > 0) SyncDeviceCacheFromHost(importedLength);
        else _deviceCacheLength = 0;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _blocks) b.Dispose();
        _embedding?.Dispose(); _embeddingRaw?.Dispose(); _finalNormW.Dispose(); _cos?.Dispose(); _sin?.Dispose();
        foreach (var b in _kCache) b.Dispose();
        foreach (var b in _vCache) b.Dispose();
        foreach (var c in _cpuCaches) c.Dispose();
        _arena.Dispose();
        _cpuWorkspace.Dispose();
    }
}
