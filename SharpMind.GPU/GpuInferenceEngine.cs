using SharpMind.Core;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.GPU;

/// <summary>
/// GPU inference engine, M0 hybrid: the FIRST prefill of a conversation (an empty cache,
/// prompt within <see cref="_maxPromptTokens"/>) runs on the device — reusing
/// <see cref="GpuBackpropEngine"/>'s exact forward op sequence (embed → per-block
/// norm/QKV/RoPE/attention/FFN → final norm → LM head) and is genuinely correct and tested
/// the same way (<c>GpuTestDevice</c>, no real hardware required). Every continued pass —
/// a second chat turn's incremental prefill, and each single-token
/// <see cref="DecodeStep"/> — runs on the CPU through the host <see cref="Transformer"/>
/// against a host <see cref="IKVCache"/>[] that the GPU prefill materialises from the device
/// K/V after the first turn, so the prompt is never recomputed. True incremental GPU decode
/// is a follow-up: the device K/V (<see cref="_kCache"/>/<see cref="_vCache"/>) is written
/// during the first prefill so that kernel's layout is right and tested ahead of it.
///
/// This integrates GPU prefill with the correctness and multi-turn support of the CPU decode
/// path while the GPU attention kernel still requires Q, K and V to share one <c>seqLen</c>
/// with no position offset (see the class doc on <see cref="GpuBackpropEngine"/>'s attention).
///
/// Model constraints mirror <see cref="GpuBackpropEngine.ValidateSupported"/> minus the
/// LoRA/trainable-parameter checks, which don't apply here: RMSNorm final norm and block
/// norms, RoPE or NoPE, gated FFN, no MoE. F32 weights only — quantized GGUF weights must
/// already be dequantized on the host <see cref="Transformer"/> (load with <c>LoadMode.Full</c>).
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
    private readonly DeviceBuffer _embedding, _finalNormW;
    private readonly DeviceBuffer? _cos, _sin;

    /// <summary>
    /// Persistent per-layer device K/V storage, [MaxCacheLength, NumKvHeads·HeadDim]. Written by
    /// the GPU first <see cref="Prefill"/> so the layout is right and tested ahead of a future
    /// on-device <see cref="DecodeStep"/>. The continued (CPU) turns do not refresh it — the
    /// authoritative live cache for those is <see cref="_cpuCaches"/>.
    /// </summary>
    private readonly DeviceBuffer[] _kCache, _vCache;

    /// <summary>
    /// Host CPU KV caches, one per layer, that the GPU first prefill materialises into and that
    /// every continued prefill and decode step runs against through <see cref="_model"/>. This is
    /// the engine's authoritative cache: <see cref="CachedLength"/> and the trim/reset/snapshot
    /// operations all forward here.
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
    /// Upper bound on one GPU <see cref="Prefill"/> call's token count — sizes the scratch arena.
    /// A first prompt over this bound (or a cache that is no longer empty) routes through the CPU
    /// path instead of throwing, so over-long or continuation prompts always work, just without
    /// GPU acceleration.
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
            _embedding = DeviceBuffer.From(device, model.EmbeddingWeight); owned.Add(_embedding);
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
            _arena = new DeviceArena(device, ArenaFloats(_cfg, maxPromptTokens)); owned.Add(_arena);

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

    /// <summary>Forward-only version of <see cref="GpuBackpropEngine.ArenaFloats"/>: no backward terms, no LoRA rank slots (inference weights are frozen/merged).</summary>
    private static long ArenaFloats(ModelConfig c, int rows)
    {
        long H = c.HiddenDim, qDim = (long)c.NumHeads * c.HeadDim, kvDim = (long)c.NumKvHeads * c.HeadDim, F = c.FfnDim, V = c.VocabSize;
        long rowsBH = (long)rows * c.NumHeads;
        long fwdAttn = rowsBH * rows;              // batch folded into `rows` — M0 is single-sequence (batch 1)
        long entry = (long)rows * H;
        long fwdBlock = (long)rows * (6 * H + 2 + 2 * qDim + 2 * kvDim + 3 * F) + fwdAttn;
        long head = (long)rows * (H + 1 + V);
        return (entry + fwdBlock * c.NumLayers + head) * 5 / 4 + 4096;
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

        // GPU-accelerate exactly the first turn of a fresh conversation, within the arena bound.
        // Anything else — a warm cache (a continued conversation) or a longer prompt — uses the
        // CPU path through the already-populated host caches, which never recomputes the prefix.
        if (CachedLength == 0 && tokenIds.Length <= _maxPromptTokens)
            return PrefillGpu(tokenIds, onChunkProgress, cancellationToken);

        return PrefillCpu(tokenIds, onChunkProgress, cancellationToken);
    }

    /// <summary>The device first-prefill: runs the whole prompt forward once and downloads each
    /// layer's K/V into <see cref="_cpuCaches"/> so the CPU decode path can continue from the
    /// prefilled prefix without recomputing it.</summary>
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
        k.EmbedGather(x, _embedding.Tensor, ids.View);
        if (_gemmaScale) k.Scale(x, MathF.Sqrt(H));

        for (int l = 0; l < _blocks.Length; l++)
        {
            var blk = _blocks[l];
            var norm1Out = _arena.Rent(m, H); var rInv1 = _arena.Rent(m, 1);
            k.RmsNormFwd(norm1Out, rInv1, x, blk.Norm1W.Tensor, _cfg.NormEps);

            var q = _arena.Rent(m, nh * D); var kk = _arena.Rent(m, nkv * D); var v = _arena.Rent(m, nkv * D);
            blk.Wq.Forward(q, norm1Out, _arena); blk.Wk.Forward(kk, norm1Out, _arena); blk.Wv.Forward(v, norm1Out, _arena);
            if (_rope) { k.RopeFwd(q, _cos!.Tensor, _sin!.Tensor, m, nh, D, _ropeDim, _neox); k.RopeFwd(kk, _cos.Tensor, _sin.Tensor, m, nkv, D, _ropeDim, _neox); }

            // Persist post-RoPE K/V at [0, m) on the device for a future on-device DecodeStep —
            // see the class doc comment. Not read back by the M0 decode path, which uses the
            // host caches materialised below.
            k.Copy(_kCache[l].Tensor.Slice(0, m), kk);
            k.Copy(_vCache[l].Tensor.Slice(0, m), v);

            var probs = _arena.Rent(nh * m, m);
            var attnOut = _arena.Rent(m, nh * D);
            k.AttnFwd(attnOut, probs, q, kk, v, batch: 1, seqLen: m, numHeads: nh, numKv: nkv, headDim: D);

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
        _dev.Gemm(logits, finalNormed, _embedding.Tensor, m, _cfg.VocabSize, H, saI: H, saK: 1, sbK: 1, sbJ: H);
        _dev.Synchronize();

        // Only the LAST token's logits are wanted (IInferenceEngine.Prefill's contract).
        logits.Slice(m - 1, 1).Download(_logitsHost);

        // Materialise the device K/V into the host caches the CPU decode path reads. Each layer's
        // device tensor is [m, kvDim] position-major [pos][kvHead][headDim], which is exactly the
        // [1, m, kvDim] layout KVCache.Update expects, so a flat download + Update is the bridge.
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
        return _logitsHost;
    }

    /// <summary>
    /// Runs one decode step on the CPU through the host <see cref="Transformer"/> against the
    /// host caches, which the GPU first prefill already primed with the prompt's K/V — so a turn
    /// never recomputes its prefix. True on-device decode (a kernel with independent query and
    /// key/value lengths, reading <see cref="_kCache"/>/<see cref="_vCache"/> with a position
    /// offset) is a follow-up; <see cref="Kernels.AttentionKernels.AttnFwd"/>/<c>AttnFwdFlash</c>
    /// both take one shared <c>seqLen</c> with no offset because
    /// <see cref="GpuBackpropEngine"/> only ever forwards a whole fresh training batch.
    /// </summary>
    public ReadOnlyMemory<float> DecodeStep(int tokenId, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();

        _cpuWorkspace.Reset();
        using var stepInput = _cpuWorkspace.Rent<int>([1, 1]);
        stepInput.Data[0] = tokenId;
        using var logits = _model.ForwardLastLogits(stepInput, _cpuCaches, CachedLength, _cpuWorkspace);
        logits.Data[.._cfg.VocabSize].CopyTo(_logitsHost);
        return _logitsHost;
    }

    public void TruncateCache(int length)
    {
        if (length < 0 || length > CachedLength) throw new ArgumentOutOfRangeException(nameof(length));
        foreach (var c in _cpuCaches) c.Truncate(length);
    }

    public void TrimToLast(int keep)
    {
        if (keep < 0 || keep > CachedLength) throw new ArgumentOutOfRangeException(nameof(keep));
        foreach (var c in _cpuCaches) c.TrimToLast(keep);
    }

    public void ResetCache()
    {
        foreach (var c in _cpuCaches) c.Reset();
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
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _blocks) b.Dispose();
        _embedding.Dispose(); _finalNormW.Dispose(); _cos?.Dispose(); _sin?.Dispose();
        foreach (var b in _kCache) b.Dispose();
        foreach (var b in _vCache) b.Dispose();
        foreach (var c in _cpuCaches) c.Dispose();
        _arena.Dispose();
        _cpuWorkspace.Dispose();
    }
}
