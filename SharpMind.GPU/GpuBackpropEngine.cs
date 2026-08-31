using SharpMind.Core;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
using SharpMind.Training;

namespace SharpMind.GPU;

/// <summary>
/// BackpropEngine on the device. Weights uploaded once; per-step tensors from one arena;
/// LoRA A/B re-uploaded each step, their grads accumulated into Parameter.Grad after
/// each backward. Mirrors BackpropEngine.ForwardAndRecord / Backward op for op.
///
/// The batch shape is fixed at construction — it sizes the arena — and a batch of any
/// other shape is rejected rather than silently truncated.
/// </summary>
public sealed class GpuBackpropEngine : ITrainingEngine
{
    private readonly GpuDevice _dev;
    private readonly ModelConfig _cfg;
    private readonly bool _gelu, _gemmaScale, _rope, _neox;
    private readonly bool _flash;
    private readonly int _batch, _seq;
    private readonly GpuBlock[] _blocks;
    private readonly DeviceBuffer _embedding, _finalNormW;
    private readonly DeviceBuffer? _cos, _sin;
    private readonly int _ropeDim;
    private readonly DeviceArena _arena;
    private readonly int[] _hostIds, _hostLabels;
    private readonly GpuStepProfiler _prof;
    private bool _disposed;

    /// <summary>
    /// Per-phase breakdown of <see cref="ForwardBackward"/>. Disabled unless <c>SM_PROF=1</c>;
    /// see <see cref="GpuStepProfiler"/> for why enabling it makes the step slower than the one
    /// it is reporting on.
    /// </summary>
    public GpuStepProfiler Profiler => _prof;

    /// <summary>
    /// The backend this engine actually runs on, for the UI. A real CUDA/OpenCL device shows
    /// <see cref="GpuDevice.Description"/> (e.g. <c>"[Cuda] ..., cuBLAS 12.8"</c>); an ILGPU CPU
    /// fallback device reports <c>"CPU"</c> so the display never claims GPU acceleration that
    /// isn't happening.
    /// </summary>
    public string Description => _dev.IsCpuFallback ? "CPU" : _dev.Description;

    /// <summary>Ignored-label id for the loss, as <c>CrossEntropyLoss.IgnoreId</c>.</summary>
    internal int IgnoreId { get; }
    /// <summary>Cross-entropy label smoothing, as <c>CrossEntropyLoss.LabelSmoothing</c>.</summary>
    internal float LabelSmoothing { get; }
    /// <summary>Arena floats handed out since the last reset — the measured peak of a step.</summary>
    internal long ArenaUsed => _arena.Used;

    /// <param name="flashAttention">
    /// Use <see cref="Kernels.FlashAttentionKernels"/> instead of the materialised
    /// <see cref="Kernels.AttentionKernels"/>: same values, but the per-block <c>[B·H·S, S]</c>
    /// probabilities — the arena's only S² term, kept for the whole step across every layer —
    /// are replaced by <c>[B·H·S, 3]</c> of softmax statistics, and the backward recomputes the
    /// probabilities from them. Trades score arithmetic for memory.
    /// </param>
    public GpuBackpropEngine(GpuDevice device, Transformer model, IReadOnlyList<Parameter> parameters, SharpMindConfig config,
        int batch, int seqLen, int ignoreId = -100, float labelSmoothing = 0f, bool flashAttention = false)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);
        ValidateSupported(model, parameters, config);
        _dev = device; _cfg = model.Config; _batch = batch; _seq = seqLen; IgnoreId = ignoreId; LabelSmoothing = labelSmoothing; _flash = flashAttention;
        _prof = new GpuStepProfiler(device);
        // The CPU forward picks its gate kernel from config.Gate (MappingBuilder's "gate" key)
        // while BackpropEngine's backward picks from config.Activation. They agree for both
        // shipped presets (SiLU+SwiGLU, GELU+GeGLU), so one flag mirrors both.
        _gelu = config.Gate == GateKind.GeGLU;
        _gemmaScale = config.Activation == ActivationKind.GELU && config.Gate == GateKind.GeGLU;
        // Tensor<float> does not override Equals/GetHashCode, so the default comparer is already
        // reference identity — the same dictionary BackpropEngine builds for the same lookup.
        var byTensor = new Dictionary<Tensor<float>, Parameter>();
        foreach (var p in parameters) byTensor[p.Data] = p;
        Parameter? Param(Tensor<float> t) => byTensor.TryGetValue(t, out var p) ? p : null;

        // The constructor never returns on a throw, so nothing could dispose what it had already
        // allocated — and DeviceBuffer has no finalizer. Track and unwind. ValidateSupported
        // closes the two known holes above; this covers the rest (an Allocate1D out of memory).
        var owned = new List<IDisposable>();
        try
        {
            _blocks = new GpuBlock[_cfg.NumLayers];
            for (int l = 0; l < _cfg.NumLayers; l++) { _blocks[l] = new GpuBlock(device, model.GetBlock(l)!, Param); owned.Add(_blocks[l]); }
            _embedding = DeviceBuffer.From(device, model.EmbeddingWeight); owned.Add(_embedding);
            _finalNormW = DeviceBuffer.From(device, model.FinalNorm.NormWeight); owned.Add(_finalNormW);
            if (model.GetBlock(0)!.Attention.PositionalEncoder is RoPE rope)
            {
                _rope = true; _neox = rope.NeoxStyle; _ropeDim = rope.RopeDim;
                _cos = new DeviceBuffer(device, rope.CosTable.Length / (_ropeDim / 2), _ropeDim / 2); owned.Add(_cos); _cos.Tensor.Upload(rope.CosTable);
                _sin = new DeviceBuffer(device, rope.SinTable.Length / (_ropeDim / 2), _ropeDim / 2); owned.Add(_sin); _sin.Tensor.Upload(rope.SinTable);
            }
            _hostIds = new int[batch * seqLen];
            _hostLabels = new int[batch * seqLen];
            int rank = _blocks.SelectMany(b => b.Linears()).Select(l => l.Rank).DefaultIfEmpty(0).Max();
            _arena = new DeviceArena(device, ArenaFloats(_cfg, batch, seqLen, rank, flashAttention));
        }
        catch
        {
            foreach (var d in owned) d.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Per-step activation budget, counted from the Forward/ForwardBackward code paths:
    /// the embedding x every block's residual runs through, then per block the forward keeps
    /// x, n1, rInv1, q, k, v, probs, attnOut, proj, x1, n2, rInv2, fused, act, down and six
    /// LoRA hs slots; the backward of one block (the arena is not reset between blocks, so
    /// every block's temporaries accumulate) rents dAct, dFused, dN2, dN2in, dAttnOut, dQ, dK,
    /// dV, scratch, dN1, dN1in and six dH slots. Plus the head — normed, rInv and logits
    /// forward, dN, dX and rowLoss backward. Both halves were counted rent for rent against
    /// ForwardBackward and measured at the test shape (69,216 exact + 96 padding). 25% headroom
    /// on top for the arena's 32-float alignment of every rent at other shapes.
    ///
    /// <paramref name="flash"/> replaces both S² terms — the forward's probs and the backward's
    /// equally sized dS scratch — with one [B·H·S, 3] statistics tensor per block, rented in the
    /// forward and written again (column 2) by the backward. That is the whole memory difference
    /// between the two attention paths, and the only term in this budget that grows as S².
    /// </summary>
    internal static long ArenaFloats(ModelConfig c, int batch, int seq, int rank, bool flash = false)
    {
        long m = (long)batch * seq, H = c.HiddenDim, qDim = (long)c.NumHeads * c.HeadDim, kvDim = (long)c.NumKvHeads * c.HeadDim, F = c.FfnDim, V = c.VocabSize;
        long rowsBHS = (long)batch * c.NumHeads * seq;
        long fwdAttn = flash ? rowsBHS * Kernels.FlashAttentionKernels.StatCols : rowsBHS * seq;
        long bwdAttn = flash ? 0 : rowsBHS * seq;   // flash reuses the forward's statistics tensor
        long entry = m * H;
        long fwdBlock = m * (6 * H + 2 + qDim * 2 + kvDim * 2 + 3 * F + 6 * rank) + fwdAttn;
        long bwdBlock = m * (4 * H + 3 * F + qDim * 2 + kvDim * 2 + 6 * rank) + bwdAttn;
        long head = m * (3 * H + V + 2);
        return (entry + (fwdBlock + bwdBlock) * c.NumLayers + head) * 5 / 4 + 4096;
    }

    public static void ValidateSupported(Transformer model, IReadOnlyList<Parameter> parameters, SharpMindConfig config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(config);
        var c = model.Config;
        static string Why(string s) => $"GPU engine (M1) does not support {s}; use the CPU engine (BackpropEngine), which does.";
        if (model.FinalNorm is not RmsNormLayer) throw new NotSupportedException(Why($"a {model.FinalNorm.GetType().Name} final norm — LayerNorm, only RMSNorm"));
        if (c.PositionalEncoding is not (PositionalEncoding.RoPE or PositionalEncoding.NoPE)) throw new NotSupportedException(Why($"positional encoding {c.PositionalEncoding}"));
        if (config.Ffn == FfnKind.MoE) throw new NotSupportedException(Why("MoE"));
        if (model.QuantAwareTrainingTarget is not null and not Core.Quantization.QuantDType.F32) throw new NotSupportedException(Why("quantization-aware training"));
        if (config.Gate == GateKind.None) throw new NotSupportedException(Why("dense (ungated) FFN"));
        if (c.NumLayers <= 0 || model.GetBlock(0) is null) throw new NotSupportedException(Why("a model without blocks"));

        var lora = new HashSet<Tensor<float>>(ReferenceEqualityComparer.Instance);
        var adapters = new List<(string Name, Tensor<float> A, Tensor<float> B)>();
        for (int l = 0; l < c.NumLayers; l++)
        {
            var b = model.GetBlock(l) ?? throw new NotSupportedException(Why($"a model missing block {l}"));
            if (b.PostAttnNorm is not null || b.PostFfnNorm is not null) throw new NotSupportedException(Why("Gemma post-attention/post-FFN norms"));
            if (b.Norm1 is not RmsNormLayer || b.Norm2 is not RmsNormLayer) throw new NotSupportedException(Why($"a {b.Norm1.GetType().Name} block norm — LayerNorm, only RMSNorm"));
            if (b.Ffn is not GatedFfnLayer) throw new NotSupportedException(Why($"FFN kind {b.Ffn.GetType().Name}"));
            foreach (var lin in AllLinears(b))
            {
                if (lin is not TrainingLinearLayer t)
                    throw new NotSupportedException(Why($"linear layer kind {lin.GetType().Name} (build the model with ModelFactory.CreateTrainingTransformer)"));
                if (!t.HasLoRA) continue;
                // GpuLinear rejects rank 1 (five of its eight GEMMs collapse to an ambiguous (1,1)
                // stride pair) — catch it here, before a single byte is allocated on the device.
                if (t.LoRARank < 2)
                    throw new NotSupportedException(Why($"LoRA rank {t.LoRARank} on '{lin.Name}' (the GPU adapter GEMMs need rank >= 2)"));
                lora.Add(t.LoRAA!); lora.Add(t.LoRAB!);
                adapters.Add((lin.Name, t.LoRAA!, t.LoRAB!));
            }
        }
        foreach (var p in parameters)
            if (!lora.Contains(p.Data)) throw new NotSupportedException(Why($"training '{p.Name}' (only LoRA adapters are trainable in M1)"));
        // Every adapter needs its Parameter: GpuBlock would otherwise hand GpuLinear a null A/B
        // and get an ArgumentException about GEMM strides from inside the constructor.
        var given = new HashSet<Tensor<float>>(parameters.Select(p => p.Data), ReferenceEqualityComparer.Instance);
        foreach (var (name, a, b) in adapters)
            if (!given.Contains(a) || !given.Contains(b))
                throw new NotSupportedException(Why($"an adapted layer whose LoRA parameters were not passed in ('{name}')"));
    }

    private static IEnumerable<LinearLayer> AllLinears(TransformerBlock b)
    {
        yield return b.Attention.Wq; yield return b.Attention.Wk; yield return b.Attention.Wv; yield return b.Attention.Wo;
        if (b.Ffn is GatedFfnLayer g) { yield return g.WGated!; yield return g.WDown!; }
    }

    // ── forward ──────────────────────────────────────────────────────────────
    private DeviceTensor Forward(DeviceIntBuffer ids, out DeviceTensor finalNormed, out DeviceTensor finalRInv, out DeviceTensor finalIn)
    {
        var k = _dev.Kernels; int m = _batch * _seq, H = _cfg.HiddenDim, D = _cfg.HeadDim, nh = _cfg.NumHeads, nkv = _cfg.NumKvHeads;
        var x = _arena.Rent(m, H);
        k.EmbedGather(x, _embedding.Tensor, ids.View);
        if (_gemmaScale) k.Scale(x, MathF.Sqrt(H));
        _prof.Mark("fwd/embed");
        foreach (var blk in _blocks)
        {
            blk.X = _arena.Rent(m, H); k.Copy(blk.X, x);                       // x_in (kept for the residual path's identity)
            _prof.Mark("fwd/copy");
            blk.Norm1Out = _arena.Rent(m, H); blk.RInv1 = _arena.Rent(m, 1);
            k.RmsNormFwd(blk.Norm1Out, blk.RInv1, x, blk.Norm1W.Tensor, _cfg.NormEps);
            _prof.Mark("fwd/norm");
            blk.Q = _arena.Rent(m, nh * D); blk.K = _arena.Rent(m, nkv * D); blk.V = _arena.Rent(m, nkv * D);
            blk.Wq.Forward(blk.Q, blk.Norm1Out, _arena); blk.Wk.Forward(blk.K, blk.Norm1Out, _arena); blk.Wv.Forward(blk.V, blk.Norm1Out, _arena);
            _prof.Mark("fwd/qkv-proj");
            if (_rope) { k.RopeFwd(blk.Q, _cos!.Tensor, _sin!.Tensor, _seq, nh, D, _ropeDim, _neox); k.RopeFwd(blk.K, _cos.Tensor, _sin.Tensor, _seq, nkv, D, _ropeDim, _neox); }
            _prof.Mark("fwd/rope");
            blk.AttnOut = _arena.Rent(m, nh * D);
            if (_flash)
            {
                // Every element the flash path reads back is written by the same launch that
                // produced it, so unlike Probs below there is no stale upper triangle to clear.
                blk.Stats = _arena.Rent(_batch * nh * _seq, Kernels.FlashAttentionKernels.StatCols);
                k.AttnFwdFlash(blk.AttnOut, blk.Stats, blk.Q, blk.K, blk.V, _batch, _seq, nh, nkv, D);
            }
            else
            {
                // No pre-zero: AttnFwd's softmax writes the whole row, upper triangle included,
                // so the previous step's probabilities in this arena memory cannot survive.
                blk.Probs = _arena.Rent(_batch * nh * _seq, _seq);
                k.AttnFwd(blk.AttnOut, blk.Probs, blk.Q, blk.K, blk.V, _batch, _seq, nh, nkv, D);
            }
            _prof.Mark("fwd/attn");                                            // both paths, so the two are comparable
            var proj = _arena.Rent(m, H); blk.Wo.Forward(proj, blk.AttnOut, _arena);
            _prof.Mark("fwd/wo");
            k.AddInPlace(x, proj);
            _prof.Mark("fwd/resid-add");
            blk.X1 = _arena.Rent(m, H); k.Copy(blk.X1, x);
            _prof.Mark("fwd/copy");
            blk.Norm2Out = _arena.Rent(m, H); blk.RInv2 = _arena.Rent(m, 1);
            k.RmsNormFwd(blk.Norm2Out, blk.RInv2, x, blk.Norm2W.Tensor, _cfg.NormEps);
            _prof.Mark("fwd/norm");
            blk.Fused = _arena.Rent(m, 2 * _cfg.FfnDim); blk.WGated.Forward(blk.Fused, blk.Norm2Out, _arena);
            _prof.Mark("fwd/ffn-gated");
            blk.Act = _arena.Rent(m, _cfg.FfnDim); k.GateFwd(blk.Act, blk.Fused, _gelu);
            _prof.Mark("fwd/gate");
            var down = _arena.Rent(m, H); blk.WDown.Forward(down, blk.Act, _arena);
            _prof.Mark("fwd/ffn-down");
            k.AddInPlace(x, down);
            _prof.Mark("fwd/resid-add");
        }
        finalIn = x;
        finalNormed = _arena.Rent(m, H); finalRInv = _arena.Rent(m, 1);
        k.RmsNormFwd(finalNormed, finalRInv, x, _finalNormW.Tensor, _cfg.NormEps);
        _prof.Mark("fwd/final-norm");
        var logits = _arena.Rent(m, _cfg.VocabSize);
        _dev.Gemm(logits, finalNormed, _embedding.Tensor, m, _cfg.VocabSize, H, saI: H, saK: 1, sbK: 1, sbJ: H);   // logits = n·Eᵀ, E [V,H]
        _prof.Mark("fwd/lm-head");
        return logits;
    }

    /// <summary>Logits for one batch as a host copy [Batch·SeqLen · VocabSize]. Test hook.</summary>
    internal float[] ForwardLogitsForTest(Tensor<int> tokenIds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(tokenIds);
        CheckShape(tokenIds, "tokenIds", nameof(tokenIds));
        _arena.Reset();
        foreach (var b in _blocks) foreach (var l in b.Linears()) l.SyncLoRAToDevice();
        tokenIds.Data.CopyTo(_hostIds);
        using var ids = _dev.UploadInts(_hostIds);
        var logits = Forward(ids, out _, out _, out _);
        _dev.Synchronize();
        return logits.ToArray();
    }

    /// <summary>
    /// The arena is sized for one shape and RoPE derives each token's position from _seq, so
    /// both dimensions must match — a [4,4] batch has the same token count as a [2,8] one and
    /// would come out silently wrong rather than short.
    /// </summary>
    /// <param name="what">The tensor's role, for the message.</param>
    /// <param name="paramName">The CALLER's parameter — a TrainingBatch arrives as one argument,
    /// so "labels" would name something no caller can pass.</param>
    private void CheckShape(Tensor<int> t, string what, string paramName)
    {
        if (t.Rank != 2 || t.Shape.Rows != _batch || t.Shape.Cols != _seq)
            throw new ArgumentException($"This engine is sized for a [{_batch}, {_seq}] batch, got {what} {t.Shape}.", paramName);
    }

    // ── backward ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One training step on the device: forward, fused CE loss + dLogits, then the reverse
    /// pass, op for op as <see cref="Training.Autograd.BackpropEngine.Backward"/> runs it.
    /// Returns the scalar loss and ADDS every adapter's gradient into its Parameter.Grad, so
    /// micro-batch accumulation works the same way it does on the host.
    ///
    /// Both halves share one arena fill: <see cref="GpuLinear.Forward"/> parks its s·(x·A) in
    /// arena memory for the matching <see cref="GpuLinear.Backward"/>, so the reset happens
    /// once, here at the top, and never between the two.
    ///
    /// The embedding and the (weight-tied) head are frozen in M1, so dX is never scattered back
    /// into the embedding table and the √H Gemma scale the CPU applies to dX just before that
    /// scatter has nothing left to affect.
    /// </summary>
    public float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        CheckShape(batch.TokenIds, "tokenIds", nameof(batch));
        CheckShape(batch.Labels, "labels", nameof(batch));
        cancellationToken.ThrowIfCancellationRequested();
        var k = _dev.Kernels;
        int m = _batch * _seq, H = _cfg.HiddenDim, D = _cfg.HeadDim, nh = _cfg.NumHeads, nkv = _cfg.NumKvHeads, F = _cfg.FfnDim, V = _cfg.VocabSize;

        _arena.Reset();
        _prof.BeginStep();
        // The optimizer moved A and B on the host since the last step; the device grads are
        // this step's alone (the host Parameter.Grad is what accumulates across micro-batches).
        foreach (var b in _blocks) foreach (var l in b.Linears()) { l.SyncLoRAToDevice(); l.ZeroLoRAGrads(); }
        _prof.Mark("step/lora-upload");
        batch.TokenIds.Data.CopyTo(_hostIds); batch.Labels.Data.CopyTo(_hostLabels);
        using var ids = _dev.UploadInts(_hostIds);
        using var labels = _dev.UploadInts(_hostLabels);
        _prof.Mark("step/ids-upload");

        var logits = Forward(ids, out _, out var finalRInv, out var finalIn);

        // Loss + dLogits, in place: logits becomes (softmax − u)/N.
        var rowLoss = _arena.Rent(m, 1);
        float loss = k.CeLossAndGrad(logits, labels.View, _hostLabels, rowLoss, IgnoreId, LabelSmoothing);
        _prof.Mark("loss/ce+dlogits");

        // Head (frozen, weight-tied): dN = dLogits · E, then the final norm.
        var dN = _arena.Rent(m, H);
        _dev.Gemm(dN, logits, _embedding.Tensor, m, H, V, saI: V, saK: 1, sbK: H, sbJ: 1);
        _prof.Mark("bwd/lm-head");
        var dX = _arena.Rent(m, H);
        k.RmsNormBwd(dX, dN, finalIn, finalRInv, _finalNormW.Tensor);
        _prof.Mark("bwd/final-norm");

        for (int l = _blocks.Length - 1; l >= 0; l--)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var blk = _blocks[l];
            // FFN: WDown → gate → WGated → norm2, then the residual's direct path.
            var dAct = _arena.Rent(m, F); blk.WDown.Backward(dAct, dX, blk.Act, _arena);
            _prof.Mark("bwd/ffn-down");
            var dFused = _arena.Rent(m, 2 * F); k.GateBwd(dFused, dAct, blk.Fused, _gelu);
            _prof.Mark("bwd/gate");
            var dN2 = _arena.Rent(m, H); blk.WGated.Backward(dN2, dFused, blk.Norm2Out, _arena);
            _prof.Mark("bwd/ffn-gated");
            // RmsNormBwd wants the norm's INPUT: X1 is x after the attention residual, which is
            // exactly what norm2 read in the forward. Its output (Norm2Out) would be wrong.
            var dN2in = _arena.Rent(m, H); k.RmsNormBwd(dN2in, dN2, blk.X1, blk.RInv2, blk.Norm2W.Tensor);
            _prof.Mark("bwd/norm");
            k.AddInPlace(dX, dN2in);
            _prof.Mark("bwd/resid-add");

            // Attention: Wo → attention → the inverse RoPE rotation → Wq/Wk/Wv → norm1.
            var dAttnOut = _arena.Rent(m, nh * D); blk.Wo.Backward(dAttnOut, dX, blk.AttnOut, _arena);
            _prof.Mark("bwd/wo");
            var dQ = _arena.Rent(m, nh * D); var dK = _arena.Rent(m, nkv * D); var dV = _arena.Rent(m, nkv * D);
            if (_flash)
            {
                // Writes the row constant into the statistics tensor's third column, then
                // recomputes the probabilities from it — no S² scratch to rent.
                k.AttnBwdFlash(dQ, dK, dV, dAttnOut, blk.AttnOut, blk.Q, blk.K, blk.V, blk.Stats, _batch, _seq, nh, nkv, D);
            }
            else
            {
                var scratch = _arena.Rent(_batch * nh * _seq, _seq);
                k.AttnBwd(dQ, dK, dV, dAttnOut, blk.Q, blk.K, blk.V, blk.Probs, scratch, _batch, _seq, nh, nkv, D);
            }
            _prof.Mark("bwd/attn");                                            // both paths, so the two are comparable
            // Q/K went into attention post-RoPE, so dQ/dK come out pre-inverse-rotation and the
            // rotation belongs here — after AttnBwd, before the Wq/Wk/Wv backwards read them.
            if (_rope) { k.RopeBwd(dQ, _cos!.Tensor, _sin!.Tensor, _seq, nh, D, _ropeDim, _neox); k.RopeBwd(dK, _cos.Tensor, _sin.Tensor, _seq, nkv, D, _ropeDim, _neox); }
            _prof.Mark("bwd/rope");
            var dN1 = _arena.Rent(m, H);
            blk.Wq.Backward(dN1, dQ, blk.Norm1Out, _arena);                  // starts the sum
            blk.Wk.Backward(dN1, dK, blk.Norm1Out, _arena, betaDx: 1f);      // accumulates
            blk.Wv.Backward(dN1, dV, blk.Norm1Out, _arena, betaDx: 1f);
            _prof.Mark("bwd/qkv");
            var dN1in = _arena.Rent(m, H); k.RmsNormBwd(dN1in, dN1, blk.X, blk.RInv1, blk.Norm1W.Tensor);
            _prof.Mark("bwd/norm");
            k.AddInPlace(dX, dN1in);
            _prof.Mark("bwd/resid-add");
        }

        _dev.Synchronize();
        foreach (var b in _blocks) foreach (var lin in b.Linears()) lin.AccumulateLoRAGradsToHost();
        _prof.Mark("step/grad-download");
        _prof.EndStep();
        return loss;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _blocks) b.Dispose();
        _embedding.Dispose(); _finalNormW.Dispose(); _cos?.Dispose(); _sin?.Dispose(); _arena.Dispose();
    }
}
