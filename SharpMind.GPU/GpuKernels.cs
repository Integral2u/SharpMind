using ILGPU;
using ILGPU.Runtime;
using SharpMind.GPU.Kernels;

namespace SharpMind.GPU;

/// <summary>Loaded kernels for one accelerator. Each method is one launch on the default stream; no synchronisation.</summary>
internal sealed class GpuKernels
{
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _add, _copy;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _addBias;
    private readonly Action<Index1D, ArrayView<float>, float> _scale;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int> _gather;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float> _rmsFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> _rmsBwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float> _rope;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _gateFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _gateBwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _attnFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> _attnBwdScores;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _attnBwdQ;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _attnBwdKv;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _flashRowDot;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashBwdQ;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashBwdKv;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, float, float> _ceRow;
    private float[]? _rowLossHost;

    internal GpuKernels(Accelerator acc)
    {
        _add = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.AddInPlace);
        _copy = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.Copy);
        _addBias = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(ElementwiseKernels.AddBiasRows);
        _scale = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(ElementwiseKernels.Scale);
        _gather = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int>(ElementwiseKernels.EmbedGather);
        _rmsFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(NormKernels.RmsNormFwd);
        _rmsBwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(NormKernels.RmsNormBwd);
        _rope = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float>(RopeKernels.Rope);
        _gateFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(GateKernels.Fwd);
        _gateBwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(GateKernels.Bwd);
        _attnFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(AttentionKernels.Fwd);
        _attnBwdScores = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(AttentionKernels.BwdScores);
        _attnBwdQ = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(AttentionKernels.BwdQ);
        _attnBwdKv = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(AttentionKernels.BwdKV);
        _flashFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.Fwd);
        _flashRowDot = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(FlashAttentionKernels.BwdRowDot);
        _flashBwdQ = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.BwdQ);
        _flashBwdKv = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.BwdKV);
        _ceRow = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, float, float>(LossKernels.CeRow);
    }

    public void AddInPlace(DeviceTensor dst, DeviceTensor src) { Same(dst, src); _add(dst.Length, dst.View, src.View); }
    public void Copy(DeviceTensor dst, DeviceTensor src) { Same(dst, src); _copy(dst.Length, dst.View, src.View); }
    public void AddBiasRows(DeviceTensor x, DeviceTensor bias) { if (bias.Length != x.Cols) throw new ArgumentException("bias length != cols"); _addBias(x.Length, x.View, bias.View, x.Cols); }
    public void Scale(DeviceTensor x, float s) => _scale(x.Length, x.View, s);
    public void EmbedGather(DeviceTensor x, DeviceTensor table, ArrayView<int> ids) { if (table.Cols != x.Cols || ids.Length != x.Rows) throw new ArgumentException("gather shapes"); _gather(x.Length, x.View, table.View, ids, x.Cols); }
    public void RmsNormFwd(DeviceTensor y, DeviceTensor rInv, DeviceTensor x, DeviceTensor w, float eps) { Same(y, x); CheckNormOperands(x, rInv, w); _rmsFwd(x.Rows, y.View, rInv.View, x.View, w.View, x.Cols, eps); }
    public void RmsNormBwd(DeviceTensor dx, DeviceTensor dy, DeviceTensor x, DeviceTensor rInv, DeviceTensor w) { Same(dx, dy); Same(dx, x); CheckNormOperands(x, rInv, w); _rmsBwd(x.Rows, dx.View, dy.View, x.View, rInv.View, w.View, x.Cols); }

    public void RopeFwd(DeviceTensor x, DeviceTensor cos, DeviceTensor sin, int seqLen, int numHeads, int headDim, int ropeDim, bool neox)
    {
        CheckRopeOperands(x, cos, sin, seqLen, numHeads, headDim, ropeDim);
        _rope(x.Rows * numHeads * (ropeDim / 2), x.View, cos.View, sin.View, seqLen, numHeads, headDim, ropeDim, neox ? 1 : 0, 1f);
    }

    public void RopeBwd(DeviceTensor x, DeviceTensor cos, DeviceTensor sin, int seqLen, int numHeads, int headDim, int ropeDim, bool neox)
    {
        CheckRopeOperands(x, cos, sin, seqLen, numHeads, headDim, ropeDim);
        _rope(x.Rows * numHeads * (ropeDim / 2), x.View, cos.View, sin.View, seqLen, numHeads, headDim, ropeDim, neox ? 1 : 0, -1f);
    }

    /// <summary>act = gate(g)·u, g/u read from fused's [M, 2F] columns (gate first, up last).</summary>
    public void GateFwd(DeviceTensor act, DeviceTensor fused, bool gelu)
    {
        CheckGateOperands(act, fused);
        _gateFwd(act.Length, act.View, fused.View, act.Cols, gelu ? 1 : 0);
    }

    /// <summary>dGate/dUp written into dFused's [M, 2F] columns from dAct and the forward's fused input.</summary>
    public void GateBwd(DeviceTensor dFused, DeviceTensor dAct, DeviceTensor fused, bool gelu)
    {
        Same(dFused, fused);
        CheckGateOperands(dAct, fused);
        _gateBwd(dAct.Length, dFused.View, dAct.View, fused.View, dAct.Cols, gelu ? 1 : 0);
    }

    /// <summary>
    /// Causal GQA attention forward: <paramref name="outp"/> [B·S, H·D] and the materialised
    /// <paramref name="probs"/> [B·H·S, S] (row (b·H + h)·S + i), scale 1/√headDim. Only j ≤ i
    /// of each probs row is written — zero the tensor first if the upper triangle must read 0.
    /// <paramref name="q"/>/<paramref name="k"/> are post-RoPE, as BackpropEngine's bc.Q/bc.K are:
    /// the caller applies the rotation before this call.
    /// </summary>
    public void AttnFwd(DeviceTensor outp, DeviceTensor probs, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckAttnOperands(q, k, v, probs, batch, seqLen, numHeads, numKv, headDim);
        Same(outp, q);
        _attnFwd(batch * numHeads * seqLen, outp.View, probs.View, q.View, k.View, v.View, seqLen, numHeads, numKv, headDim, 1f / MathF.Sqrt(headDim));
    }

    /// <summary>
    /// Causal GQA attention backward, three launches on the in-order default stream: dS into
    /// <paramref name="dProbsScratch"/>, then dQ, then dK/dV. Every output element is written
    /// exactly once by exactly one thread, so dQ/dK/dV need no pre-zeroing and are overwritten,
    /// not accumulated — the GQA group sum the CPU engine builds with per-head AddHead calls is
    /// done inside the single (b, kvHead, j) thread that owns the destination.
    /// Two aliases are rejected because a later launch still reads what an earlier one overwrites:
    /// <paramref name="dProbsScratch"/> over <paramref name="probs"/> (corrupts dV) and
    /// <paramref name="dQ"/> over <paramref name="q"/> (corrupts dK) — the latter is what an
    /// in-place dQ would look like, and Same(dQ, q) makes the two shape-identical. dK/dV may alias
    /// k/v: those are read in launches 1-2, before launch 3 writes. Nothing else may overlap.
    /// <paramref name="q"/>/<paramref name="k"/> are post-RoPE, so dQ/dK come out pre-inverse-
    /// rotation: the caller owns the RoPE backward, as BackpropEngine.AttentionBackward does.
    /// </summary>
    public void AttnBwd(DeviceTensor dQ, DeviceTensor dK, DeviceTensor dV, DeviceTensor dOut, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        DeviceTensor probs, DeviceTensor dProbsScratch, int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckAttnOperands(q, k, v, probs, batch, seqLen, numHeads, numKv, headDim);
        Same(dQ, q); Same(dOut, q); Same(dK, k); Same(dV, k);
        Same(dProbsScratch, probs);
        NoOverlap(dProbsScratch, probs, nameof(dProbsScratch), nameof(probs));
        NoOverlap(dQ, q, nameof(dQ), nameof(q));
        int qRows = batch * numHeads * seqLen;
        float scale = 1f / MathF.Sqrt(headDim);
        _attnBwdScores(qRows, dProbsScratch.View, dOut.View, v.View, probs.View, seqLen, numHeads, numKv, headDim);
        _attnBwdQ(qRows, dQ.View, dProbsScratch.View, k.View, seqLen, numHeads, numKv, headDim, scale);
        _attnBwdKv(batch * numKv * seqLen, dK.View, dV.View, dProbsScratch.View, probs.View, q.View, dOut.View, seqLen, numHeads, numKv, headDim, scale);
    }

    /// <summary>
    /// Flash-style causal GQA attention forward: same values as <see cref="AttnFwd"/>, but the
    /// only per-row state kept for the backward is <paramref name="stats"/> [B·H·S, 3] (row
    /// (b·H + h)·S + i; columns max, sum-of-exp, and the backward's row constant, written later
    /// by <see cref="AttnBwdFlash"/>). No [B·H·S, S] probabilities exist at any point, so unlike
    /// <see cref="AttnFwd"/> there is nothing to pre-zero. q/k are post-RoPE, as there.
    /// </summary>
    public void AttnFwdFlash(DeviceTensor outp, DeviceTensor stats, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckQkv(q, k, v, batch, seqLen, numHeads, numKv, headDim);
        CheckStats(stats, batch, seqLen, numHeads);
        Same(outp, q);
        NoOverlap(outp, k, nameof(outp), nameof(k));
        NoOverlap(outp, v, nameof(outp), nameof(v));
        _flashFwd(batch * numHeads * seqLen, outp.View, stats.View, q.View, k.View, v.View, seqLen, numHeads, numKv, headDim, 1f / MathF.Sqrt(headDim));
    }

    /// <summary>
    /// Flash-style causal GQA attention backward, three launches on the in-order default stream:
    /// the row constant into <paramref name="stats"/> column 2, then dQ, then dK/dV. Each output
    /// element is written by exactly one thread, so dQ/dK/dV are overwritten rather than
    /// accumulated and need no pre-zeroing, as in <see cref="AttnBwd"/>.
    ///
    /// <paramref name="outp"/> is the forward's attention output for the same rows: the row
    /// constant is Σ_d dO·O, which the materialised backward gets from P instead.
    ///
    /// Aliasing is stricter than <see cref="AttnBwd"/>. There k and v are fully read before
    /// launch 3 writes dK/dV, so those may alias; here launch 3 re-reads k and v inside the loop
    /// that writes dK/dV, so they may not. dQ over q is rejected for the same reason as there.
    /// q/k are post-RoPE, so dQ/dK come out pre-inverse-rotation and the caller owns the RoPE
    /// backward.
    /// </summary>
    public void AttnBwdFlash(DeviceTensor dQ, DeviceTensor dK, DeviceTensor dV, DeviceTensor dOut, DeviceTensor outp,
        DeviceTensor q, DeviceTensor k, DeviceTensor v, DeviceTensor stats,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckQkv(q, k, v, batch, seqLen, numHeads, numKv, headDim);
        CheckStats(stats, batch, seqLen, numHeads);
        Same(dQ, q); Same(dOut, q); Same(outp, q); Same(dK, k); Same(dV, k);
        NoOverlap(dQ, q, nameof(dQ), nameof(q));
        NoOverlap(dK, k, nameof(dK), nameof(k));
        NoOverlap(dK, v, nameof(dK), nameof(v));
        NoOverlap(dV, k, nameof(dV), nameof(k));
        NoOverlap(dV, v, nameof(dV), nameof(v));
        int qRows = batch * numHeads * seqLen;
        float scale = 1f / MathF.Sqrt(headDim);
        _flashRowDot(qRows, stats.View, dOut.View, outp.View, seqLen, numHeads, headDim);
        _flashBwdQ(qRows, dQ.View, dOut.View, q.View, k.View, v.View, stats.View, seqLen, numHeads, numKv, headDim, scale);
        _flashBwdKv(batch * numKv * seqLen, dK.View, dV.View, dOut.View, q.View, k.View, v.View, stats.View, seqLen, numHeads, numKv, headDim, scale);
    }

    /// <summary>
    /// Fused cross-entropy loss + gradient. In place: <paramref name="logits"/> becomes
    /// <c>dLogits = (softmax − u)/N</c> (rows with <paramref name="ignoreId"/> zeroed);
    /// <paramref name="rowLoss"/> [rows,1] receives each row's loss (0 for ignored). Formulas:
    /// <c>CrossEntropyLoss.Compute</c> and <c>Gradients.CrossEntropySoftmax</c>.
    /// <paramref name="hostLabels"/> is the host copy of the same ids as <paramref name="labels"/>
    /// — the engine has it anyway — used to count N without a device round trip for the count.
    /// The one exception to "no synchronisation per call": the caller needs the scalar loss back
    /// on the host, so this both <see cref="GpuDevice.Synchronize"/>s and downloads.
    /// </summary>
    public float CeLossAndGrad(DeviceTensor logits, ArrayView<int> labels, int[] hostLabels, DeviceTensor rowLoss, int ignoreId, float labelSmoothing)
    {
        CheckCeOperands(logits, labels, hostLabels, rowLoss, labelSmoothing);
        int n = 0;
        foreach (int l in hostLabels) if (l != ignoreId) n++;
        float scale = n > 0 ? 1f / n : 0f;
        _ceRow(logits.Rows, logits.View, labels, rowLoss.View, logits.Cols, ignoreId, labelSmoothing, scale);
        rowLoss.View.GetAccelerator().Synchronize();
        if (_rowLossHost is null || _rowLossHost.Length < rowLoss.Rows) _rowLossHost = new float[rowLoss.Rows];
        var host = _rowLossHost.AsSpan(0, rowLoss.Rows);
        rowLoss.Download(host);
        double sum = 0.0;
        foreach (float x in host) sum += x;
        return n > 0 ? (float)(sum / n) : 0f;
    }

    private static void CheckCeOperands(DeviceTensor logits, ArrayView<int> labels, int[] hostLabels, DeviceTensor rowLoss, float labelSmoothing)
    {
        if (rowLoss.Rows != logits.Rows || rowLoss.Cols != 1) throw new ArgumentException($"rowLoss must be [{logits.Rows},1], got [{rowLoss.Rows},{rowLoss.Cols}].");
        if (labels.Length != logits.Rows) throw new ArgumentException($"labels length {labels.Length} != logits rows {logits.Rows}.");
        if (hostLabels.Length != logits.Rows) throw new ArgumentException($"hostLabels length {hostLabels.Length} != logits rows {logits.Rows}.");
        if (labelSmoothing < 0f || labelSmoothing >= 1f) throw new ArgumentException($"labelSmoothing {labelSmoothing} must be in [0,1).");
    }

    /// <summary>Rejects a written tensor that shares device memory with one still to be read.
    /// ArrayView.Index is not public in ILGPU 1.5.3, so the window is identified by its effective
    /// device address — the same trick DeviceArenaTests uses.</summary>
    internal static unsafe void NoOverlap(DeviceTensor written, DeviceTensor read, string writtenName, string readName)
    {
        long w = (long)written.View.LoadEffectiveAddressAsPtr(), r = (long)read.View.LoadEffectiveAddressAsPtr();
        if (w < r + (long)read.Length * sizeof(float) && r < w + (long)written.Length * sizeof(float))
            throw new ArgumentException($"{writtenName} overlaps {readName} in device memory; {writtenName} is written before {readName} is read for the last time.");
    }

    private static void CheckAttnOperands(DeviceTensor q, DeviceTensor k, DeviceTensor v, DeviceTensor probs,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckQkv(q, k, v, batch, seqLen, numHeads, numKv, headDim);
        if (probs.Rows != batch * numHeads * seqLen) throw new ArgumentException($"probs rows {probs.Rows} != batch*numHeads*seqLen {batch * numHeads * seqLen}.");
        if (probs.Cols != seqLen) throw new ArgumentException($"probs cols {probs.Cols} != seqLen {seqLen}.");
    }

    private static void CheckQkv(DeviceTensor q, DeviceTensor k, DeviceTensor v,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        if (batch <= 0 || seqLen <= 0 || numHeads <= 0 || numKv <= 0 || headDim <= 0)
            throw new ArgumentException($"batch/seqLen/numHeads/numKv/headDim must be positive, got {batch}/{seqLen}/{numHeads}/{numKv}/{headDim}.");
        if (numHeads % numKv != 0) throw new ArgumentException($"numHeads {numHeads} must be a multiple of numKv {numKv}.");
        int rows = batch * seqLen;
        if (q.Rows != rows) throw new ArgumentException($"q rows {q.Rows} != batch*seqLen {rows}.");
        if (k.Rows != rows || v.Rows != rows) throw new ArgumentException($"k/v rows {k.Rows}/{v.Rows} != batch*seqLen {rows}.");
        if (q.Cols != numHeads * headDim) throw new ArgumentException($"q cols {q.Cols} != numHeads*headDim {numHeads * headDim}.");
        if (k.Cols != numKv * headDim) throw new ArgumentException($"k cols {k.Cols} != numKv*headDim {numKv * headDim}.");
        if (v.Cols != numKv * headDim) throw new ArgumentException($"v cols {v.Cols} != numKv*headDim {numKv * headDim}.");
    }

    private static void CheckStats(DeviceTensor stats, int batch, int seqLen, int numHeads)
    {
        int rows = batch * numHeads * seqLen;
        if (stats.Rows != rows) throw new ArgumentException($"stats rows {stats.Rows} != batch*numHeads*seqLen {rows}.");
        if (stats.Cols != Kernels.FlashAttentionKernels.StatCols) throw new ArgumentException($"stats cols {stats.Cols} != {Kernels.FlashAttentionKernels.StatCols}.");
    }

    private static void CheckGateOperands(DeviceTensor act, DeviceTensor fused)
    {
        if (fused.Cols != 2 * act.Cols) throw new ArgumentException($"fused cols {fused.Cols} != 2*act cols {2 * act.Cols}.");
        if (fused.Rows != act.Rows) throw new ArgumentException($"fused rows {fused.Rows} != act rows {act.Rows}.");
    }

    private static void CheckRopeOperands(DeviceTensor x, DeviceTensor cos, DeviceTensor sin, int seqLen, int numHeads, int headDim, int ropeDim)
    {
        if (x.Cols != numHeads * headDim) throw new ArgumentException($"x cols {x.Cols} != numHeads*headDim {numHeads * headDim}.");
        if (ropeDim <= 0 || ropeDim > headDim || ropeDim % 2 != 0) throw new ArgumentException($"ropeDim {ropeDim} must be even, positive and <= headDim {headDim}.");
        int pairs = ropeDim / 2;
        if (cos.Cols != pairs) throw new ArgumentException($"cos cols {cos.Cols} != ropeDim/2 {pairs}.");
        if (sin.Cols != pairs) throw new ArgumentException($"sin cols {sin.Cols} != ropeDim/2 {pairs}.");
        if (cos.Rows < seqLen) throw new ArgumentException($"cos rows {cos.Rows} < seqLen {seqLen} (positions actually indexed).");
        if (sin.Rows < seqLen) throw new ArgumentException($"sin rows {sin.Rows} < seqLen {seqLen} (positions actually indexed).");
    }

    private static void Same(DeviceTensor a, DeviceTensor b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException($"Shape mismatch [{a.Rows},{a.Cols}] vs [{b.Rows},{b.Cols}].");
    }

    private static void CheckNormOperands(DeviceTensor x, DeviceTensor rInv, DeviceTensor w)
    {
        if (w.Length != x.Cols) throw new ArgumentException($"w length {w.Length} != cols {x.Cols}.");
        if (rInv.Rows != x.Rows || rInv.Cols != 1) throw new ArgumentException($"rInv must be [{x.Rows},1], got [{rInv.Rows},{rInv.Cols}].");
    }
}
