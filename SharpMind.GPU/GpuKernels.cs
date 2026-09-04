using ILGPU;
using ILGPU.Runtime;
using SharpMind.Core.Quantization;
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
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, float> _ropePos;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _gateFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _gateBwd;
    private readonly Action<Index1D, ArrayView<float>, int, float> _softmaxRow;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, float> _softmaxRowBwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashFwd;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, float> _flashFwdKvLen;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float> _flashPartialKvLen;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> _flashMergeKvLen;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _flashRowDot;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashBwdQ;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float> _flashBwdKv;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, float, float> _ceRow;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<byte>, int, int, int, QuantDType> _dequantMatmul;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, int, int, QuantDType> _dequantGather;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<byte>, int, int, int, QuantDType> _dequantMatmulK;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, int, QuantDType> _dequantGatherK;
    private float[]? _rowLossHost;
    private readonly GpuDevice _dev;

    internal GpuKernels(GpuDevice device, Accelerator acc)
    {
        _dev = device;
        _add = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.AddInPlace);
        _copy = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ElementwiseKernels.Copy);
        _addBias = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(ElementwiseKernels.AddBiasRows);
        _scale = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, float>(ElementwiseKernels.Scale);
        _gather = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int>(ElementwiseKernels.EmbedGather);
        _rmsFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, float>(NormKernels.RmsNormFwd);
        _rmsBwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(NormKernels.RmsNormBwd);
        _rope = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, float>(RopeKernels.Rope);
        _ropePos = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, float>(RopeKernels.RopePos);
        _gateFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(GateKernels.Fwd);
        _gateBwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(GateKernels.Bwd);
        _softmaxRow = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, int, float>(AttentionKernels.SoftmaxRow);
        _softmaxRowBwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, float>(AttentionKernels.SoftmaxRowBwd);
        _flashFwd = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.Fwd);
        _flashFwdKvLen = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, float>(FlashAttentionKernels.FwdKvLen);
        _flashPartialKvLen = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float>(FlashChunkedAttentionKernels.PartialKvLen);
        _flashMergeKvLen = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(FlashChunkedAttentionKernels.MergeKvLen);
        _flashRowDot = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(FlashAttentionKernels.BwdRowDot);
        _flashBwdQ = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.BwdQ);
        _flashBwdKv = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, float>(FlashAttentionKernels.BwdKV);
        _ceRow = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int, float, float>(LossKernels.CeRow);
        _dequantMatmul = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<byte>, int, int, int, QuantDType>(QuantMatmulKernels.DequantMatmul);
        _dequantGather = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, int, int, QuantDType>(QuantMatmulKernels.DequantGather);
        _dequantMatmulK = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<byte>, int, int, int, QuantDType>(QuantMatmulKernels.DequantMatmulK);
        _dequantGatherK = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<int>, int, QuantDType>(QuantMatmulKernels.DequantGatherK);
    }

    public void AddInPlace(DeviceTensor dst, DeviceTensor src) { Same(dst, src); _add(dst.Length, dst.View, src.View); }
    public void Copy(DeviceTensor dst, DeviceTensor src) { Same(dst, src); _copy(dst.Length, dst.View, src.View); }
    public void AddBiasRows(DeviceTensor x, DeviceTensor bias) { if (bias.Length != x.Cols) throw new ArgumentException("bias length != cols"); _addBias(x.Length, x.View, bias.View, x.Cols); }
    public void Scale(DeviceTensor x, float s) => _scale(x.Length, x.View, s);
    public void EmbedGather(DeviceTensor x, DeviceTensor table, ArrayView<int> ids) { if (table.Cols != x.Cols || ids.Length != x.Rows) throw new ArgumentException("gather shapes"); _gather(x.Length, x.View, table.View, ids, x.Cols); }

    /// <summary>
    /// y[i,o] = Σ_k x[i,k]·wQ[o,k] for a [N, K] quantized matrix in raw bytes — the block quants
    /// (<see cref="QuantMatmulKernels.DequantMatmul"/>, 32-element blocks) and the K-quants
    /// (<see cref="QuantMatmulKernels.DequantMatmulK"/>, 256-element super-blocks). One thread per
    /// output element; the reduction order replicates the matching CPU <c>VecDot*_Scalar</c> oracle.
    /// Used for every quantized linear (y=[m,Out], x=[m,In], w=the layer's raw weights, K=In, N=Out)
    /// and the weight-tied LM head (y=[m,Vocab], x=[m,Hidden], w=the embedding's raw bytes,
    /// K=Hidden, N=Vocab).
    /// </summary>
    public void DequantMatmul(DeviceTensor y, DeviceTensor x, DeviceByteBuffer w, int K, int N, QuantDType q)
    {
        if (x.Cols != K) throw new ArgumentException($"x cols {x.Cols} != K {K}.");
        if (y.Rows != x.Rows || y.Cols != N) throw new ArgumentException($"y must be [{x.Rows},{N}], got [{y.Rows},{y.Cols}].");
        bool k = QuantMatmulKernels.IsKQuant(q);
        int qk = k ? QuantMatmulKernels.QK_K : QuantMatmulKernels.QK;
        int nBlocks = (K + qk - 1) / qk;
        long expect = (long)N * nBlocks * (k ? QuantMatmulKernels.BlockBytesK(q) : QuantMatmulKernels.BlockBytes(q));
        if (w.Length < expect) throw new ArgumentException($"raw {q} weights hold {w.Length} bytes, need {expect} for [{N},{K}].");
        if (k) _dequantMatmulK(y.Length, y.View, x.View, w.View, K, N, nBlocks, q);
        else _dequantMatmul(y.Length, y.View, x.View, w.View, K, N, nBlocks, q);
    }

    /// <summary>
    /// x[i,d] = dequant(embedding[ids[i]][d]) from a [V, K] quantized embedding table in raw bytes
    /// — the block quants (<see cref="QuantMatmulKernels.DequantGather"/>) and the K-quants
    /// (<see cref="QuantMatmulKernels.DequantGatherK"/>). The embedding row for a token is the
    /// same output column the LM head matmul reads, so one physical table serves both.
    /// </summary>
    public void DequantGather(DeviceTensor x, DeviceByteBuffer table, ArrayView<int> ids, int K, QuantDType q)
    {
        if (ids.Length != x.Rows) throw new ArgumentException($"ids length {ids.Length} != x rows {x.Rows}.");
        if (x.Cols != K) throw new ArgumentException($"x cols {x.Cols} != K {K}.");
        bool k = QuantMatmulKernels.IsKQuant(q);
        int qk = k ? QuantMatmulKernels.QK_K : QuantMatmulKernels.QK;
        int nBlocks = (K + qk - 1) / qk;
        long need = (long)x.Rows * nBlocks * (k ? QuantMatmulKernels.BlockBytesK(q) : QuantMatmulKernels.BlockBytes(q));
        if (table.Length < need) throw new ArgumentException($"raw {q} embedding holds {table.Length} bytes, need {need} of table.");
        if (k) _dequantGatherK(x.Length, x.View, table.View, ids, K, q);
        else _dequantGather(x.Length, x.View, table.View, ids, K, nBlocks, q);
    }

    /// <summary>Q8_0 convenience over <see cref="DequantMatmul"/>; see <see cref="QuantMatmulKernels.Q8_0Matmul"/>.</summary>
    public void Q8_0Matmul(DeviceTensor y, DeviceTensor x, DeviceByteBuffer w, int K, int N)
        => DequantMatmul(y, x, w, K, N, QuantDType.Q8_0);

    /// <summary>Q8_0 convenience over <see cref="DequantGather"/>; see <see cref="QuantMatmulKernels.Q8_0Gather"/>.</summary>
    public void EmbedGatherQ8_0(DeviceTensor x, DeviceByteBuffer table, ArrayView<int> ids, int K)
        => DequantGather(x, table, ids, K, QuantDType.Q8_0);
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

    /// <summary>
    /// Position-offset forward RoPE: rows are rotated at absolute positions [pos0, pos0+seqLen)
    /// instead of [0, seqLen). <paramref name="x"/> has <paramref name="seqLen"/> rows; cos/sin
    /// are the full [MaxSeqLen, ropeDim/2] tables. Inference decode and continued prefill rotate
    /// a fresh row at <c>pos0 = cache length</c>. No inverse variant (no backward in inference).
    /// </summary>
    public void RopeFwdPos(DeviceTensor x, DeviceTensor cos, DeviceTensor sin, int seqLen, int pos0, int numHeads, int headDim, int ropeDim, bool neox)
    {
        CheckRopeOperands(x, cos, sin, seqLen, numHeads, headDim, ropeDim);
        int pairs = ropeDim / 2;
        if (cos.Rows < pos0 + seqLen) throw new ArgumentException($"cos rows {cos.Rows} < pos0+seqLen {pos0 + seqLen} (positions actually indexed).");
        if (sin.Rows < pos0 + seqLen) throw new ArgumentException($"sin rows {sin.Rows} < pos0+seqLen {pos0 + seqLen} (positions actually indexed).");
        _ropePos(x.Rows * numHeads * pairs, x.View, cos.View, sin.View, seqLen, pos0, numHeads, headDim, ropeDim, neox ? 1 : 0, 1f);
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
    /// <paramref name="probs"/> [B·H·S, S] (row (b·H + h)·S + i), scale 1/√headDim. The whole
    /// probs row is written, the upper triangle as zero, so the caller need not pre-zero it.
    /// <paramref name="q"/>/<paramref name="k"/> are post-RoPE, as BackpropEngine's bc.Q/bc.K are:
    /// the caller applies the rotation before this call.
    ///
    /// Two GEMMs per (b, h) around one softmax launch: scores = Q·Kᵀ into the head's probs block,
    /// then out = P·V straight into the head's column band of <paramref name="outp"/> (hence the
    /// ldc). Both are ordinary strided products, so on CUDA they are cuBLAS calls.
    /// </summary>
    public void AttnFwd(DeviceTensor outp, DeviceTensor probs, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        int batch, int seqLen, int numHeads, int numKv, int headDim)
    {
        CheckAttnOperands(q, k, v, probs, batch, seqLen, numHeads, numKv, headDim);
        Same(outp, q);
        int grp = numHeads / numKv, qDim = numHeads * headDim, kvDim = numKv * headDim;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                // scores[i,j] = Σ_d Q[i,d]·K[j,d]: K is read transposed by its strides, not moved.
                _dev.Gemm(HeadScores(probs, b, h, seqLen, numHeads), Head(q, b, h, seqLen, qDim, headDim), Head(k, b, h / grp, seqLen, kvDim, headDim),
                    seqLen, seqLen, headDim, saI: qDim, saK: 1, sbK: 1, sbJ: kvDim);
            }
        _softmaxRow(batch * numHeads * seqLen, probs.View, seqLen, 1f / MathF.Sqrt(headDim));
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
            {
                // out[i,d] = Σ_j P[i,j]·V[j,d], written into head h's columns of a [B·S, H·D] tensor.
                _dev.Gemm(Head(outp, b, h, seqLen, qDim, headDim), HeadScores(probs, b, h, seqLen, numHeads), Head(v, b, h / grp, seqLen, kvDim, headDim),
                    seqLen, headDim, seqLen, saI: seqLen, saK: 1, sbK: kvDim, sbJ: 1, ldc: qDim);
            }
    }

    /// <summary>One head's [seqLen, headDim] block of a [B·S, heads·headDim] tensor, as the flat
    /// window that spans it: rows a <paramref name="stride"/> apart, headDim wide.</summary>
    private static DeviceTensor Head(DeviceTensor t, int b, int head, int seqLen, int stride, int headDim)
        => t.Window((long)b * seqLen * stride + (long)head * headDim, (long)(seqLen - 1) * stride + headDim);

    /// <summary>One head's [seqLen, seqLen] score block of a [B·H·S, S] tensor. Dense, unlike
    /// <see cref="Head"/>, because probs is laid out head-major.</summary>
    private static DeviceTensor HeadScores(DeviceTensor probs, int b, int h, int seqLen, int numHeads)
        => probs.Slice((b * numHeads + h) * seqLen, seqLen);

    /// <summary>
    /// Causal GQA attention backward, in four stages on the in-order default stream: dP = dO·Vᵀ
    /// into <paramref name="dProbsScratch"/>, the softmax backward that turns it into dS in place,
    /// then dQ, then dK/dV. Every one of those products is a GEMM per (b, h); only the softmax
    /// backward is a kernel. dQ/dK/dV are overwritten, not accumulated, so they need no
    /// pre-zeroing — the GQA group sum the CPU engine builds with per-head AddHead calls is here
    /// the beta = 1 of the second and later GEMMs writing one kv head's block.
    ///
    /// The stage order is what makes the aliasing rules below hold, so it is not free to change:
    /// v is fully read in stage 1 and k in stage 3, both before stage 4 writes dV and dK.
    /// Two aliases are rejected because a later stage still reads what an earlier one overwrites:
    /// <paramref name="dProbsScratch"/> over <paramref name="probs"/> (corrupts dV) and
    /// <paramref name="dQ"/> over <paramref name="q"/> (corrupts dK) — the latter is what an
    /// in-place dQ would look like, and Same(dQ, q) makes the two shape-identical. dK/dV may alias
    /// k/v. Nothing else may overlap.
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
        int grp = numHeads / numKv, qDim = numHeads * headDim, kvDim = numKv * headDim;

        // 1: dP[i,j] = Σ_d dO[i,d]·V[j,d] — V transposed by its strides.
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                _dev.Gemm(HeadScores(dProbsScratch, b, h, seqLen, numHeads), Head(dOut, b, h, seqLen, qDim, headDim), Head(v, b, h / grp, seqLen, kvDim, headDim),
                    seqLen, seqLen, headDim, saI: qDim, saK: 1, sbK: 1, sbJ: kvDim);

        // 2: dP → dS, scale folded in so stages 3 and 4 are plain products.
        _softmaxRowBwd(batch * numHeads * seqLen, dProbsScratch.View, probs.View, seqLen, 1f / MathF.Sqrt(headDim));

        // 3: dQ[i,d] = Σ_j dS[i,j]·K[j,d].
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                _dev.Gemm(Head(dQ, b, h, seqLen, qDim, headDim), HeadScores(dProbsScratch, b, h, seqLen, numHeads), Head(k, b, h / grp, seqLen, kvDim, headDim),
                    seqLen, headDim, seqLen, saI: seqLen, saK: 1, sbK: kvDim, sbJ: 1, ldc: qDim);

        // 4: dK[j,d] = Σ_{h∈group} Σ_i dS[i,j]·Q[i,d] and dV[j,d] = Σ_{h∈group} Σ_i P[i,j]·dO[i,d].
        // dS and P are read transposed by their strides; the group sum is the beta below.
        for (int b = 0; b < batch; b++)
            for (int kvh = 0; kvh < numKv; kvh++)
                for (int g = 0; g < grp; g++)
                {
                    int h = kvh * grp + g;
                    float beta = g == 0 ? 0f : 1f;
                    _dev.Gemm(Head(dK, b, kvh, seqLen, kvDim, headDim), HeadScores(dProbsScratch, b, h, seqLen, numHeads), Head(q, b, h, seqLen, qDim, headDim),
                        seqLen, headDim, seqLen, saI: 1, saK: seqLen, sbK: qDim, sbJ: 1, beta: beta, ldc: kvDim);
                    _dev.Gemm(Head(dV, b, kvh, seqLen, kvDim, headDim), HeadScores(probs, b, h, seqLen, numHeads), Head(dOut, b, h, seqLen, qDim, headDim),
                        seqLen, headDim, seqLen, saI: 1, saK: seqLen, sbK: qDim, sbJ: 1, beta: beta, ldc: kvDim);
                }
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
    /// Positioned, KV-length-forward flash attention for inference: queries Q [qLen, H·D] at
    /// absolute positions [pos0, pos0+qLen) attend the contiguous cache K/V [kvLen, kvH·D]
    /// (kvLen = pos0 + qLen) with causal mask <c>j &lt;= pos0+i</c>. Single batch. Used by decode
    /// (<c>pos0 = c, qLen = 1, kvLen = c+1</c>) and continued prefill (<c>pos0 = c, qLen = p</c>).
    /// Same numbers as <see cref="AttnFwdFlash"/>, no probabilities materialised; q/k post-RoPE.
    /// </summary>
    public void AttnFwdKvLen(DeviceTensor outp, DeviceTensor stats, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        int pos0, int qLen, int kvLen, int numHeads, int numKv, int headDim)
    {
        int qDim = numHeads * headDim, kvDim = numKv * headDim;
        if (pos0 < 0 || qLen <= 0 || kvLen <= 0 || kvLen != pos0 + qLen) throw new ArgumentException($"pos0 {pos0}, qLen {qLen}, kvLen {kvLen}: need kvLen == pos0 + qLen > 0.");
        if (numHeads % numKv != 0) throw new ArgumentException($"numHeads {numHeads} must be a multiple of numKv {numKv}.");
        if (outp.Rows != qLen || outp.Cols != qDim) throw new ArgumentException($"outp must be [{qLen},{qDim}], got [{outp.Rows},{outp.Cols}].");
        if (q.Rows != qLen || q.Cols != qDim) throw new ArgumentException($"q must be [{qLen},{qDim}], got [{q.Rows},{q.Cols}].");
        if (k.Rows != kvLen || k.Cols != kvDim) throw new ArgumentException($"k must be [{kvLen},{kvDim}], got [{k.Rows},{k.Cols}].");
        if (v.Rows != kvLen || v.Cols != kvDim) throw new ArgumentException($"v must be [{kvLen},{kvDim}], got [{v.Rows},{v.Cols}].");
        StatsColsOrThrow(stats, qLen, numHeads);
        // NoOverlap(outp, k/v) is deliberately omitted here: in inference the output is always an
        // arena rent while k/v are always slices of the persistent device KV cache — two different
        // allocations, so aliasing is impossible by construction. The generic range check would
        // false-positive as the cache grows toward the arena's address span (larger kvLen extends
        // the k/v view upward into the region the small arena output occupies), which is why the
        // shared AttnFwd/AttnBwd path, where all operands share one arena, has this and it does not.
        _flashFwdKvLen(numHeads * qLen, outp.View, stats.View, q.View, k.View, v.View, pos0, qLen, kvLen, numHeads, numKv, headDim, 1f / MathF.Sqrt(headDim));
    }

    /// <summary>
    /// Split-K variant of <see cref="AttnFwdKvLen"/>, the same positioned flash attention for
    /// inference but run as two launches: <see cref="Kernels.FlashChunkedAttentionKernels.PartialKvLen"/>
    /// computes per-chunk partials in parallel (one thread per (h, i, chunk) — numHeads·qLen·numChunks
    /// threads, so decode's numHeads-thread launch grows to numHeads·numChunks), then
    /// <see cref="Kernels.FlashChunkedAttentionKernels.MergeKvLen"/> reduces them back to the real
    /// output and statistics with one thread per (h, i). Both launches are friends on the in-order
    /// default stream. Same numbers as <see cref="AttnFwdKvLen"/>, so <paramref name="partialOut"/>/
    /// <paramref name="partialStat"/> are chunk-private scratch: [numHeads·qLen·numChunks, headDim]
    /// and [numHeads·qLen·numChunks, 2].
    ///
    /// Aliasing is the same story as <see cref="AttnFwdKvLen"/>: partials are always arena rents,
    /// k/v are always cache slices — two allocations, no overlap by construction, so NoOverlap is
    /// deliberately omitted again.
    /// </summary>
    public void AttnFwdKvLenChunked(DeviceTensor outp, DeviceTensor stats, DeviceTensor q, DeviceTensor k, DeviceTensor v,
        DeviceTensor partialOut, DeviceTensor partialStat,
        int pos0, int qLen, int kvLen, int numHeads, int numKv, int headDim, int numChunks)
    {
        int qDim = numHeads * headDim, kvDim = numKv * headDim;
        if (numChunks <= 0) throw new ArgumentException($"numChunks {numChunks} must be positive.");
        if (pos0 < 0 || qLen <= 0 || kvLen <= 0 || kvLen != pos0 + qLen) throw new ArgumentException($"pos0 {pos0}, qLen {qLen}, kvLen {kvLen}: need kvLen == pos0 + qLen > 0.");
        if (numHeads % numKv != 0) throw new ArgumentException($"numHeads {numHeads} must be a multiple of numKv {numKv}.");
        if (outp.Rows != qLen || outp.Cols != qDim) throw new ArgumentException($"outp must be [{qLen},{qDim}], got [{outp.Rows},{outp.Cols}].");
        if (q.Rows != qLen || q.Cols != qDim) throw new ArgumentException($"q must be [{qLen},{qDim}], got [{q.Rows},{q.Cols}].");
        if (k.Rows != kvLen || k.Cols != kvDim) throw new ArgumentException($"k must be [{kvLen},{kvDim}], got [{k.Rows},{k.Cols}].");
        if (v.Rows != kvLen || v.Cols != kvDim) throw new ArgumentException($"v must be [{kvLen},{kvDim}], got [{v.Rows},{v.Cols}].");
        StatsColsOrThrow(stats, qLen, numHeads);
        int partialRows = numHeads * qLen * numChunks;
        if (partialOut.Rows != partialRows || partialOut.Cols != headDim) throw new ArgumentException($"partialOut must be [{partialRows},{headDim}], got [{partialOut.Rows},{partialOut.Cols}].");
        if (partialStat.Rows != partialRows || partialStat.Cols != Kernels.FlashChunkedAttentionKernels.PartialStatCols) throw new ArgumentException($"partialStat must be [{partialRows},{Kernels.FlashChunkedAttentionKernels.PartialStatCols}], got [{partialStat.Rows},{partialStat.Cols}].");
        _flashPartialKvLen(numHeads * qLen * numChunks, partialOut.View, partialStat.View, q.View, k.View, v.View, pos0, qLen, kvLen, numHeads, numKv, headDim, numChunks, 1f / MathF.Sqrt(headDim));
        _flashMergeKvLen(numHeads * qLen, outp.View, stats.View, partialOut.View, partialStat.View, qLen, numHeads, headDim, numChunks);
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

    private static void StatsColsOrThrow(DeviceTensor stats, int qLen, int numHeads)
    {
        int rows = numHeads * qLen;
        if (stats.Rows != rows) throw new ArgumentException($"stats rows {stats.Rows} != numHeads*qLen {rows}.");
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
