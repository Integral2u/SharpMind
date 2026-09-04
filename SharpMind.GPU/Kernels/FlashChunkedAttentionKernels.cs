using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// Split-K variant of <see cref="FlashAttentionKernels.FwdKvLen"/> for inference. Decode's one
/// query row per head is the whole launch — <c>numHeads</c> threads, each looping the entire cache
/// serially — so per-token latency grows linearly with conversation length with no parallelism to
/// offset it. Here <c>[0, kvLen)</c> is cut into <see cref="ChunkSize"/>-wide chunks and
/// <see cref="PartialKvLen"/> gives each (h, i, chunk) its own thread; then
/// <see cref="MergeKvLen"/> reduces the chunk partials per (h, i) with one thread, the same launch
/// size as the source kernel.
///
/// The merge is what makes the split correct. Each chunk's partial is normalised to its own local
/// max <c>m_c</c> (its <c>l_c</c> is Σ exp(s − m_c) over the chunk), so summing partials directly
/// across chunks is numerically wrong — the earlier failed attempt did exactly that — and sharing
/// the destination accumulator across threads is a race. Writing chunks to a chunk-private slot
/// removes the race by construction; rescaling each partial by <c>exp(m_c − globalM)</c> before
/// summing makes the merge the same rescale-on-new-max identity <see cref="FlashAttentionKernels.FwdKvLen"/>
/// uses across individual keys, just applied across chunks instead. An empty chunk (wholly beyond a
/// row's causal bound) keeps <c>m = −inf, l = 0</c> and contributes <c>alpha = 0</c>.
///
/// Layouts are chunk-slot-major so adjacent threads never share a cell; inter-chunk coalescing is
/// strided and secondary to the disjoint-write guarantee. <c>scale</c> is folded into kernel 1 the
/// way the parent kernel folds it in, so the merge operates on already-scaled scores.
/// <see cref="PartialStatCols"/> is 2 (m, l): the training backward's third column is not used here.
/// </summary>
internal static class FlashChunkedAttentionKernels
{
    /// <summary>
    /// Chunk width: wide enough that the merge launch is cheap, narrow enough that decode gets real
    /// parallelism. Tunable; the engine routes below <see cref="MinChunksForSplit"/> to the single
    /// kernel, so this value only biases the split/merge balance once splitting is active.
    /// </summary>
    public const int ChunkSize = 128;

    /// <summary>
    /// Below this many chunks the merge kernel's extra launch is not worth its occupancy gain and
    /// the engine calls <see cref="FlashAttentionKernels.FwdKvLen"/> instead.
    /// </summary>
    public const int MinChunksForSplit = 4;

    /// <summary>Per-chunk statistics: local column max <c>m</c> and sum-of-exp <c>l</c>.</summary>
    public const int PartialStatCols = 2;

    /// <summary>
    /// One thread per (h, i, chunk), idx = (h·qLen + i)·numChunks + chunk. Runs exactly the
    /// <see cref="FlashAttentionKernels.FwdKvLen"/> inner loop over [start, end) ∩ [0, pos0 + i],
    /// where start/end come from <paramref name="chunk"/> and <see cref="ChunkSize"/>. Writes this
    /// chunk's unnormalised Σ p·V into a chunk-private slot of <paramref name="partialOut"/> and
    /// the chunk's local (m, l) into <paramref name="partialStat"/>; no thread writes memory another
    /// thread writes, so there is no race. A chunk wholly beyond the row's causal bound loops zero
    /// times and leaves m = −inf, l = 0.
    /// </summary>
    public static void PartialKvLen(Index1D idx, ArrayView<float> partialOut, ArrayView<float> partialStat,
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        int pos0, int qLen, int kvLen, int numHeads, int numKv, int headDim, int numChunks, float scale)
    {
        int row = idx / numChunks;
        int chunk = idx - row * numChunks;
        int i = row % qLen;
        int h = row / qLen;
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        int absPos = pos0 + i;
        long qOff = (long)i * qDim + (long)h * headDim;

        long pOut = ((long)row * numChunks + chunk) * headDim;
        for (int d = 0; d < headDim; d++) partialOut[pOut + d] = 0f;

        int start = chunk * ChunkSize;
        int end = start + ChunkSize < kvLen ? start + ChunkSize : kvLen;
        float m = float.NegativeInfinity, l = 0f;
        for (int j = start; j < end && j <= absPos; j++)
        {
            long kvOff = (long)j * kvDim + (long)kvh * headDim;
            float s = 0f;
            for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kvOff + d];
            s *= scale;
            if (s > m)
            {
                // Same -inf guard as FlashAttentionKernels.FwdKvLen: on a chunk's first key nothing
                // is accumulated yet, so the rescale is a no-op and exp(-inf - s) is not 0.
                float alpha = m == float.NegativeInfinity ? 0f : XMath.Exp(m - s);
                l *= alpha;
                for (int d = 0; d < headDim; d++) partialOut[pOut + d] *= alpha;
                m = s;
            }
            float p = XMath.Exp(s - m);
            l += p;
            for (int d = 0; d < headDim; d++) partialOut[pOut + d] += p * v[kvOff + d];
        }
        long pSt = (((long)row * numChunks) + chunk) * PartialStatCols;
        partialStat[pSt] = m; partialStat[pSt + 1] = l;
    }

    /// <summary>
    /// One thread per (h, i), idx = h·qLen + i — the same launch size as
    /// <see cref="FlashAttentionKernels.FwdKvLen"/>. Reduces the <paramref name="numChunks"/> chunk
    /// partials into <paramref name="outp"/> and writes the row's (globalM, l) to
    /// <paramref name="stats"/> cols 0/1 so the statistics layout matches the single kernel. The
    /// accumulator lives in the destination row, owned exclusively by this thread, as the parent
    /// kernels do — ILGPU kernels cannot size a local array from a runtime headDim.
    /// </summary>
    public static void MergeKvLen(Index1D idx, ArrayView<float> outp, ArrayView<float> stats,
        ArrayView<float> partialOut, ArrayView<float> partialStat,
        int qLen, int numHeads, int headDim, int numChunks)
    {
        int i = idx % qLen;
        int h = idx / qLen;
        int qDim = numHeads * headDim;
        long qOff = (long)i * qDim + (long)h * headDim;
        long rowBase = (long)h * qLen + i;
        long pOutBase = (rowBase * numChunks) * headDim;
        long pStBase = (rowBase * numChunks) * PartialStatCols;

        for (int d = 0; d < headDim; d++) outp[qOff + d] = 0f;

        float globalM = float.NegativeInfinity;
        for (int c = 0; c < numChunks; c++)
        {
            float m = partialStat[pStBase + (long)c * PartialStatCols];
            if (m > globalM) globalM = m;
        }

        float l = 0f;
        for (int c = 0; c < numChunks; c++)
        {
            long pc = pStBase + (long)c * PartialStatCols;
            float m = partialStat[pc];
            if (m == float.NegativeInfinity) continue;   // empty chunk: alpha would be 0 anyway
            float alpha = XMath.Exp(m - globalM);
            l += partialStat[pc + 1] * alpha;
            long co = pOutBase + (long)c * headDim;
            for (int d = 0; d < headDim; d++) outp[qOff + d] += partialOut[co + d] * alpha;
        }
        float inv = 1f / l;
        for (int d = 0; d < headDim; d++) outp[qOff + d] *= inv;

        long st = rowBase * FlashAttentionKernels.StatCols;
        stats[st] = globalM; stats[st + 1] = l;
    }
}