using SharpMind.GPU;
using SharpMind.GPU.Kernels;
using Xunit;

namespace SharpMind.Tests.GPU;

/// <summary>
/// The split-K variant (<see cref="GpuKernels.AttnFwdKvLenChunked"/> = chunk partials + merge) must
/// compute what the known-correct single kernel (<see cref="GpuKernels.AttnFwdKvLen"/>) computes,
/// so it is checked oracle-style against that kernel on identical inputs — the same pattern the
/// flash suite uses, and specifically not against end-to-end generation (a wrong softmax merge can
/// still look fluent for a few tokens; the tensor comparison catches it at the source).
///
/// Cases cover the three structural conditions the merge must survive: chunks cut by the causal
/// bound mid-row (leading continued-prefill rows), chunks left empty by it (the alpha = 0 guard),
/// and the ragged kvLen tail — plus chunk counts from 1 (below MinChunksForSplit, the wrapper must
/// still be right) to 16, and GQA groups. Tolerance is the flash suite's 1e-5; the two kernels
/// rescale in different orders so they agree to float rounding, not bit for bit.
/// </summary>
[Collection("GPU")]
public sealed class FlashChunkedAttentionKernelTests
{
    const int D = AttentionKernelTests.D;

    /// <summary>(pos0, qLen, numHeads, numKv). kvLen = pos0 + qLen.</summary>
    public static TheoryData<int, int, int, int> Cases => new()
    {
        // decode, 4 chunks (final chunk partial by kvLen)
        { 499, 1, 4, 4 }, { 499, 1, 4, 2 }, { 499, 1, 4, 1 },
        // continued prefill: leading rows partially-mask chunk 0 and leave trailing chunks empty
        { 20, 120, 4, 4 }, { 20, 120, 4, 2 }, { 20, 120, 4, 1 },
        // ragged tail, 5 chunks
        { 512, 1, 4, 4 },
        // single chunk (below MinChunksForSplit — the chunked path must still be correct)
        { 63, 1, 4, 4 },
        // 16 chunks
        { 2047, 1, 4, 4 },
        // wider continued prefill spanning 3 chunks with mixed masks
        { 60, 200, 4, 4 },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Chunked_MatchesSingleKernel(int pos0, int qLen, int numHeads, int numKv)
    {
        var dev = GpuTestDevice.Device;
        int kvLen = pos0 + qLen;
        int qDim = numHeads * D, kvDim = numKv * D;
        int numChunks = (kvLen + FlashChunkedAttentionKernels.ChunkSize - 1) / FlashChunkedAttentionKernels.ChunkSize;
        string what = $"pos0={pos0} qLen={qLen} H={numHeads} KV={numKv} kvLen={kvLen} chunks={numChunks}";

        var q = GpuTestDevice.Random(qLen * qDim, 901, 1f);
        var k = GpuTestDevice.Random(kvLen * kvDim, 902, 1f);
        var v = GpuTestDevice.Random(kvLen * kvDim, 903, 1f);

        using var arena = new DeviceArena(dev, 1 << 18);
        var tq = arena.Rent(qLen, qDim); tq.Upload(q);
        var tk = arena.Rent(kvLen, kvDim); tk.Upload(k);
        var tv = arena.Rent(kvLen, kvDim); tv.Upload(v);

        var oOracle = arena.Rent(qLen, qDim);
        var sOracle = arena.Rent(qLen * numHeads, FlashAttentionKernels.StatCols);
        dev.Kernels.AttnFwdKvLen(oOracle, sOracle, tq, tk, tv, pos0, qLen, kvLen, numHeads, numKv, D);
        dev.Synchronize();

        int rows = qLen * numHeads * numChunks;
        var oChunk = arena.Rent(qLen, qDim);
        var sChunk = arena.Rent(qLen * numHeads, FlashAttentionKernels.StatCols);
        var po = arena.Rent(rows, D);
        var ps = arena.Rent(rows, FlashChunkedAttentionKernels.PartialStatCols);
        dev.Kernels.AttnFwdKvLenChunked(oChunk, sChunk, tq, tk, tv, po, ps, pos0, qLen, kvLen, numHeads, numKv, D, numChunks);
        dev.Synchronize();

        GpuTestDevice.AssertClose(oOracle.ToArray(), oChunk.ToArray(), 1e-5, $"out {what}");

        // stats cols 0/1 (globalM, l) must equal the single kernel's row statistics.
        var so = sOracle.ToArray(); var sc = sChunk.ToArray();
        var wantM = new float[qLen * numHeads]; var gotM = new float[qLen * numHeads];
        var wantL = new float[qLen * numHeads]; var gotL = new float[qLen * numHeads];
        for (int r = 0; r < qLen * numHeads; r++) { wantM[r] = so[r * 3]; gotM[r] = sc[r * 3]; wantL[r] = so[r * 3 + 1]; gotL[r] = sc[r * 3 + 1]; }
        GpuTestDevice.AssertClose(wantM, gotM, 1e-5, $"m {what}");
        GpuTestDevice.AssertClose(wantL, gotL, 1e-5, $"l {what}");
    }
}