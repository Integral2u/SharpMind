using System;
using System.Threading.Tasks;
using SharpMind.Model.Layers.Attention;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// AttentionLayer runs its head loop under Parallel.For once the work is large
/// enough. That is only sound if heads touch disjoint output memory while
/// sharing K/V reads — which is exactly the case grouped-query attention makes
/// least obvious, since many query heads alias the same K/V tensor.
///
/// These run the head loop both ways over the same buffers and require bit
/// equality, not tolerance: the two orders perform identical arithmetic per
/// head, so any difference is interference rather than rounding.
/// </summary>
public class AttentionHeadParallelismTests
{
    [Theory]
    [InlineData(14, 2, 1, 1691)]    // Qwen2-0.5B decode: GQA, 7 query heads per KV head
    [InlineData(14, 2, 128, 1691)]  // ...and one prefill chunk
    [InlineData(8, 8, 4, 96)]       // plain MHA, one KV head each
    [InlineData(6, 1, 1, 64)]       // MQA: every head shares a single KV tensor
    public unsafe void ParallelHeadLoopMatchesSerial(int numHeads, int numKvHeads, int seqLen, int kvLen)
    {
        const int headDim = 64;
        int qStride = numHeads * headDim;
        int oStride = numHeads * headDim;
        int kvGroup = numHeads / numKvHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var rng = new Random(4242);
        var q = new float[(long)seqLen * qStride];
        var k = new float[(long)numKvHeads * kvLen * headDim];
        var v = new float[(long)numKvHeads * kvLen * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);

        var serial = new float[(long)seqLen * oStride];
        var parallel = new float[(long)seqLen * oStride];

        fixed (float* pQ = q, pK = k, pV = v, pSerial = serial, pParallel = parallel)
        {
            float* fq = pQ, fk = pK, fv = pV;

            void Head(int h, float* outBase) =>
                AttentionKernels.ScaledDotProductFlashFMA(
                    fq + (long)h * headDim,
                    fk + (long)(h / kvGroup) * kvLen * headDim,
                    fv + (long)(h / kvGroup) * kvLen * headDim,
                    outBase + (long)h * headDim,
                    seqLen, kvLen, headDim, scale, causal: true, qStride, oStride, alibiSlope: 0f, windowSize: 0);

            float* so = pSerial, po = pParallel;
            for (int h = 0; h < numHeads; h++) Head(h, so);
            Parallel.For(0, numHeads, h => Head(h, po));
        }

        for (int i = 0; i < serial.Length; i++)
            Assert.True(
                BitConverter.SingleToInt32Bits(serial[i]) == BitConverter.SingleToInt32Bits(parallel[i]),
                $"index {i}: serial={serial[i]} parallel={parallel[i]}");
    }

    /// <summary>
    /// Every output slot must be written. A head loop that skipped a head, or
    /// wrote it to the wrong offset, would leave zeros behind that the equality
    /// test above would happily accept on both sides.
    /// </summary>
    [Fact]
    public unsafe void EveryHeadWritesItsOwnOutputSlice()
    {
        const int numHeads = 14, numKvHeads = 2, headDim = 64, seqLen = 3, kvLen = 40;
        int qStride = numHeads * headDim, oStride = numHeads * headDim;
        int kvGroup = numHeads / numKvHeads;

        var rng = new Random(7);
        var q = new float[(long)seqLen * qStride];
        var k = new float[(long)numKvHeads * kvLen * headDim];
        var v = new float[(long)numKvHeads * kvLen * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() + 0.5); // strictly positive

        var output = new float[(long)seqLen * oStride];
        Array.Fill(output, float.NaN);

        fixed (float* pQ = q, pK = k, pV = v, pOut = output)
        {
            float* fq = pQ, fk = pK, fv = pV, fo = pOut;
            Parallel.For(0, numHeads, h =>
                AttentionKernels.ScaledDotProductFlashFMA(
                    fq + (long)h * headDim,
                    fk + (long)(h / kvGroup) * kvLen * headDim,
                    fv + (long)(h / kvGroup) * kvLen * headDim,
                    fo + (long)h * headDim,
                    seqLen, kvLen, headDim, 1f / MathF.Sqrt(headDim), causal: true, qStride, oStride, 0f, windowSize: 0));
        }

        // V is positive and softmax weights sum to 1, so every slot must be finite
        // and positive — a NaN means nothing wrote there.
        for (int i = 0; i < output.Length; i++)
            Assert.True(output[i] > 0f, $"output[{i}] = {output[i]} — slot not written by any head");
    }
}
