using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

/// <summary>
/// Rotary Position Encoding (RoPE) — Su et al., 2021.
/// Used by LLaMA 2/3, Mistral, Falcon, and most modern open-weight LLMs.
///
/// RoPE encodes position by rotating query and key vectors in pairs.
/// Unlike additive sinusoidal embeddings, RoPE is applied inside each
/// attention head rather than to the input sequence.
///
/// Key property: the dot product q·k depends only on the relative position
/// (m - n), making RoPE naturally length-generalising.
///
/// Usage:
///   var rope = new RoPE(headDim: 128, maxSeqLen: 4096, theta: 10000f);
///   rope.Apply(query, positionOffset: 0);  // mutates query in-place
///   rope.Apply(key,   positionOffset: 0);  // mutates key in-place
/// </summary>
public sealed class RoPE
{
    private readonly float[] _cosCache; // [MaxSeqLen, HeadDim/2]
    private readonly float[] _sinCache;
    private readonly int     _headDim;
    private readonly int     _maxSeqLen;

    // ── Construction ──────────────────────────────────────────────────────

    /// <param name="headDim">Dimension of a single attention head. Must be even.</param>
    /// <param name="maxSeqLen">Maximum sequence length to pre-compute freqs for.</param>
    /// <param name="theta">Base frequency. 10000 for LLaMA 2; 500000 for LLaMA 3.</param>
    public RoPE(int headDim, int maxSeqLen, float theta = 10_000f)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"HeadDim must be even, got {headDim}.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        _headDim   = headDim;
        _maxSeqLen = maxSeqLen;

        int halfDim = headDim / 2;
        _cosCache = new float[maxSeqLen * halfDim];
        _sinCache = new float[maxSeqLen * halfDim];

        PrecomputeFreqs(theta, halfDim);
    }

    // ── Properties ────────────────────────────────────────────────────────

    public int HeadDim   => _headDim;
    public int MaxSeqLen => _maxSeqLen;

    // ── Forward (in-place) ────────────────────────────────────────────────

    /// <summary>
    /// Applies RoPE rotation in-place to a query or key tensor.
    /// Input shape: [SeqLen, NumHeads, HeadDim]  — the standard QKV projection output.
    /// </summary>
    /// <param name="x">Query or key tensor. Modified in place.</param>
    /// <param name="positionOffset">
    /// Starting position index. Set to 0 for prefill; set to current KV-cache
    /// length for incremental decode.
    /// </param>
    public void Apply(Tensor<float> x, int positionOffset = 0)
    {
        if (x.Rank != 3)
            throw new ArgumentException(
                $"RoPE.Apply expects rank-3 [SeqLen, NumHeads, HeadDim], got rank {x.Rank}.");

        int seqLen  = x.Shape[0];
        int numHead = x.Shape[1];
        int dim     = x.Shape[2];

        if (dim != _headDim)
            throw new ArgumentException(
                $"RoPE HeadDim mismatch: expected {_headDim}, got {dim}.");
        if (positionOffset + seqLen > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positionOffset),
                $"Position {positionOffset + seqLen} exceeds MaxSeqLen {_maxSeqLen}.");

        int halfDim = _headDim / 2;
        var data    = x.Data;

        for (int s = 0; s < seqLen; s++)
        {
            int pos       = positionOffset + s;
            int cacheBase = pos * halfDim;

            for (int h = 0; h < numHead; h++)
            {
                int offset = (s * numHead + h) * _headDim;

                int i = 0;
                if (Avx.IsSupported)
                {
                    for (; i <= halfDim - 4; i += 4)
                    {
                        var cos = Vector256.LoadUnsafe(ref _cosCache[cacheBase + i]);
                        var sin = Vector256.LoadUnsafe(ref _sinCache[cacheBase + i]);
                        var vx0 = Vector256.LoadUnsafe(ref data[offset + i]);
                        var vx1 = Vector256.LoadUnsafe(ref data[offset + halfDim + i]);

                        Vector256.StoreUnsafe(
                            Avx.Subtract(Avx.Multiply(vx0, cos), Avx.Multiply(vx1, sin)),
                            ref data[offset + i]);
                        Vector256.StoreUnsafe(
                            Avx.Add(Avx.Multiply(vx1, cos), Avx.Multiply(vx0, sin)),
                            ref data[offset + halfDim + i]);
                    }
                }
                for (; i < halfDim; i++)  //finishes off or does all if avx missing
                {
                    float cos = _cosCache[cacheBase + i];
                    float sin = _sinCache[cacheBase + i];
                    float x0  = data[offset + i];
                    float x1  = data[offset + halfDim + i];

                    data[offset + i]           = x0 * cos - x1 * sin;
                    data[offset + halfDim + i] = x1 * cos + x0 * sin;
                }
            }
        }
    }

    /// <summary>
    /// Applies RoPE to a batched tensor [Batch, SeqLen, NumHeads, HeadDim].
    /// </summary>
    public void ApplyBatched(Tensor<float> x, int positionOffset = 0)
    {
        if (x.Rank != 4)
            throw new ArgumentException(
                $"ApplyBatched expects rank-4 [Batch, SeqLen, NumHeads, HeadDim], got rank {x.Rank}.");

        int batch = x.Shape[0];

        for (int b = 0; b < batch; b++)
        {
            using var slice = x.Slice(b);
            Apply(slice, positionOffset);
        }
    }

    // ── Cache building ────────────────────────────────────────────────────

    private void PrecomputeFreqs(float theta, int halfDim)
    {
        // Frequency for each pair: θ_i = 1 / (theta ^ (2i / headDim))
        var freqs = new float[halfDim];
        for (int i = 0; i < halfDim; i++)
            freqs[i] = 1f / MathF.Pow(theta, 2f * i / _headDim);

        for (int pos = 0; pos < _maxSeqLen; pos++)
        {
            int cacheBase = pos * halfDim;
            for (int i = 0; i < halfDim; i++)
            {
                float angle = pos * freqs[i];
                _cosCache[cacheBase + i] = MathF.Cos(angle);
                _sinCache[cacheBase + i] = MathF.Sin(angle);
            }
        }
    }
}
