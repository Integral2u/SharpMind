using SharpMind.Core.Tensors;
using System.Collections.Concurrent;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
public sealed class RoPE : PositionalEncoder
{
    private static readonly ConcurrentDictionary<int, (float[] cos, float[] sin)> TableCache = [];
    private readonly float[] _cosCache; // [MaxSeqLen, HeadDim/2]
    private readonly float[] _sinCache;
    private readonly int     _headDim;
    private readonly int     _maxSeqLen;

    // Construction

    private readonly int _ropeDim;    // dimensions that actually get rotated (null=full headDim)

    /// <param name="headDim">Dimension of a single attention head. Must be even.</param>
    /// <param name="maxSeqLen">Maximum sequence length to pre-compute freqs for.</param>
    /// <param name="theta">Base frequency. 10000 for LLaMA 2; 500000 for LLaMA 3.</param>
    /// <param name="ropeDim">Optional subset of headDim to apply RoPE to. Null means apply to all.</param>
    /// <param name="ropeScalingType">RoPE scaling type ("linear", "llama3", etc.). Null = no scaling.</param>
    /// <param name="ropeScalingFactor">RoPE scaling factor (e.g. 2.0 for 2x context).</param>
    /// <param name="ropeOriginalContextLength">Original context length before scaling (NTK-by-parts).</param>
    /// <param name="lowFreqFactor">NTK-by-parts low frequency factor (llama3).</param>
    /// <param name="highFreqFactor">NTK-by-parts high frequency factor (llama3).</param>
    public RoPE(int headDim, int maxSeqLen, float theta = 10_000f,
        int? ropeDim = null, string? ropeScalingType = null, float? ropeScalingFactor = null,
        int? ropeOriginalContextLength = null, float? lowFreqFactor = null, float? highFreqFactor = null,
        float[]? precomputedFreqs = null)
    {
        if (headDim % 2 != 0)
            throw new ArgumentException($"HeadDim must be even, got {headDim}.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        _headDim   = headDim;
        _maxSeqLen = maxSeqLen;
        _ropeDim = ropeDim ?? headDim;

        int halfDim = _ropeDim / 2;

        if (precomputedFreqs != null)
        {
            var cos = new float[_maxSeqLen * halfDim];
            var sin = new float[_maxSeqLen * halfDim];
            for (int pos = 0; pos < _maxSeqLen; pos++)
            {
                int cacheBase = pos * halfDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float angle = pos * precomputedFreqs[i];
                    cos[cacheBase + i] = MathF.Cos(angle);
                    sin[cacheBase + i] = MathF.Sin(angle);
                }
            }
            _cosCache = cos;
            _sinCache = sin;
        }
        else
        {
            (_cosCache, _sinCache) = TableCache.GetOrAdd(
                HashCode.Combine(theta, halfDim, _maxSeqLen, ropeScalingFactor, ropeScalingType,
                    ropeOriginalContextLength, lowFreqFactor, highFreqFactor), (b) =>
            {
                var cos = new float[_maxSeqLen * halfDim];
                var sin = new float[_maxSeqLen * halfDim];
                var freqs = new float[halfDim];
                for (int i = 0; i < halfDim; i++)
                    freqs[i] = 1f / MathF.Pow(theta, 2f * i / _ropeDim);

                float scaleFactor = ropeScalingFactor ?? 1f;
                bool linearScaling = string.Equals(ropeScalingType, "linear", StringComparison.OrdinalIgnoreCase);
                bool isLlama3 = string.Equals(ropeScalingType, "llama3", StringComparison.OrdinalIgnoreCase);

                if (isLlama3 && ropeOriginalContextLength.HasValue && ropeOriginalContextLength.Value > 0
                    && lowFreqFactor.HasValue && highFreqFactor.HasValue)
                {
                    float lowFreqWavelen = ropeOriginalContextLength.Value / lowFreqFactor.Value;
                    float highFreqWavelen = ropeOriginalContextLength.Value / highFreqFactor.Value;

                    for (int i = 0; i < halfDim; i++)
                    {
                        float wavelen = 2f * MathF.PI / freqs[i];
                        if (wavelen >= highFreqWavelen)
                        {
                            if (wavelen > lowFreqWavelen)
                                freqs[i] /= scaleFactor;
                            else
                            {
                                float t = (wavelen - highFreqWavelen) / (lowFreqWavelen - highFreqWavelen);
                                float smooth = 1f + t * (scaleFactor - 1f);
                                freqs[i] /= smooth;
                            }
                        }
                    }
                }

                for (int pos = 0; pos < _maxSeqLen; pos++)
                {
                    int cacheBase = pos * halfDim;
                    for (int i = 0; i < halfDim; i++)
                    {
                        float scaledPos = linearScaling ? (pos / scaleFactor) : pos;
                        float angle = scaledPos * freqs[i];
                        cos[cacheBase + i] = MathF.Cos(angle);
                        sin[cacheBase + i] = MathF.Sin(angle);
                    }
                }
                return (cos, sin);
            });
        }
    }

    // Properties

    public int HeadDim   => _headDim;
    public int MaxSeqLen => _maxSeqLen;

    // Forward (in-place)

    /// <summary>
    /// Applies RoPE rotation in-place to a query or key tensor.
    /// Input shape: [SeqLen, NumHeads, HeadDim]  — the standard QKV projection output.
    /// </summary>
    /// <param name="x">Query or key tensor. Modified in place.</param>
    /// <param name="positionOffset">
    /// Starting position index. Set to 0 for prefill; set to current KV-cache
    /// length for incremental decode.
    /// </param>
    public override void Apply(Tensor<float> x, int positionOffset = 0)
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

        int ropePairs = _ropeDim / 2;
        var data    = x.Data;

        for (int s = 0; s < seqLen; s++)
        {
            int pos       = positionOffset + s;
            int cacheBase = pos * ropePairs;

            for (int h = 0; h < numHead; h++)
            {
                int offset = (s * numHead + h) * _headDim;

                // Half-based RoPE pairing (matches HF's rotate_half convention):
                // pairs are (data[i], data[ropePairs + i]) for i = 0..ropePairs-1
                int i = 0;
                if (Avx.IsSupported)
                {
                    for (; i <= ropePairs - 8; i += 8)
                    {
                        var cos = Vector256.LoadUnsafe(ref _cosCache[cacheBase + i]);
                        var sin = Vector256.LoadUnsafe(ref _sinCache[cacheBase + i]);
                        var vx0 = Vector256.LoadUnsafe(ref data[offset + i]);
                        var vx1 = Vector256.LoadUnsafe(ref data[offset + ropePairs + i]);

                        Vector256.StoreUnsafe(
                            Avx.Subtract(Avx.Multiply(vx0, cos), Avx.Multiply(vx1, sin)),
                            ref data[offset + i]);
                        Vector256.StoreUnsafe(
                            Avx.Add(Avx.Multiply(vx1, cos), Avx.Multiply(vx0, sin)),
                            ref data[offset + ropePairs + i]);
                    }
                }
                for (; i < ropePairs; i++)
                {
                    float cos = _cosCache[cacheBase + i];
                    float sin = _sinCache[cacheBase + i];
                    float x0  = data[offset + i];
                    float x1  = data[offset + ropePairs + i];

                    data[offset + i]               = x0 * cos - x1 * sin;
                    data[offset + ropePairs + i]   = x1 * cos + x0 * sin;
                }
            }
        }
    }

    /// <summary>
    /// Applies RoPE to a batched tensor [Batch, SeqLen, NumHeads, HeadDim].
    /// </summary>
    public override void ApplyBatched(Tensor<float> x, int positionOffset = 0)
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
}
