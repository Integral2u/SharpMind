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
    /// <summary>
    /// Identifies one set of cos/sin tables. Keyed by the values themselves, not
    /// by HashCode.Combine of them: a hash is 32 bits and collides, and a
    /// collision here silently hands a model another configuration's rotary
    /// tables. neoxStyle is deliberately absent — it changes how the tables are
    /// applied, not their contents.
    /// </summary>
    private readonly record struct TableKey(
        float Theta, int RopeDim, int MaxSeqLen, float? ScalingFactor,
        string? ScalingType, int? OriginalContextLength, float? LowFreqFactor, float? HighFreqFactor);

    private static readonly ConcurrentDictionary<TableKey, (float[] cos, float[] sin)> TableCache = [];
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
    /// <summary>
    /// True = NeoX/"rotate_half" pairing (d, d + ropeDim/2); false = adjacent
    /// pairing (2i, 2i+1). llama.cpp picks this per architecture: LLaMA-family
    /// GGUFs are converted with Q/K permuted so adjacent pairing is correct,
    /// while Qwen/Gemma/Phi/StableLM/GPT-NeoX are not permuted and need NeoX.
    /// </summary>
    private readonly bool _neoxStyle;

    public RoPE(int headDim, int maxSeqLen, float theta = 10_000f,
        int? ropeDim = null, string? ropeScalingType = null, float? ropeScalingFactor = null,
        int? ropeOriginalContextLength = null, float? lowFreqFactor = null, float? highFreqFactor = null,
        float[]? precomputedFreqs = null, bool neoxStyle = false)
    {
        _neoxStyle = neoxStyle;
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
                new TableKey(theta, _ropeDim, _maxSeqLen, ropeScalingFactor, ropeScalingType,
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
    public ReadOnlySpan<float> CosTable => _cosCache;
    public ReadOnlySpan<float> SinTable => _sinCache;
    public int RopeDim => _ropeDim;
    public bool NeoxStyle => _neoxStyle;

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

                if (_neoxStyle)
                {
                    // NeoX / HF "rotate_half": pair dimension d with d + ropePairs.
                    // ponytail: scalar only — correctness first; vectorise like the
                    // adjacent path below if this shows up in a profile.
                    for (int d = 0; d < ropePairs; d++)
                    {
                        float cosN = _cosCache[cacheBase + d];
                        float sinN = _sinCache[cacheBase + d];
                        float a = data[offset + d];
                        float bHalf = data[offset + d + ropePairs];

                        data[offset + d] = a * cosN - bHalf * sinN;
                        data[offset + d + ropePairs] = bHalf * cosN + a * sinN;
                    }
                    continue;
                }

                // Adjacent-pairing (matches llama.cpp / ggml):
                // pairs are (data[2i], data[2i + 1]) for i = 0..ropePairs-1,
                // rotated by angle pos * theta^{-2i/_ropeDim}.
                // (Empirically verified against llama.cpp for Llama-3.2: the
                //  half-pairing convention diverges for positions >= 1.)
                int i = 0;
                if (Avx.IsSupported)
                {
                    // 4 adjacent pairs (8 floats) per outer iteration, processed
                    // as two 128-bit lanes of 2 pairs each.
                    for (; i + 4 <= ropePairs; i += 4)
                    {
                        for (int lane = 0; lane < 2; lane++)
                        {
                            int pairBase = i + 2 * lane;
                            var vec = Vector128.LoadUnsafe(ref data[offset + 2 * pairBase]);
                            var even = Sse2.Shuffle(vec, vec, 0x88);  // [x0, x2, x0, x2]
                            var odd  = Sse2.Shuffle(vec, vec, 0xDD);  // [x1, x3, x1, x3]
                            var cos  = Vector128.Create(
                                _cosCache[cacheBase + pairBase], _cosCache[cacheBase + pairBase + 1],
                                _cosCache[cacheBase + pairBase], _cosCache[cacheBase + pairBase + 1]);
                            var sin  = Vector128.Create(
                                _sinCache[cacheBase + pairBase], _sinCache[cacheBase + pairBase + 1],
                                _sinCache[cacheBase + pairBase], _sinCache[cacheBase + pairBase + 1]);

                            var outEven = Sse.Subtract(Sse.Multiply(even, cos), Sse.Multiply(odd, sin));
                            var outOdd  = Sse.Add(Sse.Multiply(odd, cos), Sse.Multiply(even, sin));
                            // Interleave [p0_even, p0_odd, p1_even, p1_odd] back into dim order
                            Sse2.UnpackLow(outEven, outOdd).StoreUnsafe(ref data[offset + 2 * pairBase]);
                        }
                    }
                }
                for (; i < ropePairs; i++)
                {
                    float cos = _cosCache[cacheBase + i];
                    float sin = _sinCache[cacheBase + i];
                    float x0  = data[offset + 2 * i];
                    float x1  = data[offset + 2 * i + 1];

                    data[offset + 2 * i]         = x0 * cos - x1 * sin;
                    data[offset + 2 * i + 1]     = x1 * cos + x0 * sin;
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

    /// <summary>
    /// Backward of <see cref="Apply"/> for batched tensors. Mutates
    /// <paramref name="dx"/> in place, rotating each pair's gradient back by the
    /// inverse of the forward rotation:
    ///   dIn0 = cos·dOut0 + sin·dOut1
    ///   dIn1 = −sin·dOut0 + cos·dOut1
    /// </summary>
    public override void ApplyBatchedBackward(Tensor<float> dx, int positionOffset = 0)
    {
        if (dx.Rank != 4)
            throw new ArgumentException(
                $"ApplyBatchedBackward expects rank-4 [Batch, SeqLen, NumHeads, HeadDim], got rank {dx.Rank}.");

        int batch = dx.Shape[0];

        for (int b = 0; b < batch; b++)
        {
            using var slice = dx.Slice(b);
            ApplyBackward(slice, positionOffset);
        }
    }

    /// <summary>Backward of <see cref="Apply"/> for a rank-3 [SeqLen, NumHeads, HeadDim] tensor.</summary>
    private void ApplyBackward(Tensor<float> dx, int positionOffset)
    {
        int seqLen  = dx.Shape[0];
        int numHead = dx.Shape[1];
        //int dim     = dx.Shape[2];

        int ropePairs = _ropeDim / 2;
        var data      = dx.Data;

        for (int s = 0; s < seqLen; s++)
        {
            int pos       = positionOffset + s;
            int cacheBase = pos * ropePairs;

            for (int h = 0; h < numHead; h++)
            {
                int offset = (s * numHead + h) * _headDim;

                if (_neoxStyle)
                {
                    for (int d = 0; d < ropePairs; d++)
                    {
                        float cosN = _cosCache[cacheBase + d];
                        float sinN = _sinCache[cacheBase + d];
                        float g0 = data[offset + d];
                        float g1 = data[offset + d + ropePairs];

                        data[offset + d] = cosN * g0 + sinN * g1;
                        data[offset + d + ropePairs] = -sinN * g0 + cosN * g1;
                    }
                    continue;
                }

                for (int i = 0; i < ropePairs; i++)
                {
                    float cos = _cosCache[cacheBase + i];
                    float sin = _sinCache[cacheBase + i];
                    float d0  = data[offset + 2 * i];
                    float d1  = data[offset + 2 * i + 1];

                    data[offset + 2 * i]     =  cos * d0 + sin * d1;
                    data[offset + 2 * i + 1] = -sin * d0 + cos * d1;
                }
            }
        }
    }
}
