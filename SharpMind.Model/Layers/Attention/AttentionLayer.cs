using JigSawDotNet;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Attention;

/// <summary>
/// Attention layer assembled by JigSawDotNet.
/// The "attention" mapping key (e.g. "mha_avx2", "gqa_scalar") selects
/// which scaled-dot-product kernel is wired in.
///
/// MHA, GQA, and MQA all share the same QKV projection weights and output
/// projection — only the inner kernel differs:
///   MHA  — each Q head attends to its own K/V head (NumKvHeads == NumHeads)
///   GQA  — groups of Q heads share K/V heads  (NumKvHeads &lt; NumHeads)
///   MQA  — all Q heads share a single K/V head (NumKvHeads == 1)
/// </summary>
public abstract class AttentionLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(Attention)}.{nameof(AttentionKernels)}";

    protected readonly ModelConfig Config;
    protected readonly LinearLayer Wq;   // [HiddenDim, HiddenDim]
    protected readonly LinearLayer Wk;   // [KvDim, HiddenDim]
    protected readonly LinearLayer Wv;   // [KvDim, HiddenDim]
    protected readonly LinearLayer Wo;   // [HiddenDim, HiddenDim]
    protected readonly RoPE Rope;
    private bool _disposed;

    protected AttentionLayer(ModelConfig config)
    {
        Config = config;
        int kvDim = config.NumKvHeads * config.HeadDim;

        Wq = new LinearLayer(config.HiddenDim, config.HiddenDim);
        Wk = new LinearLayer(config.HiddenDim, kvDim);
        Wv = new LinearLayer(config.HiddenDim, kvDim);
        Wo = new LinearLayer(config.HiddenDim, config.HiddenDim);
        Rope = new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta);
    }

    // ── PuzzleCornerPieces — attention variant × hw ───────────────────────

    [PuzzleCornerPiece(SharpMindConfig.KeyAttention,
        SharpMindConfig.ValMhaAvx2,
            NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMhaScalar,
            NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValGqaAvx2,
            NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValGqaScalar,
            NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValMqaAvx2,
            NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMqaScalar,
            NS + "." + nameof(AttentionKernels.ScaledDotProductScalar))]
    public abstract unsafe void ScaledDotProduct(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal);

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Full attention forward pass.
    /// Input:  [Batch, SeqLen, HiddenDim]
    /// Output: [Batch, SeqLen, HiddenDim]
    /// </summary>
    public Tensor<float> Forward(
        Tensor<float> x,
        TensorOps ops,
        int positionOffset = 0,
        bool causal = true)
    {
        ThrowIfDisposed();
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hidden = x.Shape[2];
        int numH = Config.NumHeads;
        int numKv = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int kvDim = numKv * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        // Project Q, K, V
        using var q = Wq.Forward(x, ops); // [B, S, H]
        using var k = Wk.Forward(x, ops); // [B, S, KvDim]
        using var v = Wv.Forward(x, ops);

        // Apply RoPE to Q and K — reshape to [B*S, numH, headDim] first
        using var qr = q.Reshape(batch, seqLen, numH, headDim);
        using var kr = k.Reshape(batch, seqLen, numKv, headDim);
        Rope.ApplyBatched(qr, positionOffset);
        Rope.ApplyBatched(kr, positionOffset);

        // Run attention head by head across the batch
        var output = new Tensor<float>(batch, seqLen, hidden);

        unsafe
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numH; h++)
                {
                    // KV head index — GQA: multiple Q heads share one KV head
                    int kvHead = h / Config.KvGroupSize;

                    // Pointers to Q slice [seqLen, headDim]
                    float* pQ = qr.DataPtr + (long)(b * seqLen * numH + h) * headDim;
                    float* pK = kr.DataPtr + (long)(b * seqLen * numKv + kvHead) * headDim;
                    float* pV = v.DataPtr + (long)(b * seqLen * kvDim + kvHead * headDim);
                    float* pO = output.DataPtr + (long)(b * seqLen * hidden + h * headDim);

                    // Head stride helpers — K and V are interleaved across kvLen
                    ScaledDotProduct(pQ, pK, pV, pO, seqLen, seqLen, headDim, scale, causal);
                }
            }
        }

        // Output projection
        var projected = Wo.Forward(output, ops);
        output.Dispose();
        return projected;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
        }
        _disposed = true;
    }

    ~AttentionLayer() => Dispose(false);

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}