using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Model;

public sealed class Transformer : IDisposable
{
    private readonly ModelConfig _config;
    private readonly EmbeddingTable _embedding;
    private readonly IArchitecture _arch;
    private readonly NormLayer _finalNorm;
    private readonly TensorOps _ops;
    private bool _disposed;

    private Tensor<float>? _cachedEmbedding;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed;

    public Transformer(
        ModelConfig config,
        EmbeddingTable embedding,
        IArchitecture arch,
        NormLayer finalNorm,
        TensorOps ops)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(embedding);
        ArgumentNullException.ThrowIfNull(arch);
        ArgumentNullException.ThrowIfNull(finalNorm);
        ArgumentNullException.ThrowIfNull(ops);

        _config = config;
        _embedding = embedding;
        _arch = arch;
        _finalNorm = finalNorm;
        _ops = ops;
    }

    public ModelConfig Config => _config;

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in _embedding.Parameters())
            yield return p;

        foreach (var p in _arch.Parameters())
            yield return p;

        foreach (var p in _finalNorm.Parameters())
            yield return p;
    }

    // ── Forward ───────────────────────────────────────────────────────────

    /// <summary>
    /// Full forward pass.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, SeqLen, VocabSize]
    /// </summary>
    public unsafe Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0)
    {
        ThrowIfDisposed();

        DisposeCache();

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds);
        using var embedded = _cachedEmbedding;

        // 2. Architecture (stack of transformer blocks)
        _cachedHidden = _arch.Forward(embedded, positionOffset);
        using var hidden = _cachedHidden;

        // 3. Final normalisation
        _cachedNormed = _finalNorm.Forward(hidden);
        using var normed = _cachedNormed;

        // 4. LM head: [Batch, SeqLen, HiddenDim] @ EmbeddingWeight^T
        //    → [Batch, SeqLen, VocabSize]
        //    Weight tying: reuse embedding table rows as the projection matrix.
        //    EmbeddingWeight is [VocabSize, HiddenDim]; we need [HiddenDim, VocabSize]
        //    TensorOps.MatMul transposes B internally so this is correct.
        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hidden2 = _config.HiddenDim;

        using var normedFlat = normed.Reshape(batch * seqLen, hidden2);
        using var embedT = TensorOps.Transpose(_embedding.Weight);
        var logits = _ops.MatMul(normedFlat, embedT);

        // Restore [Batch, SeqLen, VocabSize]
        var result = logits.Reshape(batch, seqLen, _config.VocabSize);
        logits.Dispose();
        return result;
    }

    private void DisposeCache()
    {
        _cachedEmbedding?.Dispose();
        _cachedHidden?.Dispose();
        _cachedNormed?.Dispose();
        _cachedEmbedding = null;
        _cachedHidden = null;
        _cachedNormed = null;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────

    /// <summary>Approximate total parameter count.</summary>
    public long ParameterCount
    {
        get
        {
            long embed = (long)_config.VocabSize * _config.HiddenDim;
            long perBlock = ParametersPerBlock();
            long norm = _config.HiddenDim;
            return embed + perBlock * _config.NumLayers + norm;
        }
    }

    private long ParametersPerBlock()
    {
        long h = _config.HiddenDim;
        long kv = (long)_config.NumKvHeads * _config.HeadDim;
        // Q + K + V + O projections
        long attn = h * h + h * kv + h * kv + h * h;
        // FFN (gated uses 3 matrices, dense uses 2)
        long ffn = _config.FfnDim * h * 3L; // conservative: gated
        // 2 norms
        long norms = h * 2;
        return attn + ffn + norms;
    }

    public override string ToString() =>
        $"Transformer ({_config.NumLayers}L × {_config.HiddenDim}D, " +
        $"~{ParameterCount / 1_000_000}M params)";

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _embedding.Dispose();
        _arch.Dispose();
        _finalNorm.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(Transformer));
}