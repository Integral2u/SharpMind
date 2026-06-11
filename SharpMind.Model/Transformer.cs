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
    private readonly TransformerWeights _weights;
    private readonly EmbeddingTable _embedding;
    private readonly IArchitecture _arch;
    private readonly NormLayer _finalNorm;
    private readonly TensorOps _ops;
    private bool _disposed;

    // Separate LM head for non-weight-tied models (e.g. LLaMA 2/3).
    // Null means the model is weight-tied — the embedding weight is used instead.
    private readonly Tensor<float>? _lmHead;

    private readonly TransformerBlock[]? _blocks; // For training backward

    private Tensor<float>? _cachedEmbedding;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed;

    public Transformer(
        TransformerWeights weights,
        EmbeddingTable embedding,
        IArchitecture arch,
        NormLayer finalNorm,
        TensorOps ops,
        Tensor<float>? lmHead = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(embedding);
        ArgumentNullException.ThrowIfNull(arch);
        ArgumentNullException.ThrowIfNull(finalNorm);
        ArgumentNullException.ThrowIfNull(ops);
        
        _weights = weights;
        _embedding = embedding;
        _arch = arch;
        _finalNorm = finalNorm;
        _ops = ops;
        _lmHead = lmHead;


        if (arch is DecoderArch decodeArch)
            _blocks = decodeArch.Blocks;
    }

    public ModelConfig Config => _weights.Config;

    // ── Diagnostics accessors ──────────────────────────────────────────────
    public NormLayer FinalNorm => _finalNorm;
    public Tensor<float>? LmHead => _lmHead;
    public Tensor<float> EmbeddingWeight => _embedding.Weight;
    public Tensor<float> ForwardEmbedding(Tensor<int> tokenIds) => _embedding.Forward(tokenIds);
    public TransformerBlock? GetBlock(int layer) => _blocks is not null && layer < _blocks.Length ? _blocks[layer] : null;
    public TensorOps Ops => _ops;

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        var (_ , block, rawField) = _weights.ResolveTarget(name);
        if (block != null && rawField != null)
        {
            TransformerWeights.SetRawField(block, rawField, rawData, dtype);
            return true;
        }
        return false;
    }

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        var target = _weights.ResolveFloatTarget(name);
        if (target != null)
        {
            if (data.Length != target.ElementCount) return false;
            data.CopyTo(target.Data);
            return true;
        }
        return false;
    }

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
    public unsafe Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0, SharpMind.Core.Memory.Workspace? workspace = null) => Forward(tokenIds, null, positionOffset, workspace);

    public unsafe Tensor<float> Forward(Tensor<int> tokenIds, IKVCache[]? caches, int positionOffset = 0, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);
        using var embedded = _cachedEmbedding;

        // 2. Architecture (stack of transformer blocks).
        //    Returns embedded (in-place residuals), so no separate using — embedded owns the buffer.
        _cachedHidden = _arch.Forward(embedded, caches ?? [], positionOffset, workspace);

        // 3. Final normalisation
        _cachedNormed = _finalNorm.Forward(_cachedHidden, workspace);
        using var normed = _cachedNormed;

        // 4. LM head: [Batch, SeqLen, HiddenDim] @ LmHead^T → [Batch, SeqLen, VocabSize]
        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hidden2 = _weights.Config.HiddenDim;

        using var normedFlat = normed.Reshape(batch * seqLen, hidden2);
        var projectionWeight = _lmHead ?? _embedding.Weight;
        
        Tensor<float> logits;
        if (workspace != null)
        {
            logits = workspace.Rent<float>([batch * seqLen, _weights.Config.VocabSize]);
            _ops.MatMulWithBTInto(normedFlat, projectionWeight, logits);
        }
        else
        {
            logits = _ops.MatMulWithBT(normedFlat, projectionWeight);
        }

        // Restore [Batch, SeqLen, VocabSize]
        var result = logits.Reshape(batch, seqLen, _weights.Config.VocabSize);
        logits.Dispose();
        return result;
    }


    /// <summary>
    /// Inference fast path that returns logits only for the final token in each batch row.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, VocabSize]
    /// </summary>
    public unsafe Tensor<float> ForwardLastLogits(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset = 0, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);
        using var embedded = _cachedEmbedding;

        _cachedHidden = _arch.Forward(embedded, caches, positionOffset, workspace);


        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hiddenDim = _weights.Config.HiddenDim;
        var projectionWeight = _lmHead ?? _embedding.Weight;

        // Single-token decode: normalise in-place, no allocation, no copy
        if (batch == 1 && seqLen == 1)
        {
            _finalNorm.ForwardInPlace(_cachedHidden);
            using var flatHidden = _cachedHidden.Reshape(batch, hiddenDim);
            
        if (workspace != null)
        {
            var resultLogits = workspace.Rent<float>([batch, _weights.Config.VocabSize]);
            _ops.MatMulWithBTInto(flatHidden, projectionWeight, resultLogits);
            return resultLogits;
        }
            return _ops.MatMulWithBT(flatHidden, projectionWeight);
        }

        // Prefill path: extract last token's hidden state, norm in-place
        Tensor<float> lastHidden = workspace != null 
            ? workspace.Rent<float>([batch, hiddenDim]) 
            : new Tensor<float>(batch, hiddenDim);
        for (int b = 0; b < batch; b++)
        {
            int srcOffset = (b * seqLen + (seqLen - 1)) * hiddenDim;
            _cachedHidden.Data.Slice(srcOffset, hiddenDim).CopyTo(lastHidden.Data.Slice(b * hiddenDim, hiddenDim));
        }
        _finalNorm.ForwardInPlace(lastHidden);
        
        Tensor<float> finalLogits;
        if (workspace != null)
        {
            finalLogits = workspace.Rent<float>([batch, _weights.Config.VocabSize]);
            _ops.MatMulWithBTInto(lastHidden, projectionWeight, finalLogits);
        }
        else
        {
            finalLogits = _ops.MatMulWithBT(lastHidden, projectionWeight);
        }
        lastHidden.Dispose();
        return finalLogits;
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
            long embed = (long)_weights.Config.VocabSize * _weights.Config.HiddenDim;
            long perBlock = ParametersPerBlock();
            long norm = _weights.Config.HiddenDim;
            return embed + perBlock * _weights.Config.NumLayers + norm;
        }
    }

    private long ParametersPerBlock()
    {
        long h = _weights.Config.HiddenDim;
        long kv = (long)_weights.Config.NumKvHeads * _weights.Config.HeadDim;
        // Q + K + V + O projections
        long attn = h * h + h * kv + h * kv + h * h;
        // FFN (gated uses 3 matrices, dense uses 2)
        long ffn = _weights.Config.FfnDim * h * 3L; // conservative: gated
        // 2 norms
        long norms = h * 2;
        return attn + ffn + norms;
    }

    public override string ToString() =>
        $"Transformer ({_weights.Config.NumLayers}L × {_weights.Config.HiddenDim}D, " +
        $"~{ParameterCount / 1_000_000}M params)";


    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _embedding.Dispose();
        _arch.Dispose();
        _finalNorm.Dispose();
        _lmHead?.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(Transformer));

    // ── Training Support ───────────────────────────────────────────────────

    public (Tensor<float> Logits, TransformerState State) ForwardWithState(Tensor<int> tokenIds, int positionOffset = 0)
    {
        ThrowIfDisposed();
        DisposeCache();
        
        _cachedEmbedding = _embedding.Forward(tokenIds);
        using var embedded = _cachedEmbedding;
        
        _cachedHidden = _arch.Forward(embedded, positionOffset);
        using var hidden = _cachedHidden;
        
        _cachedNormed = _finalNorm.Forward(hidden);
        using var normed = _cachedNormed;
        
        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        
        using var normedFlat = normed.Reshape(batch * seqLen, _weights.Config.HiddenDim);
        var projectionWeight = _lmHead ?? _embedding.Weight;
        var logits = _ops.MatMulWithBT(normedFlat, projectionWeight);
        
        var result = logits.Reshape(batch, seqLen, _weights.Config.VocabSize);
        logits.Dispose();
        
        var state = new TransformerState
        {
            TokenIds = tokenIds,
            Embedded = embedded,
            Hidden = hidden,
            Normed = normed,
            Logits = result,
            PositionOffset = positionOffset
        };
        
        return (result, state);
    }

    public Tensor<float> Backward(Tensor<float> gradLogits, TransformerState state)
    {
        int batch = state.TokenIds.Shape.Rows;
        int seqLen = state.TokenIds.Shape.Cols;
        int hidden = _weights.Config.HiddenDim;
        
        // gradLogitsFlat [M, Vocab] @ W [Vocab, Hidden] → gradNormedFlat [M, Hidden]
        using var gradNormedFlat = _ops.MatMul(gradLogits.Reshape(batch * seqLen, _weights.Config.VocabSize), _embedding.Weight);
        
        using var gradHidden = _finalNorm.Backward(gradNormedFlat.Reshape(batch, seqLen, hidden), new NormLayerState(1, hidden));


        // Backward through architecture - simplified pass-through
        var gradInput = new Tensor<float>(batch, seqLen, hidden);

        return gradInput;
    }
}

public class TransformerState
{
    public Tensor<int> TokenIds { get; init; } = null!;
    public Tensor<float> Embedded { get; init; } = null!;
    public Tensor<float> Hidden { get; init; } = null!;
    public Tensor<float> Normed { get; init; } = null!;
    public Tensor<float> Logits { get; init; } = null!;
    public int PositionOffset { get; init; }
}