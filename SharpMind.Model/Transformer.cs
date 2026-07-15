using SharpMind.Core.Embeddings;
using SharpMind.Core.Quantization;
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
    private bool _disposed;

    // Separate LM head for non-weight-tied models (e.g. LLaMA 2/3).
    // Null means the model is weight-tied — the embedding weight is used instead.
    private readonly Tensor<float>? _lmHead;
    private readonly QuantizationOps? _qOps;
    private readonly LogitOps _logitOps;

    private readonly TransformerBlock[]? _blocks; // For training backward

    private Tensor<float>? _cachedEmbedding;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed;

    public Transformer(
        TransformerWeights weights,
        EmbeddingTable embedding,
        IArchitecture arch,
        NormLayer finalNorm,
        Tensor<float>? lmHead = null,
        QuantizationOps? qOps = null,
        Dictionary<string, string>? mapping = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(embedding);
        ArgumentNullException.ThrowIfNull(arch);
        ArgumentNullException.ThrowIfNull(finalNorm);
        
        _weights = weights;
        _embedding = embedding;
        _arch = arch;
        _finalNorm = finalNorm;
        _lmHead = lmHead;
        _qOps = qOps;
        var projWeight = _lmHead ?? _embedding.Weight;
        var rawW = _lmHead != null ? _weights.RawLmHead : _weights.RawEmbedding;
        var rawDtype = _lmHead != null ? _weights.RawLmHeadDtype : _weights.RawEmbeddingDtype;
        _logitOps = LogitOpsFactory.Create(projWeight, rawW, rawDtype, mapping);

        if (arch is DecoderArch decodeArch)
            _blocks = decodeArch.Blocks;
    }

    public ModelConfig Config => _weights.Config;

    // Diagnostics accessors
    public NormLayer FinalNorm => _finalNorm;
    public Tensor<float>? LmHead => _lmHead;
    public Tensor<float> EmbeddingWeight => _embedding.Weight;
    public Tensor<float> ForwardEmbedding(Tensor<int> tokenIds) => _embedding.Forward(tokenIds);
    public TransformerBlock? GetBlock(int layer) => _blocks is not null && layer < _blocks.Length ? _blocks[layer] : null;

    /// <summary>
    /// Frees float weight memory for all layers using quantized forward.
    /// Safe to call after model loading is complete — float weights are not
    /// needed for inference when UseQuantizedForward is active.
    /// </summary>
    public void FreeFloatWeights()
    {
        if (_blocks != null)
        {
            foreach (var block in _blocks)
                block.FreeFloatWeights();
        }
    }

    /// <summary>Sets an activation hook on all blocks in the model.</summary>
    public void SetActivationHook(IActivationHook? hook)
    {
        if (_blocks is null) return;
        foreach (var block in _blocks)
            block.SetActivationHook(hook);
    }

    public byte[]? RawEmbedding => _weights.RawEmbedding;
    public QuantDType? RawEmbeddingDtype => _weights.RawEmbeddingDtype;
    public QuantizationOps? QOps => _qOps;

    /// <summary>
    /// Exposes the cached hidden state from the last Forward/ForwardLastLogits call.
    /// Contains the arch output (pre-final-norm) for all processed positions.
    /// Shape: [Batch, SeqLen, HiddenDim]. May be modified by ForwardInPlace when
    /// the single-token path of ForwardLastLogits is used.
    /// </summary>
    public Tensor<float>? LastCachedHidden => _cachedHidden;

    /// <summary>
    /// Copies row <paramref name="positionIndex"/> from the last cached hidden state,
    /// applies final norm, and returns a new [1, HiddenDim] tensor.
    /// Returns null if no cached hidden state is available.
    /// </summary>
    public Tensor<float>? GetNormedHiddenRow(int positionIndex, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        if (_cachedHidden == null) return null;
        int hiddenDim = _weights.Config.HiddenDim;
        Tensor<float> row;
        if (workspace != null)
        {
            row = workspace.Rent<float>([1, hiddenDim]);
        }
        else
        {
            row = new Tensor<float>(1, hiddenDim);
        }
        _cachedHidden.Data.Slice(positionIndex * hiddenDim, hiddenDim).CopyTo(row.Data);
        _finalNorm.ForwardInPlace(row);
        return row;
    }

    public bool SetRawWeight(string name, byte[] rawData, QuantDType dtype)
    {
        var (target, block, rawField) = _weights.ResolveTarget(name);
        if (block != null && rawField != null)
        {
            TransformerWeights.SetRawField(block, rawField, rawData, dtype);
            return true;
        }
        if (target != null && block == null && rawData.Length > 0)
        {
            _weights.RawEmbedding = rawData;
            _weights.RawEmbeddingDtype = dtype;
            return true;
        }
        return false;
    }

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        var target = _weights.ResolveFloatTarget(name);
        if (target == null) return false;        
        if (data.Length != target.ElementCount) return false;
        data.CopyTo(target.Data);
        return true;
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

    /// <summary>
    /// Full forward pass.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, SeqLen, VocabSize]
    /// </summary>
    public Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0, Core.Memory.Workspace? workspace = null) => Forward(tokenIds, null, positionOffset, workspace);

    public Tensor<float> Forward(Tensor<int> tokenIds, IKVCache[]? caches, int positionOffset = 0, Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);

        // 2. Architecture (stack of transformer blocks).
        //    Blocks mutate in-place and return the same tensor, so _cachedHidden
        //    may alias _cachedEmbedding. Keep both alive until we exit this method.
        _cachedHidden = _arch.Forward(_cachedEmbedding, caches ?? new IKVCache[_arch.NumLayers], positionOffset, workspace);

        // 3. Final normalisation
        _cachedNormed?.Dispose();
        _cachedNormed = _finalNorm.Forward(_cachedHidden, workspace);

        // 4. LM head: [Batch, SeqLen, HiddenDim] @ LmHead^T → [Batch, SeqLen, VocabSize]
        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hiddenDim = _weights.Config.HiddenDim;

        using var normedFlat = _cachedNormed.Reshape(batch * seqLen, hiddenDim);
        int M = batch * seqLen;
        int K = hiddenDim;
        int N = _weights.Config.VocabSize;

        var logits = _logitOps.Project(normedFlat, M, K, N, workspace);
        var result = logits.Reshape(batch, seqLen, N);
        logits.Dispose();
        return result;
    }


    /// <summary>
    /// Inference fast path that returns logits only for the final token in each batch row.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, VocabSize]
    /// </summary>
    public Tensor<float> ForwardLastLogits(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset = 0, Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);

        _cachedHidden = _arch.Forward(_cachedEmbedding, caches, positionOffset, workspace);


        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hiddenDim = _weights.Config.HiddenDim;
        int K = hiddenDim;
        int N = _weights.Config.VocabSize;

        // Single-token decode: normalise in-place, no allocation, no copy
        if (batch == 1 && seqLen == 1)
        {
            _finalNorm.ForwardInPlace(_cachedHidden);
            using var flatHidden = _cachedHidden.Reshape(batch, hiddenDim);
            return _logitOps.Project(flatHidden, batch, K, N, workspace);
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

        Tensor<float> result = _logitOps.Project(lastHidden, batch, K, N, workspace);
        lastHidden.Dispose();
        return result;
    }

    private void DisposeCache()
    {
        // _cachedHidden may alias _cachedEmbedding (blocks return input tensor in-place).
        // Dispose embedding first; if hidden is the same object, skip its disposal.
        var emb = _cachedEmbedding;
        var hid = _cachedHidden;
        _cachedEmbedding?.Dispose();
        if (hid != null && !ReferenceEquals(hid, emb))
            hid.Dispose();
        _cachedNormed?.Dispose();
        _cachedEmbedding = null;
        _cachedHidden = null;
        _cachedNormed = null;
    }

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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _embedding.Dispose();
        _arch.Dispose();
        _finalNorm.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(Transformer));

    [Obsolete("Use Forward() with workspace instead.")]
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
        using var logits = new Tensor<float>(batch * seqLen, _weights.Config.VocabSize);
        unsafe
        {
            var fn = _qOps!.QuantizedMatMulOpFor(QuantDType.F32);
            fn(normedFlat.DataPtr, (byte*)projectionWeight.DataPtr, logits.DataPtr, batch * seqLen, _weights.Config.HiddenDim, _weights.Config.VocabSize);
        }
        
        var result = logits.Reshape(batch, seqLen, _weights.Config.VocabSize);
        
        var state = new TransformerState(positionOffset)
        {
            TokenIds = tokenIds,
            Embedded = embedded,
            Hidden = hidden,
            Normed = normed,
            Logits = result
        };
        
        return (result, state);
    }

    [Obsolete("Training path placeholder — will be removed in v2.")]
    public Tensor<float> Backward(Tensor<float> gradLogits, TransformerState state)
    {
        int batch = state.TokenIds.Shape.Rows;
        int seqLen = state.TokenIds.Shape.Cols;
        int hidden = _weights.Config.HiddenDim;
        
        // gradLogitsFlat [M, Vocab] @ W [Vocab, Hidden] → gradNormedFlat [M, Hidden]
        using var embeddingBT = _embedding.Weight.Transpose();
        var gradLogitsFlat = gradLogits.Reshape(batch * seqLen, _weights.Config.VocabSize);
        using var gradNormedFlat = new Tensor<float>(batch * seqLen, _weights.Config.HiddenDim);
        unsafe
        {
            var fn = _qOps!.QuantizedMatMulOpFor(QuantDType.F32);
            fn(gradLogitsFlat.DataPtr, (byte*)embeddingBT.DataPtr, gradNormedFlat.DataPtr, batch * seqLen, _weights.Config.VocabSize, _weights.Config.HiddenDim);
        }
        
        using var gradHidden = _finalNorm.Backward(gradNormedFlat.Reshape(batch, seqLen, hidden), new NormLayerState(1, hidden));


        // Backward through architecture - simplified pass-through
        var gradInput = new Tensor<float>(batch, seqLen, hidden);

        return gradInput;
    }
}
