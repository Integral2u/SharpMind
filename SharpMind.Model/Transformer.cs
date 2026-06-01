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

    // Separate LM head for non-weight-tied models (e.g. LLaMA 2/3).
    // Null means the model is weight-tied — the embedding weight is used instead.
    private Tensor<float>? _lmHead;

    private readonly TransformerBlock[]? _blocks; // For training backward

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

        if (arch is DecoderArch decodeArch)
            _blocks = decodeArch.Blocks;
    }

    public ModelConfig Config => _config;

    // ── Diagnostics accessors ──────────────────────────────────────────────
    public NormLayer FinalNorm => _finalNorm;
    public Tensor<float>? LmHead => _lmHead;
    public Tensor<float> EmbeddingWeight => _embedding.Weight;
    public Tensor<float> ForwardEmbedding(Tensor<int> tokenIds) => _embedding.Forward(tokenIds);
    public TransformerBlock? GetBlock(int layer) => _blocks is not null && layer < _blocks.Length ? _blocks[layer] : null;
    public TensorOps Ops => _ops;

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        var lower = name.ToLower();

        if (lower.Contains("embed") || lower.Contains("token") || lower.Contains(" emb")
            || lower.Contains("wte") || lower.Contains("model.embed"))
        {
            int hidden = _config.HiddenDim;
            long expected = (long)_config.VocabSize * hidden;
            if (expected == data.Length)
            {
                data.CopyTo(_embedding.Weight.Data);
            }
            else if (data.Length > 0 && data.Length % hidden == 0 && data.Length < expected)
            {
                // Partial fill: GGUF has fewer vocab rows than the current VocabSize
                // (from injected special tokens). Copy existing rows, init new with mean.
                int copyLen = data.Length;
                data.CopyTo(_embedding.Weight.Data[..copyLen]);
                double sum = 0;
                for (int i = 0; i < copyLen; i++) sum += data[i];
                float mean = (float)(sum / copyLen);
                _embedding.Weight.Data[copyLen..].Fill(mean);
            }
            return true;
            /*  ------Old
            long expected = (long)_config.VocabSize * _config.HiddenDim;

            if (expected == data.Length)
            {
                int vocab = _config.VocabSize;
                int hidden = _config.HiddenDim;
                for (int v = 0; v < vocab; v++)
                    for (int h = 0; h < hidden; h++)
                        _embedding.Weight.Data[v * hidden + h] = data[h * vocab + v];
            }
            else
            {
                data.CopyTo(_embedding.Weight.Data);
            }
            return true;
            */
        }
        else if (lower.Contains("output_norm") || lower.Contains("norm") && lower.Contains("output"))
        {
            _finalNorm.LoadWeight(data);
            return true;
        }
        else if (lower.Contains("lm_head") || lower.StartsWith("output."))
        {
            int hidden = _config.HiddenDim;
            long expected = (long)_config.VocabSize * hidden;
            if (data.Length == expected)
            {
                _lmHead ??= new Tensor<float>(_config.VocabSize, hidden);
                data.CopyTo(_lmHead.Data);
            }
            else if (data.Length > 0 && data.Length % hidden == 0 && data.Length < expected)
            {
                // Partial fill: GGUF has fewer vocab rows. Init new rows to zero.
                _lmHead ??= new Tensor<float>(_config.VocabSize, hidden);
                int copyLen = data.Length;
                data.CopyTo(_lmHead.Data[..copyLen]);
                _lmHead.Data[copyLen..].Clear();
            }
            return true;
            /* --- old
            // LM head projection weight (GGUF: "output.weight", HF: "lm_head.weight").
            // Loaded into a dedicated tensor so the embedding table is never overwritten.
            // Falls back to the embedding at inference time if this weight is absent
            // (weight-tied models don't export it separately).
            long expected = (long)_config.VocabSize * _config.HiddenDim;
                if (data.Length == expected)
                {
                    _lmHead ??= new Tensor<float>(_config.VocabSize, _config.HiddenDim);
                    int vocab = _config.VocabSize;
                    int hidden = _config.HiddenDim;
                    for (int v = 0; v < vocab; v++)
                        for (int h = 0; h < hidden; h++)
                            _lmHead.Data[v * hidden + h] = data[h * vocab + v];
                }
            return true;
            */
        }
        else
        {
            return LoadDecoderWeight(name, data);
        }
    }

    private bool LoadDecoderWeight(string name, ReadOnlySpan<float> data)
    {
        if (_arch is DecoderArch dec)
            return dec.LoadWeight(name, data);
        return false;
    }

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        var lower = name.ToLower();
        if (lower.Contains("embed") || lower.Contains("token") || lower.Contains("wte") ||
            lower.Contains("output_norm") || lower.Contains("lm_head") || lower.StartsWith("output."))
            return false;
        return _arch is DecoderArch dec && dec.SetRawWeight(name, rawData, dtype);
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
    public unsafe Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0) => Forward(tokenIds, null, positionOffset);

    public unsafe Tensor<float> Forward(Tensor<int> tokenIds, IKVCache[]? caches, int positionOffset = 0)
    {
        ThrowIfDisposed();

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds);
        using var embedded = _cachedEmbedding;

        // 2. Architecture (stack of transformer blocks).
        //    Returns embedded (in-place residuals), so no separate using — embedded owns the buffer.
        _cachedHidden = _arch.Forward(embedded, caches ?? [], positionOffset);

        // 3. Final normalisation
        _cachedNormed = _finalNorm.Forward(embedded);
        using var normed = _cachedNormed;

        // 4. LM head: [Batch, SeqLen, HiddenDim] @ LmHead^T → [Batch, SeqLen, VocabSize]
        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hidden2 = _config.HiddenDim;

        using var normedFlat = normed.Reshape(batch * seqLen, hidden2);
        var projectionWeight = _lmHead ?? _embedding.Weight;
        var logits = _ops.MatMulWithBT(normedFlat, projectionWeight);

        // Restore [Batch, SeqLen, VocabSize]
        var result = logits.Reshape(batch, seqLen, _config.VocabSize);
        logits.Dispose();
        return result;
    }

    /// <summary>
    /// Inference fast path that returns logits only for the final token in each batch row.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, VocabSize]
    /// </summary>
    public unsafe Tensor<float> ForwardLastLogits(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset = 0)
    {
        ThrowIfDisposed();

        _cachedEmbedding = _embedding.Forward(tokenIds);
        using var embedded = _cachedEmbedding;

        _cachedHidden = _arch.Forward(embedded, caches, positionOffset);

        int batch = tokenIds.Shape.Rows;
        int seqLen = tokenIds.Shape.Cols;
        int hiddenDim = _config.HiddenDim;
        var projectionWeight = _lmHead ?? _embedding.Weight;

        // Single-token decode: normalise in-place, no allocation, no copy
        // _cachedHidden may be the same tensor as embedded (in-place residuals),
        // so we use embedded directly to avoid double-dispose of the shared buffer.
        if (batch == 1 && seqLen == 1)
        {
            _finalNorm.ForwardInPlace(embedded);
            using var flatEmbedded = embedded.Reshape(batch, hiddenDim);
            return _ops.MatMulWithBT(flatEmbedded, projectionWeight);
        }

        // Prefill path: extract last token's hidden state, norm in-place
        var lastHidden = new Tensor<float>(batch, hiddenDim);
        for (int b = 0; b < batch; b++)
        {
            int srcOffset = (b * seqLen + (seqLen - 1)) * hiddenDim;
            _cachedHidden.Data.Slice(srcOffset, hiddenDim).CopyTo(lastHidden.Data.Slice(b * hiddenDim, hiddenDim));
        }
        _finalNorm.ForwardInPlace(lastHidden);
        var logits = _ops.MatMulWithBT(lastHidden, projectionWeight);
        lastHidden.Dispose();
        return logits;
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

        using var normedFlat = normed.Reshape(batch * seqLen, _config.HiddenDim);
        var projectionWeight = _lmHead ?? _embedding.Weight;
        var logits = _ops.MatMulWithBT(normedFlat, projectionWeight);

        var result = logits.Reshape(batch, seqLen, _config.VocabSize);
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
        int hidden = _config.HiddenDim;

        // gradLogitsFlat [M, Vocab] @ W [Vocab, Hidden] → gradNormedFlat [M, Hidden]
        using var gradNormedFlat = _ops.MatMul(gradLogits.Reshape(batch * seqLen, _config.VocabSize), _embedding.Weight);

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