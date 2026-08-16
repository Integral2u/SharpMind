using SharpMind.Core.Embeddings;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Arch;
using SharpMind.Model.Config;
using SharpMind.Model.Encoders;
using SharpMind.Model.Layers;

namespace SharpMind.Model;

public sealed class Transformer : IDisposable
{
    private readonly TransformerWeights _weights;
    private readonly EmbeddingTable _embedding;
    private readonly IArchitecture _arch;
    private readonly NormLayer _finalNorm;
    private readonly VisionEncoder? _visionEncoder;
    private readonly AudioEncoder? _audioEncoder;
    private bool _disposed;

    // Learned positional embeddings [MaxSeqLen, HiddenDim]; null unless the
    // model config selects PositionalEncoding.Learned.
    private readonly Tensor<float>? _positionEmbedding;

    // Separate LM head for non-weight-tied models (e.g. LLaMA 2/3).
    // Null means the model is weight-tied — the embedding weight is used instead.
    private readonly Tensor<float>? _lmHead;
    private readonly QuantizationOps? _qOps;
    private readonly LogitOps _logitOps;

    private readonly TransformerBlock[]? _blocks; // For training backward
    private readonly bool _gemmaEmbeddingScale;    // Gemma models scale embeddings by sqrt(hiddenDim)

    private Tensor<float>? _cachedEmbedding;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed;

    /// <summary>Active quantization-aware training target, or null when disabled.</summary>
    public QuantDType? QuantAwareTrainingTarget { get; private set; }

    public Transformer(
        TransformerWeights weights,
        EmbeddingTable embedding,
        IArchitecture arch,
        NormLayer finalNorm,
        Tensor<float>? lmHead = null,
        QuantizationOps? qOps = null,
        Dictionary<string, string>? mapping = null,
        bool gemmaEmbeddingScale = false,
        VisionEncoder? visionEncoder = null,
        AudioEncoder? audioEncoder = null)
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
        _visionEncoder = visionEncoder;
        _audioEncoder = audioEncoder;
        var projWeight = _lmHead ?? _embedding.Weight;
        var rawW = _lmHead != null ? _weights.RawLmHead : _weights.RawEmbedding;
        var rawDtype = _lmHead != null ? _weights.RawLmHeadDtype : _weights.RawEmbeddingDtype;

        _gemmaEmbeddingScale = gemmaEmbeddingScale;
        _positionEmbedding = weights.PositionEmbedding;
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

    /// <summary>Learned positional-embedding table [MaxSeqLen, HiddenDim], or null when not in use.</summary>
    public Tensor<float>? PositionEmbedding => _positionEmbedding;
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

    /// <summary>
    /// Enables quantization-aware training across every linear layer in the
    /// transformer (attention + FFN projections). The tied LM head is covered by
    /// <see cref="BackpropEngine"/> using this property. F32/null = disabled.
    /// Block formats (Q8_0/Q4_0) throw via <see cref="TrainingLinearLayer"/> when
    /// a layer's weight dimensions are not multiples of 32; K-quant targets
    /// require a HiddenDim multiple of 128 and a flattened tied-head length that
    /// is a multiple of 256.
    /// </summary>
    public void EnableQuantAwareTraining(QuantDType? target)
    {
        int V = _weights.Config.VocabSize, H = _weights.Config.HiddenDim;
        if (target is QuantDType.Q8_0 or QuantDType.Q4_0 && (V % 32 != 0 || H % 32 != 0))
            throw new InvalidOperationException(
                $"QAT with {target} requires the tied-head weight [{V}, {H}] to be a " +
                "multiple of 32 in both dimensions. Use F16 or disable QAT.");
        if (IsKQuant(target) &&
            (H % 128 != 0 || ((long)V * H) % 256 != 0))
            throw new InvalidOperationException(
                $"QAT with {target} requires the tied-head weight [{V}, {H}] to have a " +
                $"HiddenDim multiple of 128 and a flattened length ({V * H}) that is a " +
                "multiple of 256. Use F16 or disable QAT.");
        QuantAwareTrainingTarget = target;
        if (_blocks is null) return;
        foreach (var block in _blocks)
        {
            var attn = block.Attention;
            foreach (var layer in new[] { attn.Wq, attn.Wk, attn.Wv, attn.Wo })
                layer.EnableQuantAwareTraining(target);
            foreach (var layer in new[] { block.Ffn.W1Layer, block.Ffn.W2Layer, block.Ffn.WGated, block.Ffn.WDown })
                layer?.EnableQuantAwareTraining(target);
        }
    }

    /// <summary>True when <paramref name="target"/> is a K-quant block format (Q2_K..Q8_K).</summary>
    private static bool IsKQuant(QuantDType? target) => target is
        QuantDType.Q2_K or QuantDType.Q2_K_S
        or QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L
        or QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M
        or QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M
        or QuantDType.Q6_K or QuantDType.Q6_K_S
        or QuantDType.Q8_K;

    public byte[]? RawEmbedding => _weights.RawEmbedding;
    public QuantDType? RawEmbeddingDtype => _weights.RawEmbeddingDtype;
    public QuantizationOps? QOps => _qOps;

    /// <summary>Vision encoder for image inputs, or null when the model is text-only.</summary>
    public VisionEncoder? VisionEncoder => _visionEncoder;
    /// <summary>Audio encoder for mel-spectrogram inputs, or null when the model is text-only.</summary>
    public AudioEncoder? AudioEncoder => _audioEncoder;

    /// <summary>
    /// Returns the distinct quantization dtypes used across the model's weight
    /// tensors (file storage dtypes, or F32 when tensors are resident as floats).
    /// Sorted by enum value; see <see cref="TransformerWeights.GetUsedQuantizations"/>.
    /// </summary>
    public QuantDType[] GetUsedQuantizations() => _weights.GetUsedQuantizations();

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
            // Only the (token) embedding and the LM head have raw storage; other
            // global tensors (e.g. learned position embeddings) stay float-backed.
            if (target == _weights.EmbeddingWeight)
            {
                _weights.RawEmbedding = rawData;
                _weights.RawEmbeddingDtype = dtype;
            }
            else if (_lmHead is not null && target == _weights.LmHeadWeight)
            {
                _weights.RawLmHead = rawData;
                _weights.RawLmHeadDtype = dtype;
            }
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

        if (_positionEmbedding is not null)
            yield return new Parameter("position_embedding", _positionEmbedding);

        foreach (var p in _arch.Parameters())
            yield return p;

        foreach (var p in _finalNorm.Parameters())
            yield return p;

        if (_visionEncoder is not null)
            foreach (var p in _visionEncoder.Parameters())
                yield return p;

        if (_audioEncoder is not null)
            foreach (var p in _audioEncoder.Parameters())
                yield return p;
    }

    /// <summary>
    /// Full forward pass.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, SeqLen, VocabSize]
    /// </summary>
    public Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0, Core.Memory.Workspace? workspace = null) => Forward(tokenIds, null, positionOffset, workspace);

    /// <summary>
    /// Multimodal forward pass: embeds optional image and audio inputs, fuses
    /// their tokens with the text embeddings along the sequence dimension, then
    /// runs the decoder stack. Modality tokens are placed before the text tokens,
    /// so text can attend to them causally.
    ///
    /// Input:  text  [Batch, SeqLen] token IDs; image [Batch, C, H, W]; mel [Batch, Frames, MelBins]
    /// Output: logits [Batch, VisionTokens + AudioTokens + SeqLen, VocabSize]
    /// </summary>
    public Tensor<float> ForwardMultimodal(
        Tensor<int> textTokenIds,
        Tensor<float>? image = null,
        Tensor<float>? mel = null,
        int positionOffset = 0,
        Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        if (image is not null && _visionEncoder is null)
            throw new InvalidOperationException("Model has no vision encoder — pass image only to a multimodal model.");
        if (mel is not null && _audioEncoder is null)
            throw new InvalidOperationException("Model has no audio encoder — pass mel only to a multimodal model.");

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        var textEmb = _embedding.Forward(textTokenIds, workspace);
        if (_gemmaEmbeddingScale)
            ScaleEmbedding(textEmb, _weights.Config.HiddenDim);

        // 2. Modality encoders → [Batch, N, HiddenDim] each
        Tensor<float>? visionTokens = null;
        Tensor<float>? audioTokens = null;
        if (image is not null) visionTokens = _visionEncoder!.Forward(image);
        if (mel is not null) audioTokens = _audioEncoder!.Forward(mel);

        int batch = textTokenIds.Shape.Rows;
        int textLen = textTokenIds.Shape.Cols;
        int visionLen = visionTokens?.Shape[1] ?? 0;
        int audioLen = audioTokens?.Shape[1] ?? 0;
        int totalLen = textLen + visionLen + audioLen;

        // 3. Fuse: [vision | audio | text] along the sequence axis.
        //    Owned by _cachedEmbedding (see DisposeCache); do NOT wrap in using.
        var fused = new Tensor<float>(batch, totalLen, _weights.Config.HiddenDim);
        var fusedData = fused.Data;
        int hiddenDim = _weights.Config.HiddenDim;
        int seqOffset = 0;
        if (visionTokens is not null)
        {
            CopyBlock(fusedData, visionTokens.Data, batch, visionTokens.Shape[1], hiddenDim, totalLen, seqOffset);
            seqOffset += visionTokens.Shape[1];
        }
        if (audioTokens is not null)
        {
            CopyBlock(fusedData, audioTokens.Data, batch, audioTokens.Shape[1], hiddenDim, totalLen, seqOffset);
            seqOffset += audioTokens.Shape[1];
        }
        CopyBlock(fusedData, textEmb.Data, batch, textLen, hiddenDim, totalLen, seqOffset);
        visionTokens?.Dispose();
        audioTokens?.Dispose();
        textEmb.Dispose();

        // 4. Architecture (blocks mutate in place; _cachedHidden may alias fused).
        _cachedEmbedding = fused;
        _cachedHidden = _arch.Forward(_cachedEmbedding, new IKVCache[_arch.NumLayers], positionOffset, workspace);

        if (_weights is TransformerWeightsStreaming sw) sw.CompleteForward();

        // 5. Final norm + LM head → [Batch, totalLen, VocabSize]
        _cachedNormed?.Dispose();
        _cachedNormed = _finalNorm.Forward(_cachedHidden, workspace);

        using var normedFlat = _cachedNormed.Reshape(batch * totalLen, hiddenDim);
        int M = batch * totalLen;
        int K = hiddenDim;
        int N = _weights.Config.VocabSize;

        var logits = _logitOps.Project(normedFlat, M, K, N, workspace);
        var result = logits.Reshape(batch, totalLen, N);
        logits.Dispose();
        return result;
    }

    /// <summary>
    /// Copies a [B, SeqLen, HiddenDim] block into the fused [B, TotalLen, HiddenDim]
    /// buffer at <paramref name="seqOffset"/>, treating the source as the leading
    /// rows of the sequence. Source and destination are both row-major; the source
    /// occupies columns seqOffset..seqOffset+SeqLen of each batch row while the
    /// destination spans the full TotalLen columns.
    /// </summary>
    private static void CopyBlock(Span<float> dst, ReadOnlySpan<float> src, int batch, int seqLen, int hiddenDim, int totalLen, int seqOffset)
    {
        // Layout note: for row-major [B, S, H] tensors, elements at buffer index
        // (b*S*H + s*H + h) map to destination (b*T*H + (seqOffset+s)*H + h) in the
        // fused [B, T, H] tensor.
        int seqStride = seqLen * hiddenDim;
        int totalStride = totalLen * hiddenDim;
        int offStride = seqOffset * hiddenDim;
        for (int b = 0; b < batch; b++)
            src.Slice(b * seqStride, seqStride).CopyTo(dst.Slice(b * totalStride + offStride, seqStride));
    }

    public Tensor<float> Forward(Tensor<int> tokenIds, IKVCache[]? caches, int positionOffset = 0, Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);
        if (_gemmaEmbeddingScale)
            ScaleEmbedding(_cachedEmbedding, _weights.Config.HiddenDim);
        if (_positionEmbedding is not null)
            AddPositionEmbeddingInPlace(_cachedEmbedding, positionOffset);

        // 2. Architecture (stack of transformer blocks).
        //    Blocks mutate in-place and return the same tensor, so _cachedHidden
        //    may alias _cachedEmbedding. Keep both alive until we exit this method.
        _cachedHidden = _arch.Forward(_cachedEmbedding, caches ?? new IKVCache[_arch.NumLayers], positionOffset, workspace);

        // Streaming: free any remaining loaded layers before the next pass
        if (_weights is TransformerWeightsStreaming sw) sw.CompleteForward();

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
        if (_gemmaEmbeddingScale)
            ScaleEmbedding(_cachedEmbedding, _weights.Config.HiddenDim);
        if (_positionEmbedding is not null)
            AddPositionEmbeddingInPlace(_cachedEmbedding, positionOffset);

        _cachedHidden = _arch.Forward(_cachedEmbedding, caches, positionOffset, workspace);

        if (_weights is TransformerWeightsStreaming sw) sw.CompleteForward();

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
        int lastOffset = (seqLen - 1) * hiddenDim;
        var cachedData = _cachedHidden.Data;
        var lastData = lastHidden.Data;
        for (int b = 0; b < batch; b++)
        {
            int srcOffset = b * seqLen * hiddenDim + lastOffset;
            cachedData.Slice(srcOffset, hiddenDim).CopyTo(lastData.Slice(b * hiddenDim, hiddenDim));
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
        _visionEncoder?.Dispose();
        _audioEncoder?.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(Transformer));

    /// <summary>Gemma models scale token embeddings by sqrt(hiddenDim) before the first block.</summary>
    private static void ScaleEmbedding(Tensor<float> embedding, int hiddenDim)
    {
        float scale = MathF.Sqrt(hiddenDim);
        var data = embedding.Data;
        for (int i = 0; i < data.Length; i++)
            data[i] *= scale;
    }

    /// <summary>
    /// Adds the learned position-embedding row for each sequence position into
    /// <paramref name="emb"/> in place (GPT-2 style: x = wte(idx) + wpe(pos)).
    /// Positions are indexed from <paramref name="positionOffset"/>; positions at
    /// or past <see cref="ModelConfig.MaxSeqLen"/> are clamped to the last row so
    /// decoding past the trained length degrades gracefully instead of faulting.
    /// </summary>
    private void AddPositionEmbeddingInPlace(Tensor<float> emb, int positionOffset)
    {
        var pos = _positionEmbedding!;
        int H = _weights.Config.HiddenDim;
        int S = emb.Shape.Cols;
        int maxPos = pos.Shape.Rows;
        var src = pos.Data;
        var dst = emb.Data;
        int embStride = S * H;

        for (int b = 0; b < emb.Shape.Rows; b++)
        {
            int ptr = b * embStride;
            for (int s = 0; s < S; s++)
            {
                int p = positionOffset + s;
                if (p >= maxPos) p = maxPos - 1;
                int pBase = p * H;
                for (int h = 0; h < H; h++)
                    dst[ptr + h] += src[pBase + h];
                ptr += H;
            }
        }
    }
}
