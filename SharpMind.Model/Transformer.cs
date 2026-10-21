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

    // Gemma-4 per-layer-embedding builder; null for every other architecture.
    private readonly PerLayerInputs? _perLayerInputs;

    // Learned positional embeddings [MaxSeqLen, HiddenDim]; null unless the
    // model config selects PositionalEncoding.Learned.
    private readonly Tensor<float>? _positionEmbedding;

    // BERT-family token-type (segment) embeddings [NumTokenTypes, HiddenDim];
    // null unless the model carries a token-type table. A single-sequence encode
    // uses row 0 for every token.
    private readonly Tensor<float>? _tokenTypeEmbedding;

    // Separate LM head for non-weight-tied models (e.g. LLaMA 2/3).
    // Null means the model is weight-tied — the embedding weight is used instead.
    private readonly QuantizationOps? _qOps;
    private readonly LogitOps _logitOps;

    private readonly TransformerBlock[]? _blocks; // For training backward
    private readonly bool _gemmaEmbeddingScale;    // Gemma models scale embeddings by sqrt(hiddenDim)

    private Tensor<float>? _cachedEmbedding;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed;
    private Tensor<float>? _lastCachedHiddenSnapshot;

    /// <summary>Active quantization-aware training target, or null when disabled.</summary>
    public QuantDType? QuantAwareTrainingTarget { get; private set; }

    public Transformer(
        TransformerWeights weights,
        EmbeddingTable embedding,
        IArchitecture arch,
        NormLayer finalNorm,
        QuantizationOps? qOps = null,
        Dictionary<string, string>? mapping = null,
        bool gemmaEmbeddingScale = false,
        VisionEncoder? visionEncoder = null,
        AudioEncoder? audioEncoder = null,
        PerLayerInputs? perLayerInputs = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(embedding);
        ArgumentNullException.ThrowIfNull(arch);
        ArgumentNullException.ThrowIfNull(finalNorm);
        
        _weights = weights;
        _embedding = embedding;
        _arch = arch;
        _finalNorm = finalNorm;
        _qOps = qOps;
        _visionEncoder = visionEncoder;
        _audioEncoder = audioEncoder;
        _perLayerInputs = perLayerInputs;
        bool untied = weights.HasLmHead;
        var rawW = untied ? _weights.RawLmHead : _weights.RawEmbedding;
        var rawDtype = untied ? _weights.RawLmHeadDtype : _weights.RawEmbeddingDtype;
        // LogitOps reads the float weight only when there are no raw bytes, so an untied head
        // with raw bytes is never dequantized here.
        var projWeight = !untied ? _embedding.Weight : rawW is null ? _weights.LmHeadWeight : null;

        _gemmaEmbeddingScale = gemmaEmbeddingScale;
        _positionEmbedding = weights.PositionEmbedding;
        _tokenTypeEmbedding = weights.TokenTypeEmbedding;
        _logitOps = LogitOpsFactory.Create(projWeight, rawW, rawDtype, mapping);

        _blocks = arch.Blocks;
    }

    public ModelConfig Config => _weights.Config;

    /// <summary>
    /// The structured weight set backing this transformer. Exposed so the training
    /// side can export (or re-merge) the exact tensors the forward pass reads,
    /// e.g. after LoRA adapters have been folded back in.
    /// </summary>
    public TransformerWeights Weights => _weights;

    // Diagnostics accessors
    public NormLayer FinalNorm => _finalNorm;
    /// <summary>The untied output head as floats (dequantized on first access), or null for a tied model.</summary>
    public Tensor<float>? LmHead => _weights.LmHeadWeight;
    /// <summary>True when the model has an untied output head; unlike <see cref="LmHead"/> this does not dequantize it.</summary>
    public bool HasLmHead => _weights.HasLmHead;
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

    /// <summary>Disables the Q8_0 wide repack on all blocks. Streaming calls this after
    /// construction because the repack would rebuild on every layer reload.</summary>
    public void SuppressWide()
    {
        if (_blocks != null)
        {
            foreach (var block in _blocks)
                block.SuppressWide();
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
            if (attn != null)
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
    public byte[]? RawLmHead => _weights.RawLmHead;
    public QuantDType? RawLmHeadDtype => _weights.RawLmHeadDtype;
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
    /// Returns a Transformer-owned snapshot of the hidden state from the last
    /// Forward/ForwardLastLogits call. It remains valid until the next forward call
    /// or disposal. The single-token ForwardLastLogits path stores its normalized
    /// hidden state, matching its previous behavior.
    /// The snapshot is allocated lazily on first access, so standard/speculative
    /// decoding (which never reads it) avoids the copy entirely.
    /// </summary>
    public Tensor<float>? LastCachedHidden
    {
        get
        {
            EnsureLastCachedHiddenSnapshot();
            return _lastCachedHiddenSnapshot;
        }
    }

    /// <summary>
    /// Copies row <paramref name="positionIndex"/> from the last cached hidden state,
    /// applies final norm, and returns a new [1, HiddenDim] tensor.
    /// Returns null if no cached hidden state is available.
    /// </summary>
    public Tensor<float>? GetNormedHiddenRow(int positionIndex, SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        EnsureLastCachedHiddenSnapshot();
        if (_lastCachedHiddenSnapshot == null) return null;
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
        _lastCachedHiddenSnapshot.Data.Slice(positionIndex * hiddenDim, hiddenDim).CopyTo(row.Data);
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
            else if (_weights.HasLmHead && target == _weights.LmHeadWeight)
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
    public Tensor<float> Forward(Tensor<int> tokenIds, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null) => Forward(tokenIds, null, positionOffset, workspace);

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
        Core.Memory.IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        DisposeCache();

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
        ValidatePositionRange(positionOffset, totalLen);

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
        // Per-layer embeddings are indexed by text token ids, so they only apply to a
        // text-only pass; a fused image/audio prefix has no token ids.
        var perLayer = visionLen == 0 && audioLen == 0 ? BuildPerLayerInputs(textTokenIds, workspace) : null;
        try
        {
            _cachedHidden = _arch.Forward(_cachedEmbedding, new IKVCache[_arch.NumLayers], positionOffset, workspace, perLayer);
        }
        finally
        {
            perLayer?.Dispose();
        }

        if (_weights is TransformerWeightsStreaming sw) sw.PrepareForNextForward();

        // 5. Final norm + LM head → [Batch, totalLen, VocabSize]
        _cachedNormed?.Dispose();
        _cachedNormed = _finalNorm.Forward(_cachedHidden, workspace);

        using var normedFlat = _cachedNormed.Reshape(batch * totalLen, hiddenDim);
        int M = batch * totalLen;
        int K = hiddenDim;
        int N = _weights.Config.VocabSize;

        var logits = _logitOps.Project(normedFlat, M, K, N, workspace);
        ApplyFinalLogitSoftcap(logits);
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

    public Tensor<float> Forward(Tensor<int> tokenIds, IKVCache[]? caches, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        DisposeCache();
        ValidatePositionRange(positionOffset, tokenIds.Shape.Cols);

        // 1. Token embeddings → [Batch, SeqLen, HiddenDim]
        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);
        if (_gemmaEmbeddingScale)
            ScaleEmbedding(_cachedEmbedding, _weights.Config.HiddenDim);
        if (_positionEmbedding is not null)
            AddPositionEmbeddingInPlace(_cachedEmbedding, positionOffset);
        if (_tokenTypeEmbedding is not null)
            AddTokenTypeEmbeddingInPlace(_cachedEmbedding);

        // 2. Architecture (stack of transformer blocks).
        //    Blocks mutate in-place and return the same tensor, so _cachedHidden
        //    may alias _cachedEmbedding. Keep both alive until we exit this method.
        var perLayer = BuildPerLayerInputs(tokenIds, workspace);
        try
        {
            _cachedHidden = _arch.Forward(_cachedEmbedding, caches ?? new IKVCache[_arch.NumLayers], positionOffset, workspace, perLayer);
        }
        finally
        {
            perLayer?.Dispose();
        }

        // Streaming: keep the resident window warm for the next pass
        if (_weights is TransformerWeightsStreaming sw) sw.PrepareForNextForward();

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
        ApplyFinalLogitSoftcap(logits);
        var result = logits.Reshape(batch, seqLen, N);
        logits.Dispose();
        return result;
    }


    /// <summary>
    /// Inference fast path that returns logits only for the final token in each batch row.
    /// Input:  token IDs [Batch, SeqLen]
    /// Output: logits    [Batch, VocabSize]
    /// </summary>
    public Tensor<float> ForwardLastLogits(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null)
    {
        // One envelope per forward: decode steps land in Forward, multi-token prompt
        // chunks in Prefill, so a profiled window can never mix the two.
        bool decodeStep = tokenIds.Shape.Rows == 1 && tokenIds.Shape.Cols == 1;
        var tStep = DecodeProfiler.Begin();
        try
        {
            RunBlocks(tokenIds, caches, positionOffset, workspace);

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
                var lastLogits = _logitOps.Project(flatHidden, batch, K, N, workspace);
                ApplyFinalLogitSoftcap(lastLogits);
                return lastLogits;
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
            ApplyFinalLogitSoftcap(result);
            lastHidden.Dispose();
            return result;
        }
        finally
        {
            DecodeProfiler.Mark(decodeStep ? DecodeStage.Forward : DecodeStage.Prefill, tStep);
        }
    }

    /// <summary>
    /// Runs <paramref name="tokenIds"/> through the embedding and every block, appending their
    /// keys and values to <paramref name="caches"/>, but skips the final norm and the vocabulary
    /// projection. For prompt chunks whose logits nobody reads; the chunk that needs logits goes
    /// through <see cref="ForwardLastLogits"/>.
    /// </summary>
    public void ForwardFill(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null)
        => RunBlocks(tokenIds, caches, positionOffset, workspace);

    /// <summary>Embedding and blocks: the body <see cref="ForwardFill"/> and
    /// <see cref="ForwardLastLogits"/> share.</summary>
    private void RunBlocks(Tensor<int> tokenIds, IKVCache[] caches, int positionOffset, Core.Memory.IWorkspace? workspace)
    {
        ThrowIfDisposed();
        DisposeCache();
        ValidatePositionRange(positionOffset, tokenIds.Shape.Cols);

        var tEmbed = DecodeProfiler.Begin();
        _cachedEmbedding = _embedding.Forward(tokenIds, workspace);
        if (_gemmaEmbeddingScale)
            ScaleEmbedding(_cachedEmbedding, _weights.Config.HiddenDim);
        if (_positionEmbedding is not null)
            AddPositionEmbeddingInPlace(_cachedEmbedding, positionOffset);
        if (_tokenTypeEmbedding is not null)
            AddTokenTypeEmbeddingInPlace(_cachedEmbedding);
        DecodeProfiler.Mark(DecodeStage.Embed, tEmbed);

        var tBlocks = DecodeProfiler.Begin();
        var perLayer = BuildPerLayerInputs(tokenIds, workspace);
        try
        {
            _cachedHidden = _arch.Forward(_cachedEmbedding, caches, positionOffset, workspace, perLayer);
        }
        finally
        {
            perLayer?.Dispose();
        }
        DecodeProfiler.Mark(DecodeStage.Blocks, tBlocks);

        if (_weights is TransformerWeightsStreaming sw) sw.PrepareForNextForward();
    }

    private void ValidatePositionRange(int positionOffset, int seqLen)
    {
        int limit = Config.MaxPosition;
        if (positionOffset < 0 || (long)positionOffset + seqLen > limit)
        {
            int last = seqLen > 0 ? positionOffset + seqLen - 1 : positionOffset;
            throw new InvalidOperationException(
                $"Sequence positions {positionOffset}..{last} exceed the model's context window of {limit} positions. " +
                "Reduce the prompt or generated length.");
        }
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
        _lastCachedHiddenSnapshot?.Dispose();
        _lastCachedHiddenSnapshot = null;
    }

    private void EnsureLastCachedHiddenSnapshot()
    {
        if (_lastCachedHiddenSnapshot is null && _cachedHidden is not null)
        {
            _lastCachedHiddenSnapshot = new Tensor<float>(_cachedHidden.Shape);
            _cachedHidden.Data.CopyTo(_lastCachedHiddenSnapshot.Data);
        }
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
        DisposeCache();
        _embedding.Dispose();
        _arch.Dispose();
        _finalNorm.Dispose();
        _visionEncoder?.Dispose();
        _audioEncoder?.Dispose();
        _perLayerInputs?.Dispose();
        // The Transformer borrows its TransformerWeights: CreateTransformer
        // never transfers ownership, so one weights instance can legitimately
        // back several transformers built and disposed over its lifetime (the
        // samples' generator/cache sweep, chat sessions across model files).
        // Disposing the weights here would tear the shared, quantized payload
        // (Raw* byte[]) out from under the next transformer — see the sandbox
        // OptionsRunner regression. The creator owns the weights and disposes
        // them; the CUI does so explicitly when its last session closes.
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
    /// Builds the gemma-4 per-layer input tensor for the current token ids, or null
    /// when the model has no per-layer embeddings. Requires <see cref="_cachedEmbedding"/>
    /// to be the (embedding-scaled) input the blocks will consume.
    /// </summary>
    private Tensor<float>? BuildPerLayerInputs(Tensor<int> tokenIds, Core.Memory.IWorkspace? workspace)
        => _perLayerInputs is null ? null : _perLayerInputs.Build(_cachedEmbedding!, tokenIds.Data, workspace);

    /// <summary>Applies gemma-4's final logit soft cap (<c>tanh(x/cap)*cap</c>) in place, when configured.</summary>
    private void ApplyFinalLogitSoftcap(Tensor<float> logits)
    {
        float cap = _weights.Config.FinalLogitSoftcap;
        if (cap <= 0f) return;
        float inv = 1f / cap;
        var d = logits.Data;
        for (int i = 0; i < d.Length; i++)
            d[i] = cap * MathF.Tanh(d[i] * inv);
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
        int S = emb.Shape[emb.Shape.Length - 2];
        int batch = emb.ElementCount / (S * H);
        int maxPos = pos.Shape.Rows;
        var src = pos.Data;
        var dst = emb.Data;
        int embStride = S * H;

        for (int b = 0; b < batch; b++)
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

    /// <summary>
    /// Adds the token-type embedding for segment 0 to every token of
    /// <paramref name="emb"/> in place. A single-sequence encode uses one segment
    /// (BERT's segment B is only set for sentence pairs), matching the all-zero
    /// token-type ids the tokenizer emits.
    /// </summary>
    private void AddTokenTypeEmbeddingInPlace(Tensor<float> emb)
    {
        var type = _tokenTypeEmbedding!;
        int H = _weights.Config.HiddenDim;
        var src = type.Data;
        var dst = emb.Data;
        for (int i = 0; i < dst.Length; i++)
            dst[i] += src[i % H];
    }

    /// <summary>
    /// Runs the encoder over <paramref name="tokenIds"/> and returns the
    /// final-normed hidden states [Batch, SeqLen, HiddenDim] without the
    /// vocabulary projection. The returned tensor is owned by this transformer
    /// and stays valid until the next forward call or disposal.
    /// </summary>
    public Tensor<float> EncodeHidden(Tensor<int> tokenIds, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        RunBlocks(tokenIds, new IKVCache[_arch.NumLayers], positionOffset, workspace);
        _cachedNormed?.Dispose();
        _cachedNormed = _finalNorm.Forward(_cachedHidden!, workspace);
        return _cachedNormed;
    }

    /// <summary>
    /// Encodes <paramref name="tokenIds"/> and pools the final-normed hidden
    /// states into one sentence embedding per batch row [Batch, HiddenDim].
    /// <paramref name="pooling"/> defaults to the model's declared strategy
    /// (falling back to CLS). When <paramref name="normalize"/> is true the
    /// embeddings are L2-normalised, matching sentence-embedding models such as
    /// bge. The returned tensor is caller-owned.
    /// </summary>
    public Tensor<float> PooledEmbedding(Tensor<int> tokenIds, PoolingType? pooling = null, bool normalize = true, int positionOffset = 0, Core.Memory.IWorkspace? workspace = null)
    {
        var hidden = EncodeHidden(tokenIds, positionOffset, workspace);
        var mode = pooling ?? (Config.Pooling != PoolingType.None ? Config.Pooling : PoolingType.Cls);
        int batch = hidden.Shape[0];
        int seqLen = hidden.Shape[1];
        int H = hidden.Shape[2];
        var pooled = new Tensor<float>(batch, H);
        var data = hidden.Data;
        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * seqLen * H;
            var dst = pooled.Data.Slice(b * H, H);
            switch (mode)
            {
                case PoolingType.Cls:
                    data.Slice(baseOffset, H).CopyTo(dst);
                    break;
                case PoolingType.Last:
                    data.Slice(baseOffset + (seqLen - 1) * H, H).CopyTo(dst);
                    break;
                default:
                    dst.Clear();
                    for (int s = 0; s < seqLen; s++)
                    {
                        var row = data.Slice(baseOffset + s * H, H);
                        for (int h = 0; h < H; h++) dst[h] += row[h];
                    }
                    for (int h = 0; h < H; h++) dst[h] /= seqLen;
                    break;
            }

            if (normalize)
            {
                double sum = 0;
                for (int h = 0; h < H; h++) sum += (double)dst[h] * dst[h];
                float inv = sum > 0 ? (float)(1.0 / Math.Sqrt(sum)) : 0f;
                for (int h = 0; h < H; h++) dst[h] *= inv;
            }
        }
        return pooled;
    }
}
