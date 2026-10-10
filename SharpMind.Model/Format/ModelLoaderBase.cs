using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;

/// <summary>
/// Shared implementation for <see cref="GgufLoader"/> and <see cref="SmmLoader"/>.
///
/// The two containers differ only in how their tensor index is read, how many
/// bytes a tensor occupies on disk, and nothing else: target resolution,
/// raw-quantized-data handling, fused QKV splitting, fused MoE expert plane
/// slicing, vocab padding, dequantization and the GGUF transpose semantics are
/// identical, so they live here exactly once.
/// </summary>
public abstract class ModelLoaderBase(
    QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false, int maxParallelLoadDegree = 0) : IModelLoader
{
    protected readonly QuantizationOps _qOps = qOps ?? throw new ArgumentNullException(nameof(qOps));
    protected readonly string _path = File.Exists(path) ? path : throw new FileNotFoundException(path);
    protected readonly ModelConfig _config = config ?? throw new ArgumentNullException(nameof(config));
    protected readonly bool _useSafeIo = useSafeIo;

    /// <summary>
    /// How many threads a full (<see cref="LoadMode.Full"/>) load may fan out
    /// across. 0 — the default — means one per core; 1 restores the original
    /// single-threaded, index-order loop. See <see cref="LoadAllWeights"/>.
    /// </summary>
    internal readonly int MaxParallelLoadDegree = maxParallelLoadDegree;

    /// <summary>Reads the container's tensor index and metadata once (header, KV pairs, tensor list).</summary>
    protected abstract ModelMetaData LoadIndex();

    /// <summary>
    /// The on-disk byte count of a tensor. GGUF aligns each row to its quant
    /// block boundary (<see cref="QuantizationOps.GetRawTensorByteCount"/>); SMM packs
    /// blocks flat over the quantized buffer
    /// (<see cref="QuantizationOps.GetFlatTensorByteCount"/>).
    /// </summary>
    protected virtual long RawByteCount(TensorInfo info)
        => QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);

    public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null, CancellationToken? cancellationToken = null)
    {
        Core.Memory.NativeBufferPool<float>.Clear();

        var meta = LoadIndex();
        weights.GgufMeta = meta;
        weights.GgufPath = _path;
        // IsMoEGguf (not ".exps.") so fused 3D expert stacks are MoE too.
        weights.IsMoE = TransformerWeights.IsMoEGguf(meta.Tensors);

        int total = meta.Tensors.Count;
        int loaded = 0;

        // Fan out by default (degree 0 = one per core). Degree 1 keeps the original
        // index-order single-threaded loop byte-for-byte.
        int degree = ParallelTensorLoad.ResolveDegree(MaxParallelLoadDegree, total, _useSafeIo);
        if (degree > 0)
        {
            LoadAllWeightsParallel(weights, meta, progress, cancellationToken, degree, total);
            return;
        }

        using var stream = ModelFileIo.OpenModelStream(_path, _useSafeIo);
        foreach (var info in meta.Tensors)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            progress?.Report((float)loaded / total);
            LoadSingleTensor(weights, meta, stream, info);
            loaded++;
        }
        progress?.Report(1f);
    }

    /// <summary>
    /// Parallel full load. Tensors are partitioned by the block they resolve
    /// to, so every worker owns a disjoint set of <see cref="TransformerWeights.BlockWeights"/>
    /// and there is no shared mutable state to lock: each worker's writes land
    /// in its own blocks' raw fields, tensor-metadata dictionaries and float
    /// tensors (all plain, non-concurrent collections created with <c>??=</c>,
    /// which is exactly what would break under two threads on one block).
    ///
    /// Tensors that resolve to no block — the embedding, the output head, the
    /// final norms — instead write single shared fields on
    /// <paramref name="weights"/>, and the head additionally self-registers
    /// under a lazy-creation guard. They run together on the calling thread, so
    /// a model whose vocab dominates pays that tail serially; correctness of
    /// those few fields is worth more than the extra overlap.
    ///
    /// Each worker opens its own file view because <see cref="LoadSingleTensor"/>
    /// drives the stream by <c>Position</c>, which is mutable per-stream state.
    /// </summary>
    private void LoadAllWeightsParallel(
        TransformerWeights weights, ModelMetaData meta,
        IProgress<float>? progress, CancellationToken? cancellationToken,
        int degree, int total)
    {
        // Bucket by block identity. ResolveTarget is called once per tensor here
        // (it was already being called per tensor in the sequential loop, just
        // inside LoadSingleTensor) so this only moves it earlier.
        var blockBuckets = new Dictionary<TransformerWeights.BlockWeights, List<TensorInfo>>(ReferenceEqualityComparer.Instance);
        var nonBlock = new List<TensorInfo>();
        foreach (var info in meta.Tensors)
        {
            // The untied head resolves to a null target before its lazy tensor is
            // materialized; route it explicitly so it never lands in a block bucket.
            if (TransformerWeights.IsLmHeadTensorName(info.Name)) { nonBlock.Add(info); continue; }
            var (_, block, _) = weights.ResolveTarget(info.Name);
            if (block is null) { nonBlock.Add(info); continue; }
            if (!blockBuckets.TryGetValue(block, out var bucket))
                blockBuckets[block] = bucket = [];
            bucket.Add(info);
        }

        // One work item per block, kept whole: splitting a block across workers
        // would put two threads in the same raw-field and tensor-meta
        // dictionaries, which are plain (non-concurrent) collections.
        var items = new List<List<TensorInfo>>(blockBuckets.Count);
        foreach (var bucket in blockBuckets.Values) items.Add(bucket);

        int loaded = 0;
        var reporter = new ParallelTensorLoad.ProgressReporter(progress, total);
        void ReportProgress() => reporter.ReportLoaded(Interlocked.Increment(ref loaded));

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = degree,
            CancellationToken = cancellationToken ?? CancellationToken.None,
        };

        // Worker bodies own their stream and only touch their own tensors.
        void RunItems(List<List<TensorInfo>> slice)
        {
            if (slice.Count == 0) return;
            using var stream = ModelFileIo.OpenModelStream(_path, _useSafeIo);
            foreach (var bucket in slice)
                foreach (var info in bucket)
                {
                    cancellationToken?.ThrowIfCancellationRequested();
                    LoadSingleTensor(weights, meta, stream, info);
                    ReportProgress();
                }
        }

        var blockWork = ParallelTensorLoad.PartitionByWeight(items, degree, bucket =>
        {
            // Weight the buckets by their on-disk byte count; see RawByteCount.
            long bytes = 0;
            foreach (var info in bucket)
                bytes += RawByteCount(info);
            return bytes;
        });
        try
        {
            if (blockWork.Count > 0)
                Parallel.For(0, blockWork.Count, options, i => RunItems(blockWork[i]));

            // Non-block tensors last, on this thread, so progress still reaches 1.
            if (nonBlock.Count > 0) RunItems([nonBlock]);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (AggregateException ex) when (ex.InnerExceptions.Count == 1 && ex.InnerException is not null)
        {
            // Surface the real failure (bad tensor, I/O error) rather than the
            // Parallel wrapper, matching what the sequential loop would throw.
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }

        reporter.ReportComplete();
    }

    public bool SupportsExpertSlicing => true;

    /// <summary>Loads a whole layer (all routed experts included).</summary>
    public void LoadLayerWeights(int layerIndex, TransformerWeights weights, CancellationToken? cancellationToken = null)
        => LoadLayerTensors(layerIndex, weights, includeExperts: true, expertFilter: null, cancellationToken);

    /// <summary>Loads a layer without the routed-expert planes (streaming MoE residency).</summary>
    public void LoadLayerNonExpertWeights(int layerIndex, TransformerWeights weights, CancellationToken? cancellationToken = null)
        => LoadLayerTensors(layerIndex, weights, includeExperts: false, expertFilter: null, cancellationToken);

    /// <summary>Loads only the given routed experts' planes — fused 3D stacks are sliced per
    /// plane, named ".exps." exports and per-expert bias tensors are whole-tensor reads. Used
    /// by streaming MoE residency on demand when the router selects an expert that is absent.</summary>
    public void LoadExpertWeights(int layerIndex, IReadOnlyList<int> expertIndices, TransformerWeights weights, CancellationToken? cancellationToken = null)
        => LoadLayerTensors(layerIndex, weights, includeExperts: true, expertFilter: new HashSet<int>(expertIndices), cancellationToken);

    /// <summary>True for tensor names that carry routed-expert bytes (fused plane stacks or
    /// per-expert named tensors). Shared experts, the router, norms and attention are not routed.</summary>
    private static bool IsRoutedExpertField(string rawField) =>
        FusedExpertLayout.IsFusedField(rawField) ||
        rawField.StartsWith("RawWgateExp_", StringComparison.Ordinal) ||
        rawField.StartsWith("RawWupExp_", StringComparison.Ordinal) ||
        rawField.StartsWith("RawWdownExp_", StringComparison.Ordinal);

    /// <summary>Parses the expert index out of a per-expert named field ("RawWgateExp_5" → 5),
    /// or null for fused plane stacks (filtered per plane in LoadSingleTensor).</summary>
    private static int? ParseExpertIndex(string rawField)
    {
        foreach (string prefix in new[] { "RawWgateExp_", "RawWupExp_", "RawWdownExp_" })
        {
            if (rawField.StartsWith(prefix, StringComparison.Ordinal) &&
                int.TryParse(rawField.AsSpan(prefix.Length), out int idx))
                return idx;
        }
        return null;
    }

    /// <summary>True for tensor names that carry per-expert bytes (named planes or the per-expert
    /// bias tensors, whose rawField is null). Mirrors FfnLayer.SetRawWeight's expert detection so
    /// the on-demand residency load recognises both with the same rules.</summary>
    private static bool IsPerExpertName(string name) =>
        name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
        name.Contains("_exps", StringComparison.OrdinalIgnoreCase) ||
        (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
        (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
        (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name));

    /// <summary>Parses the expert index out of a per-expert tensor name ("blk.0.exps.3.ffn_gate.bias" → 3),
    /// for tensors with no raw field to index by.</summary>
    private static int? ParseNameExpertIndex(string name)
    {
        var m = RegexGenerated.ExpertIndex.Match(name);
        if (m.Success && int.TryParse(m.Groups[1].Value, out int idx)) return idx;
        return null;
    }

    private void LoadLayerTensors(
        int layerIndex, TransformerWeights weights, bool includeExperts, IReadOnlySet<int>? expertFilter, CancellationToken? cancellationToken)
    {
        var meta = weights.GgufMeta;
        if (meta == null)
        {
            meta = LoadIndex();
            weights.GgufMeta = meta;
            weights.GgufPath = _path;
            // IsMoEGguf (not ".exps.") so fused 3D expert stacks converted from GGUF are MoE too.
            weights.IsMoE = TransformerWeights.IsMoEGguf(meta.Tensors);
        }

        var targetBlock = layerIndex < weights.Blocks.Length ? weights.Blocks[layerIndex] : null;
        if (targetBlock == null) return;

        using var stream = ModelFileIo.OpenModelStream(_path, _useSafeIo);

        foreach (var info in meta.Tensors)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            var (_, block, rawField) = weights.ResolveTarget(info.Name);
            if (block != targetBlock) continue;

            bool isRoutedExpert = rawField != null && IsRoutedExpertField(rawField);
            if (!includeExperts && isRoutedExpert) continue;          // residency: layer without experts
            if (expertFilter != null)
            {
                // The on-demand pass must also grab the per-expert bias tensors. Those resolve
                // to float block fields (rawField null, so isRoutedExpert is false) — without
                // the name check they fall out of the filter and a resident expert runs with a
                // zeroed bias while an identical full load applies the real one.
                bool isExpertScoped = isRoutedExpert || IsPerExpertName(info.Name);
                if (!isExpertScoped) continue;                        // on-demand: expert tensors only
                if (!FusedExpertLayout.IsFusedField(rawField ?? string.Empty))
                {
                    // Per-expert named tensors are whole-tensor reads: drop the ones not asked for.
                    // Bias tensors carry no raw field, so index them by their tensor name.
                    int? idx = ParseExpertIndex(rawField ?? string.Empty) ?? ParseNameExpertIndex(info.Name);
                    if (idx.HasValue && !expertFilter.Contains(idx.Value)) continue;
                }
            }

            LoadSingleTensor(weights, meta, stream, info, expertFilter);
        }
    }

    public void LoadGlobalTensors(TransformerWeights weights, CancellationToken? cancellationToken = null)
    {
        var meta = weights.GgufMeta ?? LoadIndex();
        if (weights.GgufMeta == null)
        {
            weights.GgufMeta = meta;
            weights.GgufPath = _path;
            // IsMoEGguf (not ".exps.") so fused 3D expert stacks converted from GGUF are MoE too.
            weights.IsMoE = TransformerWeights.IsMoEGguf(meta.Tensors);
        }

        using var stream = ModelFileIo.OpenModelStream(_path, _useSafeIo);

        foreach (var info in meta.Tensors)
        {
            cancellationToken?.ThrowIfCancellationRequested();
            var (target, block, _) = weights.ResolveTarget(info.Name);
            // The untied head must be routed even when ResolveTarget returns a null
            // target: a streaming load has no LmHeadWeight until LoadSingleTensor
            // creates it lazily, so target is null for "output.weight" there and the
            // tensor would otherwise be skipped, silently generating from the embedding.
            if ((target != null && block == null) || TransformerWeights.IsLmHeadTensorName(info.Name))
                LoadSingleTensor(weights, meta, stream, info);
        }
    }

    private void LoadSingleTensor(
        TransformerWeights weights, ModelMetaData meta,
        Stream stream, TensorInfo info, IReadOnlySet<int>? expertFilter = null)
    {
        // The untied output head keeps only its raw bytes here; TransformerWeights.LmHeadWeight
        // dequantizes them on first float access, since the CPU projection reads the raw bytes.
        bool isLmHead = TransformerWeights.IsLmHeadTensorName(info.Name);
        var (target, block, rawField) = isLmHead ? default : weights.ResolveTarget(info.Name);

        if (isLmHead && !weights.HasLmHead)
        {
            // Canonical GGUFs store output.weight as [input, vocab]; our own SMM->GGUF
            // export writes the tensor's in-memory [vocab, input] order (same bytes either
            // way — the input dim is contiguous in both). The input dim is whichever shape
            // entry is not the vocab size; trusting Shape[0] blindly built a [vocab, vocab]
            // head for exported fine-tunes, which overflows int at real vocab sizes.
            long ggufIn = info.Shape[0];
            if (info.Shape.Length > 1 && ggufIn == _config.VocabSize) ggufIn = info.Shape[1];
            int lmRows = TensorLoadHelper.CheckedInt(_config.VocabSize, "VocabSize for LmHead");
            int lmCols = TensorLoadHelper.CheckedInt(ggufIn, "LmHead input dim");
            weights.SetLazyLmHead(() => DequantizeLmHead(weights, lmRows, lmCols, info.Shape));
        }

        if (target == null && block == null && !isLmHead) return;

        // The byte span differs by container (row-aligned for GGUF, flat for SMM).
        long rawSize = RawByteCount(info);
        if (rawSize <= 0) return;

        // Record tensor metadata and top-level dtypes (consumed by SetWeights later)
        if ((isLmHead || target != null) && block == null && rawSize > 0)
        {
            if (isLmHead)
                weights.RawLmHeadDtype = info.Dtype;
            else if (target == weights.EmbeddingWeight)
                weights.RawEmbeddingDtype = info.Dtype;
        }
        if (block != null && rawField != null && rawSize > 0)
            SetTensorMeta(block, rawField, meta.DataOffset + info.Offset, TensorLoadHelper.CheckedInt(rawSize, "rawSize"), info.Dtype);

        long targetOffset = meta.DataOffset + info.Offset;
        if (targetOffset >= stream.Length) return;
        stream.Position = targetOffset;

        // Fused QKV (e.g. Phi-3: "blk.N.attn_qkv.weight" → [out, 3*in]): split
        // into separate Q/K/V byte buffers so each InferenceLinearLayer gets the
        // exact [out, in] chunk its size guard expects. The three parts sit back
        // to back in the file, so read straight into each pooled buffer instead
        // of staging the whole tensor.
        if (block != null && rawField == "RawWqkv" && stream.Position + rawSize <= stream.Length)
        {
            if (rawSize % 3 != 0)
                throw new InvalidDataException(
                    $"Fused QKV tensor '{info.Name}' has {rawSize} bytes which is not divisible by 3.");

            int partSize = (int)(rawSize / 3);
            byte[] qPart = weights.AllocateRawBuffer(partSize);
            byte[] kPart = weights.AllocateRawBuffer(partSize);
            byte[] vPart = weights.AllocateRawBuffer(partSize);
            stream.ReadExactly(qPart);
            stream.ReadExactly(kPart);
            stream.ReadExactly(vPart);

            SetRawField(block, "RawWq", qPart, info.Dtype);
            SetRawField(block, "RawWk", kPart, info.Dtype);
            SetRawField(block, "RawWv", vPart, info.Dtype);

            // Record metadata for each split portion (offsets are approximate;
            // the streaming forward path reads RawWq/RawWk/RawWv directly).
            SetTensorMeta(block, "RawWq", targetOffset, partSize, info.Dtype);
            SetTensorMeta(block, "RawWk", targetOffset + partSize, partSize, info.Dtype);
            SetTensorMeta(block, "RawWv", targetOffset + partSize * 2, partSize, info.Dtype);
            return;
        }

        // Fused MoE expert stack. qwen2moe packs all experts of one FFN
        // projection into a single 3D tensor with no per-expert index in the
        // name, e.g. "blk.N.ffn_gate_exps.weight" with GGUF shape
        // [in, out, num_experts] (e.g. [2048, 1408, 60]). GGUF lists dims
        // fastest-varying first, so the expert axis is the LAST dim and expert
        // e owns the contiguous byte range [e*plane, (e+1)*plane) — exactly the
        // [in, out] chunk each InferenceLinearLayer's size guard expects. A
        // GGUF→SMM conversion copies such a tensor verbatim, so its planes stay
        // contiguous in the .SMM data region too.
        if (block != null && rawField != null && info.Shape.Length == 3 && FusedExpertLayout.IsFusedField(rawField))
        {
            // Plane geometry (expert axis, per-expert byte range) is shared with the
            // streaming metadata pass so the two cannot drift; see FusedExpertLayout.
            var layout = FusedExpertLayout.Create(rawField, info.Shape, info.Dtype);

            long needed = targetOffset + layout.OffsetOf(layout.NumExperts);
            if (needed > stream.Length)
                throw new InvalidDataException(
                    $"Fused MoE expert tensor '{info.Name}' needs {needed} bytes " +
                    $"({layout.NumExperts} experts x {layout.PlaneSizeBytes}) but the file is only {stream.Length} bytes.");

            // Experts are contiguous planes in the file, so read each one straight
            // from the stream instead of staging the whole stack first. The fused
            // buffer was ~97 MB per tensor (down_exps) and, with three such tensors
            // per layer across 24 layers, that transient LOH churn was enough to
            // fault during a full load. An on-demand expert load skips planes outside
            // the filter by advancing the stream past them (they stay contiguous).
            stream.Position = targetOffset;
            for (int e = 0; e < layout.NumExperts; e++)
            {
                if (expertFilter is not null && !expertFilter.Contains(e))
                {
                    stream.Position += layout.PlaneSizeBytes;
                    continue;
                }

                byte[] expert = weights.AllocateRawBuffer(layout.PlaneSizeBytes);
                stream.ReadExactly(expert);

                string targetField = $"{layout.FieldPrefix}{e}";
                SetRawField(block, targetField, expert, info.Dtype);
                SetTensorMeta(block, targetField, targetOffset + layout.OffsetOf(e), layout.PlaneSizeBytes, info.Dtype);
            }
            return;
        }

        bool withinFile = stream.Position + rawSize <= stream.Length;

        // Load raw quantized data for block-level tensors. Block tensors rent from the
        // streaming buffer pool (alloc-free on reload); top-level tensors below keep a
        // fresh array since they are loaded once and never returned.
        byte[]? rawData = null;
        if (block != null && rawField != null && withinFile)
        {
            rawData = weights.AllocateRawBuffer(checked((int)rawSize));
            stream.ReadExactly(rawData);
            SetRawField(block, rawField, rawData, info.Dtype);
        }

        // Load raw quantized data for top-level tensors (embedding, output head),
        // zero-padding short tensors up to _config.VocabSize rows (extra template
        // tokens are appended past the tensor's own vocab block).
        if ((isLmHead || target != null) && block == null && withinFile)
        {
            byte[] data;
            if (info.Shape.Length >= 2)
            {
                long tensorVocab = Math.Max(info.Shape[0], info.Shape[1]);
                long paddedVocab = _config.VocabSize;
                if (paddedVocab > tensorVocab)
                {
                    long colBytes = rawSize / tensorVocab;
                    int safeColBytes = TensorLoadHelper.CheckedInt(colBytes, "colBytes");
                    data = new byte[paddedVocab * colBytes];
                    for (long r = 0; r < tensorVocab; r++)
                        stream.ReadExactly(data, (int)(r * colBytes), safeColBytes);
                }
                else
                {
                    data = new byte[rawSize];
                    stream.ReadExactly(data);
                }
            }
            else
            {
                data = new byte[rawSize];
                stream.ReadExactly(data);
            }

            if (isLmHead)
            {
                weights.RawLmHead = data;
                weights.RawLmHeadDtype = info.Dtype;
            }
            else if (target == weights.EmbeddingWeight)
            {
                weights.RawEmbedding = data;
                weights.RawEmbeddingDtype = info.Dtype;
            }
            rawData = data;
        }
        if (isLmHead) return;

        // Oversized tensor (more elements than a single float buffer can hold): the raw
        // bytes are already loaded above for block-level and top-level tensors. Skip the
        // dequant — the streaming forward pass reads the raw quantized bytes directly.
        long longCount = TensorLoadHelper.ComputeElementCount(info.Shape);
        if (longCount > int.MaxValue) return;
        int count = (int)longCount;

        // A block tensor with no float target (a quantized-resident load keeps only its raw
        // bytes) has nothing to dequantize into, so skip the dequant.
        var blockFloatTarget = target == null && block != null ? weights.ResolveFloatTarget(info.Name) : null;
        if (target == null && blockFloatTarget == null) return;

        // For non-block tensors (embedding, lm_head, norms) read directly into
        // target.Data — no temp buffer needed. This saves ~600 MB for Bonsai-8B's
        // token_embd.weight [151936, 1024] which previously allocated a duplicate
        // float buffer via ArrayPool power-of-2 bucketing on top of the target tensor.
        if (target != null && block == null)
        {
            target.Data.Clear();
            ReadTensorFloats(rawData, stream, info.Dtype, info.Shape, target.Data);
            return;
        }

        // Block tensors dequantize into a pooled float buffer, then land in the target
        // with whatever transposition the tensor needs (GGUF row-major semantics).
        float[] buffer = MemoryHelpers.RentArray<float>(count);
        try
        {
            ReadTensorFloats(rawData, stream, info.Dtype, info.Shape, buffer.AsSpan(0, count));

            var floatTarget = blockFloatTarget;
            if (floatTarget != null)
            {
                // LFM2 short-conv kernel is a ggml {l_cache, hidden} tensor, which
                // stores its raw floats channel-major ([channel][tap] at
                // channel*l_cache + tap). The conv kernel must be read as
                // [tap][channel] (ApplyConv indexes pKernel[tap*hidden + channel]),
                // so transpose the block-major data into target's row-major layout.
                if (info.Name.Contains("shortconv.conv.weight", StringComparison.OrdinalIgnoreCase))
                {
                    floatTarget.Data.Clear();
                    int taps = info.Shape[0];
                    int chan = info.Shape[1];
                    for (int c = 0; c < chan; c++)
                        for (int k = 0; k < taps; k++)
                            floatTarget.Data[k * chan + c] = buffer[c * taps + k];
                }
                else if (info.Shape.Length == 2)
                {
                    int ggufIn = info.Shape[0];
                    int ggufOut = info.Shape[1];
                    bool isFfnUp = info.Name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) &&
                                   floatTarget.Shape[1] == 2 * ggufOut;
                    int colOff = isFfnUp ? ggufOut : 0;
                    if (colOff == 0) floatTarget.Data.Clear();
                    int targetOut = floatTarget.Shape[1];
                    for (int i = 0; i < ggufIn; i++)
                        for (int o = 0; o < ggufOut; o++)
                            floatTarget.Data[i * targetOut + colOff + o] = buffer[o * ggufIn + i];
                }
                else
                {
                    floatTarget.Data.Clear();
                    buffer.AsSpan(0, count).CopyTo(floatTarget.Data);
                }
            }
        }
        finally
        {
            MemoryHelpers.ReturnArray(buffer);
        }
    }

    /// <summary>Dequantizes the output head from the raw bytes kept at load (see <see cref="TransformerWeights.LmHeadWeight"/>).</summary>
    private Tensor<float> DequantizeLmHead(TransformerWeights weights, int rows, int cols, int[] fileShape)
    {
        byte[] raw = weights.RawLmHead ?? throw new InvalidOperationException("The output head has no raw bytes to dequantize.");
        var head = new Tensor<float>(rows, cols);
        head.Data.Clear();
        using var reader = new BinaryReader(new MemoryStream(raw));
        ReadTensorInto(reader, weights.RawLmHeadDtype!.Value, fileShape, head.Data);
        return head;
    }

    /// <summary>
    /// Dequantizes the given shape into <paramref name="destination"/> from the tensor's raw
    /// bytes when they are already loaded, or straight from the file stream for float-only
    /// block tensors (bias, norm, router, shortconv kernel) that never store a raw copy.
    /// </summary>
    private void ReadTensorFloats(byte[]? rawBytes, Stream stream, QuantDType dtype, int[] shape, Span<float> destination)
    {
        if (rawBytes is not null)
        {
            using var ms = new MemoryStream(rawBytes, writable: false);
            using var reader = new BinaryReader(ms);
            ReadTensorInto(reader, dtype, shape, destination);
            return;
        }
        using var fileReader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        ReadTensorInto(fileReader, dtype, shape, destination);
    }

    private void ReadTensorInto(BinaryReader stream, QuantDType dtype, int[] shape, Span<float> destination)
    {
        int count = TensorLoadHelper.ComputeElementCountChecked(shape);
        if (destination.Length < count)
            throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");
        _qOps.ReadFor(dtype, stream, destination, count);
    }
}