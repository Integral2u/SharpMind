using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text.RegularExpressions;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;
public static partial class GgufLoader
{
    private const uint Magic = 0x46554747;

    public readonly struct KvPair { public required string Key { get; init; } public required object Value { get; init; } }
    public readonly struct TensorInfo { public required string Name { get; init; } public required GgufDtype Dtype { get; init; } public required int[] Shape { get; init; } public required long Offset { get; init; } }

    private static object ReadValue(BinaryReader reader, uint valType)
    {
        return valType switch
        {
            0 => reader.ReadByte(),
            1 => reader.ReadSByte(),
            2 => reader.ReadUInt16(),
            3 => reader.ReadInt16(),
            4 => reader.ReadUInt32(),
            5 => reader.ReadInt32(),
            6 => reader.ReadSingle(),
            7 => reader.ReadBoolean(),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new InvalidDataException("Unknown scalar type: " + valType)
        };
    }

    private static (ulong len, string str) ReadString(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        if (len > 10000) return (len, "");
        var bytes = reader.ReadBytes((int)len);
        return (len, System.Text.Encoding.UTF8.GetString(bytes));
    }

    private static string ReadStringValue(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        var bytes = reader.ReadBytes((int)len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    /// <summary>
    /// Reads a GGUF array KV value and returns it as a typed array.
    /// Supported element types: string[], float[], int[], uint[].
    /// Returns null for element types not needed for model/tokenizer metadata.
    /// Previously this was SkipArrayValue which silently discarded all tokenizer vocab data.
    /// </summary>
    private static object? ReadArrayValue(BinaryReader reader)
    {
        var elemType = reader.ReadUInt32();
        var arrLen = reader.ReadUInt64();
        int len = (int)arrLen;

        switch (elemType)
        {
            case 8: // String
                {
                    var arr = new string[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = ReadStringValue(reader);
                    return arr;
                }
            case 6: // Float32
                {
                    var arr = new float[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = reader.ReadSingle();
                    return arr;
                }
            case 5: // Int32
                {
                    var arr = new int[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = reader.ReadInt32();
                    return arr;
                }
            case 4: // UInt32
                {
                    var arr = new uint[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = reader.ReadUInt32();
                    return arr;
                }
            case 11: // Int64
                {
                    var arr = new long[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = reader.ReadInt64();
                    return arr;
                }
            default:
                {
                    // Skip element types we don't need (bool, int8, int16, uint16, uint64, float64)
                    int elemSize = elemType switch
                    {
                        0 => 1,
                        1 => 1,
                        2 => 2,
                        3 => 2,
                        7 => 1,
                        10 => 8,
                        12 => 8,
                        _ => 4
                    };
                    reader.BaseStream.Position += (long)len * elemSize;
                    return null;
                }
        }
    }

    public static string[]? GetStringArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as string[];

    public static float[]? GetFloatArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as float[];

    public static int[]? GetIntArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as int[];

    public static ModelMetaData LoadMeta(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var meta = new ModelMetaData();

        uint magic = reader.ReadUInt32();
        if (magic != Magic) throw new InvalidDataException("Not GGUF: " + magic.ToString("X8"));

        meta.Version = reader.ReadUInt32();
        meta.TensorCount = reader.ReadInt64();
        meta.KvCount = reader.ReadInt64();

        for (int i = 0; i < meta.KvCount; i++)
        {
            var (keyLen, key) = ReadString(reader);
            uint valType = reader.ReadUInt32();
            object? val = valType switch
            {
                // STRING
                8 => ReadStringValue(reader),
                // ARRAY � read all array types; tokenizer vocab lives here
                9 => ReadArrayValue(reader),
                _ => ReadValue(reader, valType),
            };
            if (val != null)
                meta.KvPairs.Add(new KvPair { Key = key, Value = val });
        }

        for (int i = 0; i < meta.TensorCount; i++)
        {
            try
            {
                var (nameLen, name) = ReadString(reader);
                if (nameLen == 0 || nameLen > 500) break;

                var nDims = reader.ReadUInt32();
                if (nDims > 10) break;

                var shape = new int[nDims];
                for (int j = 0; j < nDims; j++) shape[j] = (int)reader.ReadUInt64();

                var dtype = (GgufDtype)reader.ReadUInt32();
                var offset = reader.ReadUInt64();

                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            }
            catch { break; }
        }

        // Compute the data section start offset (aligned to 32 bytes by default).
        // GGUF stores tensor offsets relative to this position, but we need absolute
        // file offsets when seeking to each tensor's data.
        uint alignment = (uint)meta.GetLong("general.alignment", 32);
        long pos = stream.Position;
        meta.DataOffset = (pos + alignment - 1) & ~(alignment - 1);

        return meta;
    }

    public static ModelConfig? LoadConfig(ModelMetaData meta)
    {
        string arch = meta.GetString("general.architecture");
        if (string.IsNullOrWhiteSpace(arch)) return null;

        // Derive vocab + hidden from the embedding tensor shape � most reliable source.
        int vocabSize = 32000, hiddenDim = 1536, numLayers = 28;
        int numHeads = 12, numKvHeads = 12, ffnDim = 6144, maxSeqLen = 2048;

        var embdInfo = meta.Tensors.FirstOrDefault(
            t => t.Name.Contains("token_embd") && t.Name.Contains("weight"));

        if (embdInfo.Shape is { Length: >= 2 })
        {
            long d0 = embdInfo.Shape[0], d1 = embdInfo.Shape[1];
            if (d0 > d1) { vocabSize = (int)d0; hiddenDim = (int)d1; }
            else { vocabSize = (int)d1; hiddenDim = (int)d0; }
        }

        // Override with explicit meta keys where present � they are authoritative.
        hiddenDim = (int)meta.GetLong($"{arch}.embedding_length", hiddenDim);
        ffnDim = (int)meta.GetLong($"{arch}.feed_forward_length", ffnDim);
        maxSeqLen = (int)meta.GetLong($"{arch}.context_length", maxSeqLen);
        numHeads = (int)meta.GetLong($"{arch}.attention.head_count", numHeads);
        // head_count_kv may be absent for MHA models (Qwen3); default to numHeads.
        numKvHeads = (int)meta.GetLong($"{arch}.attention.head_count_kv", -1);
        if (numKvHeads <= 0) numKvHeads = numHeads;

        // Some architectures (Qwen3) specify explicit key/value head dimensions
        // that may differ from HiddenDim / NumHeads.
        long rawKeyLen = meta.GetLong($"{arch}.attention.key_length", -1);
        int? keyLength = rawKeyLen > 0 ? (int)rawKeyLen : null;
        long rawValLen = meta.GetLong($"{arch}.attention.value_length", -1);
        int? valueLength = rawValLen > 0 ? (int)rawValLen : null;

        // BUG FIX: numLayers was hardcoded to 24, never read from meta.
        // block_count is the GGUF key that holds the layer count for all architectures.
        numLayers = (int)meta.GetLong($"{arch}.block_count", numLayers);

        // RoPE frequency base — crucial for correct positional encoding.
        // Qwen2 uses 10_000; Qwen2.5 uses 1_000_000; LLaMA 3 uses 500_000.
        float ropeTheta = meta.GetFloat($"{arch}.rope.freq_base",
                          meta.GetFloat("rope_theta",
                          meta.GetFloat("rope.freq_base", 10_000f)));

        // Prefer the explicit vocab_size key over the embedding tensor dimension,
        // as some models (e.g. LLaMA-3) pad the embedding table to a round number.
        int metaVocab = (int)meta.GetLong($"{arch}.vocab_size",
                         meta.GetLong("tokenizer.ggml.token_count",
                         meta.GetLong("vocab_size", vocabSize)));
        if (metaVocab > 0) vocabSize = metaVocab;

        // Explicit head_dim override — some models (Gemma 2, DeepSeek V2)
        // specify head_dim directly rather than deriving from HiddenDim / NumHeads.
        long rawHeadDim = meta.GetLong($"{arch}.head_dim", -1);
        int? headDimOverride = rawHeadDim > 0 ? (int)rawHeadDim : null;

        // RoPE dimension count — if specified and less than HeadDim,
        // only the first N dimensions get rotary encoding.
        long rawRopeDim = meta.GetLong($"{arch}.rope.dimension_count", -1);
        int? ropeDim = rawRopeDim > 0 ? (int)rawRopeDim : null;

        // RoPE scaling for extended context
        string? ropeScalingType = meta.GetString($"{arch}.rope.scaling.type");
        float ropeFactor = meta.GetFloat($"{arch}.rope.scaling.factor", float.NaN);
        float? ropeScalingFactor = float.IsNaN(ropeFactor) ? null : ropeFactor;
        long rawRopeOrigCtx = meta.GetLong($"{arch}.rope.scaling.original_context_length", -1);
        int? ropeOriginalContextLength = rawRopeOrigCtx > 0 ? (int)rawRopeOrigCtx : null;

        // Tie word embeddings — whether LM head shares weights with token embeddings
        long rawTie = meta.GetLong($"{arch}.tie_word_embeddings", -1);
        bool? tieWordEmbeddings = rawTie >= 0 ? (rawTie != 0) : null;

        // Norm type: 0 = LayerNorm, 1 = RMSNorm
        long rawNormType = meta.GetLong($"{arch}.norm_type", -1);
        int? normTypeOverride = rawNormType >= 0 ? (int)rawNormType : null;

        // MoE expert counts
        long rawExpertCount = meta.GetLong($"{arch}.expert_count", -1);
        int expertCount = rawExpertCount > 0 ? (int)rawExpertCount : 8;
        long rawTopK = meta.GetLong($"{arch}.expert_used_count", -1);
        int topKExperts = rawTopK > 0 ? (int)rawTopK : 2;

        return new ModelConfig
        {
            Architecture = arch,
            VocabSize = vocabSize,
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            FfnDim = ffnDim,
            MaxSeqLen = maxSeqLen,
            RopeTheta = ropeTheta,
            NormEps = meta.GetFloat($"{arch}.attention.layer_norm_rms_epsilon",
                      meta.GetFloat("rms_norm_eps", 1e-5f)),
            KeyLength = keyLength,
            ValueLength = valueLength,
            HeadDimOverride = headDimOverride,
            RopeDim = ropeDim,
            RopeScalingType = ropeScalingType,
            RopeScalingFactor = ropeScalingFactor,
            RopeOriginalContextLength = ropeOriginalContextLength,
            TieWordEmbeddings = tieWordEmbeddings,
            NormTypeOverride = normTypeOverride,
            NumExperts = expertCount,
            TopKExperts = topKExperts,
        };
    }

    /// <summary>
    /// Builds a <see cref="Tokenizer"/> directly from the GGUF metadata.
    /// This is the preferred path: the vocab stored in GGUF always matches the
    /// model weights, so it can never produce the OOB token-ID crashes that
    /// occur when an external tokenizer.json has the wrong vocab size.
    /// </summary>
    public static Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta)
    {
        var tokens = GetStringArray(meta, "tokenizer.ggml.tokens");
        if (tokens == null || tokens.Length == 0) return null;

        var types = GetIntArray(meta, "tokenizer.ggml.token_type");
        var merges = GetStringArray(meta, "tokenizer.ggml.merges");

        int bosId = (int)meta.GetLong("tokenizer.ggml.bos_token_id", 1);
        int eosId = (int)meta.GetLong("tokenizer.ggml.eos_token_id", 2);

        try
        {
            return Tokenizer.FromGguf(tokens, merges, types, bosId, eosId);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Some GGUF files drop HuggingFace "added_tokens" during conversion
    /// (common with SentencePiece models like TinyLlama). These tokens are
    /// referenced in the chat template but aren't in the tokenizer vocab.
    ///
    /// We detect this: extract all {@code <...>} patterns from the template,
    /// check each against the tokenizer, and inject missing ones as new
    /// additional special tokens. The embedding/LM head get extended rows
    /// (initialized to mean/zero) when weights are loaded.
    ///
    /// This is fully dynamic — no hardcoded model names or token strings.
    /// </summary>
    private static void InjectMissingTemplateTokens(
        ModelMetaData meta, ref ModelConfig config, Tokenizer tokenizer)
    {
        string? template = meta.GetChatTemplate();
        if (string.IsNullOrEmpty(template)) return;

        // Extract <...> patterns from the template
        var candidates = new HashSet<string>();
        foreach (Match m in RegexGenerated.ChatTemplateRegex.Matches(template))// Regex.Matches(template, @"<[^>]+>"))
            candidates.Add(m.Value);

        if (candidates.Count == 0) return;

        var toAdd = new List<string>();
        foreach (string token in candidates)
        {
            if (!tokenizer.Vocab.Contains(token))
                toAdd.Add(token);
        }

        if (toAdd.Count == 0) return;

        foreach (string token in toAdd)
            tokenizer.AddAdditionalToken(token);

        config = config with { VocabSize = config.VocabSize + toAdd.Count };
    }


    /// <summary>
    /// Single-pass loader: extracts metadata, config, and tokenizer from GGUF.
    /// Tokenizer is built from GGUF vocab first (guaranteed to match weights).
    /// Falls back to <paramref name="tokenizerPath"/> only when the GGUF contains
    /// no vocab data. The path fallback is model-agnostic: caller should handle
    /// non-Qwen formats after this returns.
    /// </summary>
    public static void Load(
        string ggufPath,
        string? tokenizerPath,
        out ModelMetaData meta,
        out ModelConfig config,
        out Tokenizer? tokenizer)
    {
        meta = LoadMeta(ggufPath);
        config = LoadConfig(meta)!;

        // Prefer GGUF-embedded tokenizer ? its vocab size is guaranteed to match the weights.
        tokenizer = LoadTokenizerFromMeta(meta);

        // Fall back to file only when GGUF has no vocab data.
        if (tokenizer == null && !string.IsNullOrEmpty(tokenizerPath) && File.Exists(tokenizerPath))
        {
            try
            {
                // Detect model architecture and use the appropriate factory.
                string arch = meta.GetString("general.architecture") ?? "";
                string tokModel = meta.GetString("tokenizer.ggml.model") ?? "";

                if (arch.Contains("qwen", StringComparison.OrdinalIgnoreCase) || tokModel.Contains("qwen", StringComparison.OrdinalIgnoreCase))
                {
                    tokenizer = Tokenizer.FromQwen(tokenizerPath);
                }
                else if (arch.Contains("llama", StringComparison.OrdinalIgnoreCase))
                {
                    tokenizer = Tokenizer.FromLlama(tokenizerPath);
                }
                else if (arch.Contains("mistral", StringComparison.OrdinalIgnoreCase))
                {
                    tokenizer = Tokenizer.FromMistral(tokenizerPath);
                }
                else
                {
                    tokenizer = Tokenizer.FromFile(tokenizerPath);   // generic BPE/SentencePiece fallback
                }
            }
            catch
            {
                tokenizer = null;
            }
        }

        // Detect and inject special tokens missing from the GGUF vocab
        // (e.g. TinyLlama's <|user|>, <|assistant|> tokens dropped during conversion).
        // This runs regardless of whether the tokenizer came from GGUF or a file.
        if (tokenizer != null)
            InjectMissingTemplateTokens(meta, ref config, tokenizer);
    }

    /// <summary>Seeks to and loads raw quantized data for a single layer's tensors.
    /// Requires <see cref="TransformerWeights.GgufMeta"/> and <see cref="TransformerWeights.GgufPath"/>
    /// populated by <see cref="LoadWeightsToTransformerWeights"/> (any mode).
    /// When <paramref name="loadFloat"/> is true, non-quantized float tensors (norms, biases) are also read.</summary>
    public static void LoadLayerRawWeights(TransformerWeights weights, int layerIndex, bool loadFloat = false)
    {
        var meta = weights.GgufMeta ?? throw new InvalidOperationException("GgufMeta not set. Load weights first.");
        var path = weights.GgufPath ?? throw new InvalidOperationException("GgufPath not set. Load weights first.");
        string prefix = $"blk.{layerIndex}.";

        var layerTensors = meta.Tensors
            .Where(t => t.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (layerTensors.Count == 0) return;

        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var block = weights.Blocks[layerIndex];

        foreach (var info in layerTensors)
        {
            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;
            stream.Position = targetOffset;

            var (target, _, rawField) = weights.ResolveTarget(info.Name);
            long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);

            if (rawSize > 0 && IsQuantizedType(info.Dtype) && rawField != null)
            {
                byte[] rawData = new byte[rawSize];
                stream.ReadExactly(rawData);
                SetRawField(block, rawField, rawData, info.Dtype);
            }
            else if (loadFloat && target != null)
            {
                var floatTarget = weights.ResolveFloatTarget(info.Name);
                if (floatTarget != null)
                {
                    floatTarget.Data.Clear();
                    ReadTensorInto(reader, info.Dtype, info.Shape, floatTarget.Data);
                }
            }
        }
    }

    public static TransformerWeights LoadWeightsToTransformerWeights(string path, ModelConfig config, IProgress<float>? progress = null, LoadMode mode = LoadMode.Realtime)
    {
        // Clear pool to prevent cross-model memory accumulation
        // (different model sizes rarely reuse the same buckets)
        SharpMind.Core.Memory.NativeBufferPool<float>.Clear();

        var meta = LoadMeta(path);
        var weights = ModelFactory.CreateWeights(config, config.ForModel(HardwareTier.Auto)); // Use model-specific config for allocation shapes
        weights.GgufMeta = meta;
        weights.GgufPath = path;

        // Detect MoE: presence of .exps. expert tensors in GGUF metadata
        weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));

        // Cached mode: CachedWeightLoader handles block weight matrix raw data
        // on demand. We still run the tensor loop for non-weight tensors
        // (norms, biases, embedding, lm_head).
        bool isCached = mode == LoadMode.Cached;

        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        int loaded = 0, total = meta.Tensors.Count;

        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)loaded / total);
            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;
            stream.Position = targetOffset;

            // Create LM head tensor on demand when output.weight is found
            // Must NOT match attn_output.weight in block tensors
            // GGUF stores shape as [hidden_dim, vocab_size] but SharpMind expects
            // [vocab_size, hidden_dim] for MatMulWithBT convention.
            if (!info.Name.Contains("blk.") && info.Name.Contains("output.weight") && weights.LmHeadWeight == null)
            {
                long ggufIn = info.Shape[0];
                weights.SetLmHead(new Tensor<float>((int)config.VocabSize, (int)ggufIn));
            }

            // Identify the target tensor and whether it's a raw weight
            var (target, block, rawField) = weights.ResolveTarget(info.Name);

            if (target == null && block == null)
            {
                loaded++;
                continue;
            }

            int count = 1;
            foreach (int d in info.Shape) count *= d;

            if (IsQuantizedType(info.Dtype) && info.Shape.Length >= 2 && block != null && rawField != null)
            {
                // Cached mode: CachedWeightLoader handles block weight matrix raw data on demand
                if (isCached)
                {
                    loaded++;
                    continue;
                }

                long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0 && stream.Position + rawSize <= stream.Length)
                {
                    byte[] rawData = new byte[rawSize];
                    stream.ReadExactly(rawData);
                    stream.Position -= rawSize; // seek back for dequant read if needed

                    // Store raw data in BlockWeights
                    SetRawField(block, rawField, rawData, info.Dtype);
                    loaded++;
                    if (mode != LoadMode.Full)
                        continue;
                }
            }

            // Capture raw quantized data for non-block tensors (embedding/lm_head)
            // These are stored alongside the float dequantized copy.
            // Both embedding and LM head raw data are padded to config.VocabSize
            // so the quantized matmul never reads past the buffer.
            if (IsQuantizedType(info.Dtype) && info.Shape.Length >= 2 && target != null && block == null)
            {
                long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0 && stream.Position + rawSize <= stream.Length)
                {
                    byte[] rawData;
                    // Determine the vocab dimension from the tensor shape.
                    // GGUF stores embedding as [vocab, hidden] and output.weight as [hidden, vocab].
                    // The larger dimension is the vocab count.
                    long tensorVocab = Math.Max(info.Shape[0], info.Shape[1]);
                    long paddedVocab = config.VocabSize;
                    if (paddedVocab > tensorVocab)
                    {
                        long colBytes = rawSize / tensorVocab;
                        rawData = new byte[paddedVocab * colBytes];
                        for (long r = 0; r < tensorVocab; r++)
                            stream.ReadExactly(rawData, (int)(r * colBytes), (int)colBytes);
                        stream.Position -= rawSize;
                    }
                    else
                    {
                        rawData = new byte[rawSize];
                        stream.ReadExactly(rawData);
                        stream.Position -= rawSize;
                    }
                    // Q8_0/Q5_0/Q6_K LM head raw data has a layout mismatch:
                    // GGUF stores output.weight as [hiddenDim, vocabSize] row-major with
                    // Q8_0 blocks along vocabSize, but QuantizedMatMulQ8_0 expects
                    // [vocabSize, hiddenDim] with blocks along hiddenDim. The block stride
                    // differs for non-square matrices, so skip raw storage and use the
                    // float dequantize+matmul path instead (always correct for LM head).
                    bool isBadLayout = info.Dtype is GgufDtype.Q8_0 or GgufDtype.Q5_0
                        or GgufDtype.Q6_K or GgufDtype.Q6_K_S;
                    if (target == weights.LmHeadWeight && !isBadLayout)
                    {
                        weights.RawLmHead = rawData;
                        weights.RawLmHeadDtype = info.Dtype;
                    }
                    else if (target != weights.LmHeadWeight)
                    {
                        weights.RawEmbedding = rawData;
                        weights.RawEmbeddingDtype = info.Dtype;
                    }
                }
            }

            float[] buffer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));
                if (target != null)
                {
                    target.Data.Clear();
                    // LM head requires transpose: GGUF stores [hidden_dim, vocab_size]
                    // but SharpMind expects [vocab_size, hidden_dim].
                    if (target == weights.LmHeadWeight && info.Shape.Length == 2)
                    {
                        int ggufIn = (int)info.Shape[0], ggufOut = (int)info.Shape[1];
                        for (int i = 0; i < ggufIn; i++)
                            for (int j = 0; j < ggufOut; j++)
                                target.Data[j * ggufIn + i] = buffer[i * ggufOut + j];
                    }
                    else
                    {
                        buffer.AsSpan(0, count).CopyTo(target.Data);
                    }
                }
                else if (block != null)
                {
                    var floatTarget = weights.ResolveFloatTarget(info.Name);
                    if (floatTarget != null)
                    {
                        if (info.Shape.Length == 2)
                        {
                            int ggufIn = info.Shape[0];
                            int ggufOut = info.Shape[1];
                            // ffn_gate and ffn_up both map to Wf1 [HiddenDim, 2*FfnDim].
                            // Gate fills columns [0, FfnDim), up fills [FfnDim, 2*FfnDim).
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
                loaded++;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
        progress?.Report(1f);

        if (isCached)
            weights.CachedLoader = new Layers.CachedWeightLoader(weights, path, meta);

        return weights;
    }

    public static Dictionary<string, Tensor<float>> LoadWeights(string path, IProgress<float>? progress = null)

    {
        var meta = LoadMeta(path);
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var result = new Dictionary<string, Tensor<float>>();
        int total = meta.Tensors.Count;

        int idx = 0;
        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)idx / total);
            stream.Position = meta.DataOffset + info.Offset;
            var tensor = ReadTensor(reader, info.Dtype, info.Shape);
            result[info.Name] = tensor;
            idx++;
        }
        progress?.Report(1f);
        return result;
    }

    public static void LoadWeightsToModel(string path, ModelMetaData meta, Transformer model, IProgress<float>? progress = null)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        int loaded = 0, missing = 0, total = meta.Tensors.Count;

        // Quiet mode: no verbose FFN tensor dump

        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)loaded / total);

            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;

            stream.Position = targetOffset;

            int count = 1;
            foreach (int d in info.Shape) count *= d;

            // Read raw quantized bytes before dequantizing (save stream position)
            long savedPos = stream.Position;
            bool skipDequant = false;
            if (IsQuantizedType(info.Dtype) && info.Shape.Length >= 2)
            {
                long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0 && savedPos + rawSize <= stream.Length)
                {
                    byte[] rawData = new byte[rawSize];
                    stream.ReadExactly(rawData);
                    stream.Position = savedPos; // seek back for dequant read
                    skipDequant = model.SetRawWeight(info.Name, rawData, info.Dtype);
                }
            }

            if (skipDequant)
            {
                loaded++;
            }
            else
            {
                float[] buffer = ArrayPool<float>.Shared.Rent(count);
                try
                {
                    ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));

                    if (model.LoadWeight(info.Name, buffer.AsSpan(0, count)))
                        loaded++;
                    else
                        missing++;
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(buffer);
                }
            }
        }

        progress?.Report(1f);
    }

    internal static bool IsQuantizedType(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q2_K or GgufDtype.Q3_K or GgufDtype.Q4_K or GgufDtype.Q5_K or GgufDtype.Q6_K
        or GgufDtype.Q2_K_S or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L
        or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M or GgufDtype.Q6_K_S
        or GgufDtype.Q4_0 or GgufDtype.Q4_1 or GgufDtype.Q5_0 or GgufDtype.Q5_1
        or GgufDtype.Q8_0 or GgufDtype.Q8_1 or GgufDtype.Q8_K
        or GgufDtype.IQ4_NL => true,
        _ => false
    };

    internal static long GetRawTensorByteCount(int[] shape, GgufDtype dtype)
    {
        int blockSize, bytesPerBlock;

        switch (dtype)
        {
            case GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L: blockSize = 256; bytesPerBlock = 110; break;  // d[2]+hmask[32]+qs[64]+scales[12]
            case GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M: blockSize = 256; bytesPerBlock = 144; break;  // d[2]+dmin[2]+scales[12]+qs[128]
            case GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M: blockSize = 256; bytesPerBlock = 176; break;  // d[2]+dmin[2]+scales[12]+qh[32]+qs[128]
            case GgufDtype.Q6_K or GgufDtype.Q6_K_S: blockSize = 256; bytesPerBlock = 210; break;  // ql[128]+qh[64]+scales[16]+d[2]
            case GgufDtype.Q2_K or GgufDtype.Q2_K_S: blockSize = 256; bytesPerBlock = 84; break;  // scales[16]+qs[64]+d[2]+dmin[2]
            case GgufDtype.Q8_K: blockSize = 256; bytesPerBlock = 292; break;  // d[4]+qs[256]+bsums[32]
            case GgufDtype.Q8_0: blockSize = 32; bytesPerBlock = 34; break;
            case GgufDtype.Q8_1: blockSize = 32; bytesPerBlock = 36; break;  // d[2]+s[2]+qs[32]
            case GgufDtype.Q5_0: blockSize = 32; bytesPerBlock = 22; break;
            case GgufDtype.Q5_1: blockSize = 32; bytesPerBlock = 24; break;  // d[2]+m[2]+qh[4]+qs[16]
            case GgufDtype.Q4_0: blockSize = 32; bytesPerBlock = 18; break;
            case GgufDtype.IQ4_NL: blockSize = 32; bytesPerBlock = 18; break;  // d[2]+qs[16]
            case GgufDtype.Q4_1: blockSize = 32; bytesPerBlock = 20; break;  // d[2]+m[2]+qs[16]
            default: return 0;
        }
        // The GGUF file stores quantized blocks in a single contiguous stream:
        // total blocks = ceil(total_elements / blockSize). Blocks are NOT
        // padded to column boundaries (no ceil(InF/blockSize) * OutF).
        long totalElements = 1;
        foreach (int d in shape) totalElements *= d;
        long nBlocks = (totalElements + blockSize - 1) / blockSize;
        return nBlocks * bytesPerBlock;
    }

    private static Tensor<float> ReadTensor(BinaryReader stream, GgufDtype dtype, int[] shape)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        var result = new Tensor<float>(shape);
        ReadTensorInto(stream, dtype, shape, result.Data);
        return result;
    }

    internal static void ReadQBlockRow(BinaryReader stream, GgufDtype dtype, Span<float> dest, int count)
    {
        switch (dtype)
        {
            case GgufDtype.Q4_0: ReadQ4_0(stream, dest, count); break;
            case GgufDtype.IQ4_NL: ReadIQ4_NL(stream, dest, count); break;
            case GgufDtype.Q4_1: ReadQ4_1(stream, dest, count); break;
            case GgufDtype.Q5_0: ReadQ5_0(stream, dest, count); break;
            case GgufDtype.Q5_1: ReadQ5_1(stream, dest, count); break;
            case GgufDtype.Q8_0: ReadQ8_0(stream, dest, count); break;
            case GgufDtype.Q8_1: ReadQ8_1(stream, dest, count); break;
            case GgufDtype.Q2_K or GgufDtype.Q2_K_S: ReadQ2K(stream, dest, count); break;
            case GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L: ReadQ3_K(stream, dest, count); break;
            case GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M: ReadQ4K(stream, dest, count); break;
            case GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M: ReadQ5_K(stream, dest, count); break;
            case GgufDtype.Q6_K or GgufDtype.Q6_K_S: ReadQ6K(stream, dest, count); break;
            case GgufDtype.Q8_K: ReadQ8K(stream, dest, count); break;
        }
    }

    private static bool IsKQuant(GgufDtype dtype) => dtype is
        GgufDtype.Q2_K or GgufDtype.Q2_K_S
        or GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L
        or GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M
        or GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M
        or GgufDtype.Q6_K or GgufDtype.Q6_K_S
        or GgufDtype.Q8_K;

    internal static void ReadTensorInto(BinaryReader stream, GgufDtype dtype, int[] shape, Span<float> destination)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        if (destination.Length < count)
            throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");

        switch (dtype)
        {
            case GgufDtype.F32:
                for (int i = 0; i < count; i++) destination[i] = stream.ReadSingle();
                break;
            case GgufDtype.F16:
                for (int i = 0; i < count; i++) destination[i] = HalfToFloat(stream.ReadUInt16());
                break;
            default:
                // GGUF stores quantized blocks in flat (element-contiguous) layout:
                // blocks cover the linear element array sequentially and may span
                // column boundaries when shape[0] % blockSize != 0.
                // Read all blocks sequentially — the destination is already in
                // flat row-major order matching the GGUF element ordering.
                ReadQBlockRow(stream, dtype, destination, count);
                break;
        }
    }

    public static float HalfToFloat(ushort half)
    {
        int sign = (half >> 15) & 0x1;
        int exp = (half >> 10) & 0x1F;
        int mant = half & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            float val = mant / 1024f;
            return (sign == 0 ? 1f : -1f) * val * MathF.Pow(2f, -14f);
        }
        if (exp == 31)
        {
            if (mant == 0)
                return (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity);
            return float.NaN;
        }

        return (sign == 0 ? 1f : -1f) * MathF.Pow(2f, exp - 15) * (1f + mant / 1024f);
    }

    public static unsafe void ReadQ8_0(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 34;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            fixed (byte* pBuf = buf)
            {
                sbyte* values = (sbyte*)(pBuf + 2);
                if (Avx2.IsSupported)
                {
                    var vd = Vector256.Create(d);
                    int j = 0;
                    for (; j <= valid - 8; j += 8)
                    {
                        var vi = Avx2.ConvertToVector256Int32(values + j);
                        var vf = Avx.ConvertToVector256Single(vi);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])), Avx.Multiply(vf, vd));
                    }
                    for (; j < valid; j++)
                        data[blockStart + j] = values[j] * d;
                }
                else
                {
                    for (int j = 0; j < valid; j++)
                        data[blockStart + j] = values[j] * d;
                }
            }
        }
    }

    public static unsafe float[] ReadBlock(byte* block, string dtype, int blkSize)
    {
        float[] result = new float[blkSize];
        switch (dtype)
        {
            case "Q5_0":
            {
                float d = HalfToFloat(*(ushort*)block);
                uint qh = *(uint*)(block + 2);
                byte* qs = block + 6;
                for (int j = 0; j < blkSize; j++)
                {
                    int nibble = (qs[j / 2] >> ((j & 1) * 4)) & 0x0F;
                    int highBit = (int)(qh >> j) & 1;
                    result[j] = d * ((nibble | (highBit << 4)) - 16);
                }
                break;
            }
            case "Q6_K":
            {
                byte* ql = block;
                byte* qh = ql + 128;
                sbyte* scales = (sbyte*)(qh + 64);
                float d = HalfToFloat(*(ushort*)(block + 208));
                for (int nOff = 0; nOff < blkSize; nOff += 128)
                {
                    int qlOff = nOff == 0 ? 0 : 64;
                    int qhOff = nOff == 0 ? 0 : 32;
                    int scOff = nOff == 0 ? 0 : 8;
                    int halfRem = Math.Min(128, blkSize - nOff);
                    for (int l = 0; l < 32 && l < halfRem; l++)
                    {
                        int is_ = l / 16;
                        int q1 = (ql[qlOff + l] & 0x0F) | ((qh[qhOff + l] & 0x03) << 4);
                        int q2 = (ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 0x03) << 4);
                        int q3 = ((ql[qlOff + l] >> 4) & 0x0F) | (((qh[qhOff + l] >> 4) & 0x03) << 4);
                        int q4 = ((ql[qlOff + l + 32] >> 4) & 0x0F) | (((qh[qhOff + l] >> 6) & 0x03) << 4);
                        int idx1 = nOff + l;
                        int idx2 = nOff + l + 32;
                        int idx3 = nOff + l + 64;
                        int idx4 = nOff + l + 96;
                        if (idx1 < blkSize) result[idx1] = d * scales[scOff + is_ + 0] * (q1 - 32);
                        if (idx2 < blkSize) result[idx2] = d * scales[scOff + is_ + 2] * (q2 - 32);
                        if (idx3 < blkSize) result[idx3] = d * scales[scOff + is_ + 4] * (q3 - 32);
                        if (idx4 < blkSize) result[idx4] = d * scales[scOff + is_ + 6] * (q4 - 32);
                    }
                }
                break;
            }
            case "Q8_0":
            {
                float d = HalfToFloat(*(ushort*)block);
                sbyte* qs = (sbyte*)(block + 2);
                for (int j = 0; j < blkSize; j++)
                    result[j] = qs[j] * d;
                break;
            }
            default:
                throw new ArgumentException($"Unsupported dtype: {dtype}");
        }
        return result;
    }

    internal static unsafe void ReadQ4_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 20;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float m = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            int valid = Math.Min(qk, n - blockStart);

            if (Avx2.IsSupported)
            {
                fixed (byte* pBuf = buf)
                {
                    byte* nibbles = pBuf + 4;
                    var v16 = Unsafe.ReadUnaligned<Vector128<byte>>(nibbles);

                    var lo = Sse2.And(v16, Vector128.Create((byte)0x0F));
                    var hi = Sse2.And(
                        Sse2.ShiftRightLogical(
                            Sse2.And(v16, Vector128.Create((byte)0xF0)).AsUInt16(), 4).AsByte(),
                        Vector128.Create((byte)0x0F));

                    var iLow = Sse2.UnpackLow(lo, hi);
                    var iHigh = Sse2.UnpackHigh(lo, hi);

                    var up0 = Avx2.ConvertToVector256Int32(iLow);
                    var up1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iLow, 8));
                    var up2 = Avx2.ConvertToVector256Int32(iHigh);
                    var up3 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iHigh, 8));

                    var d256 = Vector256.Create(d);
                    var m256 = Vector256.Create(m);

                    int j = 0;
                    for (; j + 8 <= valid; j += 8)
                    {
                        var vv = j switch { 0 => up0, 8 => up1, 16 => up2, _ => up3 };
                        var vf = Avx.ConvertToVector256Single(vv);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])),
                            Avx.Add(Avx.Multiply(vf, d256), m256));
                    }
                    for (; j < valid; j++)
                    {
                        int q = (nibbles[j / 2] >> ((j & 1) * 4)) & 0x0F;
                        data[blockStart + j] = q * d + m;
                    }
                }
            }
            else
            {
                for (int j = 0; j < valid; j++)
                {
                    int q = (buf[4 + j / 2] >> ((j & 1) * 4)) & 0x0F;
                    data[blockStart + j] = q * d + m;
                }
            }
        }
    }

    internal static unsafe void ReadQ5_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 24;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float m = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            uint qh = Unsafe.ReadUnaligned<uint>(ref buf[4]);
            int valid = Math.Min(qk, n - blockStart);

            for (int i = 0; i < valid; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q = ((buf[8 + i / 2] >> (4 * (i % 2))) & 0x0F) | xh;
                data[blockStart + i] = q * d + m;
            }
        }
    }

    internal static unsafe void ReadQ8_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 36;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            if (Avx2.IsSupported)
            {
                var vd = Vector256.Create(d);
                fixed (byte* pBuf = buf)
                {
                    sbyte* values = (sbyte*)(pBuf + 4);
                    int j = 0;
                    for (; j <= valid - 8; j += 8)
                    {
                        var vi = Avx2.ConvertToVector256Int32(values + j);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])),
                            Avx.Multiply(Avx.ConvertToVector256Single(vi), vd));
                    }
                    for (; j < valid; j++)
                        data[blockStart + j] = values[j] * d;
                }
            }
            else
            {
                for (int j = 0; j < valid; j++)
                    data[blockStart + j] = (sbyte)buf[4 + j] * d;
            }
        }
    }

    internal static void ReadQ2K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 84;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);
            float dSuper = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[80]));
            float minSuper = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[82]));

            for (int i = 0; i < QK_K && blockStart + i < n; i++)
            {
                int pairIdx = i / 16;
                byte pair = buf[pairIdx];
                float s = pair & 0x0F;
                float m = pair >> 4;
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int quant = (buf[16 + qsByte] >> qsShift) & 3;
                data[blockStart + i] = (s * quant * dSuper) - (m * minSuper);
            }
        }
    }

    internal static unsafe void ReadQ8K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 292;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);
            float d = Unsafe.ReadUnaligned<float>(ref buf[0]);
            int valid = Math.Min(QK_K, n - blockStart);

            if (Avx2.IsSupported)
            {
                var vd = Vector256.Create(d);
                fixed (byte* pBuf = buf)
                {
                    sbyte* values = (sbyte*)(pBuf + 4);
                    int j = 0;
                    for (; j <= valid - 8; j += 8)
                    {
                        var vi = Avx2.ConvertToVector256Int32(values + j);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])),
                            Avx.Multiply(Avx.ConvertToVector256Single(vi), vd));
                    }
                    for (; j < valid; j++)
                        data[blockStart + j] = values[j] * d;
                }
            }
            else
            {
                for (int j = 0; j < valid; j++)
                    data[blockStart + j] = (sbyte)buf[4 + j] * d;
            }
        }
    }

    internal static unsafe void ReadQ5_0(BinaryReader reader, Span<float> data, int n)
    {
        // block_q5_0: d[2] + qh[4] + qs[16] = 22 bytes
        // Each element: 4 low bits from qs nibble, 1 high bit from qh
        //   val = d * ((qs_nibble | ((qh >> e) & 1) << 4) - 16)
        const int qk = 32;
        const int blockBytes = 22;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            uint qh = Unsafe.ReadUnaligned<uint>(ref buf[2]);
            int valid = Math.Min(qk, n - blockStart);

            if (Avx2.IsSupported)
            {
                var vd = Vector256.Create(d);
                var v16f = Vector256.Create(16f);

                fixed (byte* pBuf = buf)
                {
                    byte* packed = pBuf + 6;
                    var v16 = Unsafe.ReadUnaligned<Vector128<byte>>(packed);

                    var lo = Sse2.And(v16, Vector128.Create((byte)0x0F));
                    var hi = Sse2.And(
                        Sse2.ShiftRightLogical(
                            Sse2.And(v16, Vector128.Create((byte)0xF0)).AsUInt16(), 4).AsByte(),
                        Vector128.Create((byte)0x0F));

                    var iLow = Sse2.UnpackLow(lo, hi);
                    var iHigh = Sse2.UnpackHigh(lo, hi);

                    var up0 = Avx2.ConvertToVector256Int32(iLow);
                    var up1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iLow, 8));
                    var up2 = Avx2.ConvertToVector256Int32(iHigh);
                    var up3 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iHigh, 8));

                    Vector256<int> AddHighBits2(Vector256<int> vv, int bitOff)
                    {
                        var vqh = Vector256.Create((int)qh);
                        var shifts = Vector256.Create(
                            (uint)bitOff, (uint)(bitOff + 1), (uint)(bitOff + 2), (uint)(bitOff + 3),
                            (uint)(bitOff + 4), (uint)(bitOff + 5), (uint)(bitOff + 6), (uint)(bitOff + 7));
                        var oneBit = Avx2.And(
                            Avx2.ShiftRightLogicalVariable(vqh, shifts), Vector256.Create(1));
                        return Avx2.Add(vv, Avx2.ShiftLeftLogical(oneBit, 4));
                    }

                    var vq0 = AddHighBits2(up0, 0);
                    var vq1 = AddHighBits2(up1, 8);
                    var vq2 = AddHighBits2(up2, 16);
                    var vq3 = AddHighBits2(up3, 24);

                    int j = 0;
                    for (; j + 8 <= valid; j += 8)
                    {
                        var vv = j switch { 0 => vq0, 8 => vq1, 16 => vq2, _ => vq3 };
                        var vf = Avx.ConvertToVector256Single(vv);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])),
                            Avx.Multiply(Avx.Subtract(vf, v16f), vd));
                    }
                    for (; j < valid; j++)
                    {
                        int loNib = packed[j / 2] & 0x0F;
                        int hiNib = packed[j / 2] >> 4;
                        int nib = (j % 2 == 0) ? loNib : hiNib;
                        int h4 = ((int)(qh >> j) & 1) << 4;
                        data[blockStart + j] = ((nib | h4) - 16) * d;
                    }
                }
            }
            else
            {
                for (int j = 0; j < valid; j++)
                {
                    int loNib = buf[6 + j / 2] & 0x0F;
                    int hiNib = buf[6 + j / 2] >> 4;
                    int nib = (j % 2 == 0) ? loNib : hiNib;
                    uint qhVal = buf[2] | ((uint)buf[3] << 8) | ((uint)buf[4] << 16) | ((uint)buf[5] << 24);
                    int h4 = ((int)(qhVal >> j) & 1) << 4;
                    data[blockStart + j] = ((nib | h4) - 16) * d;
                }
            }
        }
    }

    internal static unsafe void ReadQ4_0(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            if (Avx2.IsSupported)
            {
                fixed (byte* pBuf = buf)
                {
                    byte* nibbles = pBuf + 2;
                    var v16 = Unsafe.ReadUnaligned<Vector128<byte>>(nibbles);

                    var lo = Sse2.And(v16, Vector128.Create((byte)0x0F));
                    var hi = Sse2.And(
                        Sse2.ShiftRightLogical(
                            Sse2.And(v16, Vector128.Create((byte)0xF0)).AsUInt16(), 4).AsByte(),
                        Vector128.Create((byte)0x0F));

                    var iLow = Sse2.UnpackLow(lo, hi);
                    var iHigh = Sse2.UnpackHigh(lo, hi);

                    var up0 = Avx2.ConvertToVector256Int32(iLow);
                    var up1 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iLow, 8));
                    var up2 = Avx2.ConvertToVector256Int32(iHigh);
                    var up3 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(iHigh, 8));

                    var v8 = Vector256.Create(8f);
                    var d256 = Vector256.Create(d);

                    int j = 0;
                    for (; j + 8 <= valid; j += 8)
                    {
                        var vv = j switch
                        {
                            0 => up0,
                            8 => up1,
                            16 => up2,
                            _ => up3
                        };
                        var vf = Avx.ConvertToVector256Single(vv);
                        Avx.Store((float*)(Unsafe.AsPointer(ref data[blockStart + j])),
                            Avx.Multiply(Avx.Subtract(vf, v8), d256));
                    }
                    for (; j < valid; j++)
                    {
                        int nib = (nibbles[j / 2] >> ((j & 1) * 4)) & 0x0F;
                        data[blockStart + j] = (nib - 8) * d;
                    }
                }
            }
            else
            {
                for (int j = 0; j < valid; j++)
                {
                    int nib = (buf[2 + j / 2] >> ((j & 1) * 4)) & 0x0F;
                    data[blockStart + j] = (nib - 8) * d;
                }
            }
        }
    }

    private static readonly float[] kvalues_iq4nl =
        { -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f, 1f, 13f, 25f, 38f, 53f, 69f, 89f, 113f };

    internal static unsafe void ReadIQ4_NL(BinaryReader reader, Span<float> data, int n)
    {
        // block_iq4_nl: d[2] + qs[16] = 18 bytes, QK=32 elements
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int nib = (buf[2 + (j >> 1)] >> (4 * (j & 1))) & 0x0F;
                data[blockStart + j] = d * kvalues_iq4nl[nib];
            }
        }
    }

    internal static void ReadQ4K(BinaryReader reader, Span<float> data, int n)
    {
        // block_q4_K layout (144 bytes, QK_K=256 elements, 8 sub-blocks of 32):
        //   d[2]       half-float superblock scale
        //   dmin[2]    half-float superblock min
        //   scales[12] packed 6-bit sub-block scales+mins (GetScaleMinK4 encoding)
        //   qs[128]    4-bit quants, low nibble = first 32 of pair, high nibble = second 32
        const int QK_K = 256;
        const int blockBytes = 144;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            float dSuper = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float minSuper = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[2]));

            // scales[12] at buf[4..15]
            var scaleSpan = buf.Slice(4, 12);

            // Each outer iteration covers 64 output elements split into two
            // sub-blocks of 32: low nibbles of a 32-byte qs window (d1/m1v),
            // then high nibbles of the same window (d2/m2v).
            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(idx, scaleSpan, out byte sc0, out byte m0);
                GetScaleMinK4(idx + 1, scaleSpan, out byte sc1, out byte m1);

                float d1 = dSuper * sc0;
                float m1v = minSuper * m0;
                float d2 = dSuper * sc1;
                float m2v = minSuper * m1;

                // qs window for this 64-element pair: 32 bytes at offset 16 + (j/64)*32
                int qIdx = 16 + (j / 64) * 32;

                int lim1 = Math.Min(32, n - blockStart - j);
                for (int l = 0; l < lim1; l++)
                    data[blockStart + j + l] = (sc0 * (buf[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper);

                int lim2 = Math.Min(32, n - blockStart - j - 32);
                for (int l = 0; l < lim2; l++)
                    data[blockStart + j + 32 + l] = (sc1 * (buf[qIdx + l] >> 4) * dSuper) - (m1 * minSuper);

                idx += 2;
            }
        }
    }

    private static void GetScaleMinK4(int j, ReadOnlySpan<byte> scales, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(scales[j] & 0x3F);
            m = (byte)(scales[j + 4] & 0x3F);
        }
        else
        {
            d = (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            m = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
    }

    internal static unsafe void ReadQ6K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 210;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            int valid = Math.Min(QK_K, n - blockStart);
            reader.Read(buf);

            fixed (byte* pBuf = buf)
            {
                byte* ql = pBuf;
                byte* qh = ql + 128;
                sbyte* scales = (sbyte*)(qh + 64);
                float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(pBuf + 128 + 64 + 16));

                for (int nOff = 0; nOff < valid; nOff += 128)
                {
                    int qlOff = nOff == 0 ? 0 : 64;
                    int qhOff = nOff == 0 ? 0 : 32;
                    int scOff = nOff == 0 ? 0 : 8;

                    int halfRem = Math.Min(128, valid - nOff);
                    for (int l = 0; l < 32 && l < halfRem; l++)
                    {
                        int is_ = l / 16;
                        int q1 = (ql[qlOff + l] & 0x0F) | ((qh[qhOff + l] & 0x03) << 4);
                        int q2 = (ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 0x03) << 4);
                        int q3 = ((ql[qlOff + l] >> 4) & 0x0F) | (((qh[qhOff + l] >> 4) & 0x03) << 4);
                        int q4 = ((ql[qlOff + l + 32] >> 4) & 0x0F) | (((qh[qhOff + l] >> 6) & 0x03) << 4);

                        int idx1 = nOff + l;
                        int idx2 = nOff + l + 32;

                        if (idx2 >= valid)
                        {
                            if (idx1 < valid)
                                data[blockStart + idx1] = d * scales[scOff + is_ + 0] * (q1 - 32);
                            break;
                        }

                        int idx3 = nOff + l + 64;
                        int idx4 = nOff + l + 96;

                        data[blockStart + idx1] = d * scales[scOff + is_ + 0] * (q1 - 32);
                        data[blockStart + idx2] = d * scales[scOff + is_ + 2] * (q2 - 32);
                        data[blockStart + idx3] = d * scales[scOff + is_ + 4] * (q3 - 32);
                        data[blockStart + idx4] = d * scales[scOff + is_ + 6] * (q4 - 32);
                    }
                }
            }
        }
    }

    internal static unsafe void ReadQ5_K(BinaryReader reader, Span<float> data, int n)
    {
        // 176-byte block: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
        const int QK_K = 256;
        const int blockBytes = 176;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            float d = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float dmin = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            var scaleSpan = buf.Slice(4, 12);

            for (int i = 0; i < QK_K && blockStart + i < n; i++)
            {
                // Sub-block scale/min (every 32 elements)
                int subIdx = i / 32;
                GetScaleMinK4(subIdx, scaleSpan, out byte sc, out byte m);

                // Quant: 4 bits from qs + 1 bit from qh
                int qsByteIdx = 48 + (i / 64) * 32 + (i % 32);
                int qsShift = ((i % 64) / 32) * 4;
                int q4 = (buf[qsByteIdx] >> qsShift) & 0x0F;

                int qhBit = (buf[16 + (i % 32)] >> ((i / 64) * 2 + ((i % 64) / 32))) & 1;
                int q5 = q4 | (qhBit << 4);

                data[blockStart + i] = (sc * q5 * d) - (m * dmin);
            }
        }
    }
    public static unsafe void ReadQ3_K(BinaryReader reader, Span<float> data, int n)
    {
        // block_q3_K layout (110 bytes, QK_K=256 elements, 16 sub-groups of 16):
        //   hmask[32]   1 bit per element (offset 0)
        //   qs[64]      2 low bits per element packed 4-per-byte (offset 32)
        //   scales[12]  packed 6-bit sub-group scales (offset 96)
        //   d[2]        half-float block scale (offset 108)
        const int QK_K = 256;
        const int blockBytes = 110;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];
        uint* pAux = stackalloc uint[4];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            // d is the LAST 2 bytes of the block (offset 108)
            float dAll = HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref buf[108]));

            // scales packed in 12 bytes at buf[96..107]
            pAux[0] = Unsafe.ReadUnaligned<uint>(ref buf[96]);
            pAux[1] = Unsafe.ReadUnaligned<uint>(ref buf[100]);
            pAux[2] = Unsafe.ReadUnaligned<uint>(ref buf[104]);
            uint tmp2 = pAux[2];
            pAux[2] = ((pAux[0] >> 4) & kmask2) | (((tmp2 >> 4) & kmask1) << 4);
            pAux[3] = ((pAux[1] >> 4) & kmask2) | (((tmp2 >> 6) & kmask1) << 4);
            pAux[0] = (pAux[0] & kmask2) | (((tmp2 >> 0) & kmask1) << 4);
            pAux[1] = (pAux[1] & kmask2) | (((tmp2 >> 2) & kmask1) << 4);
            // pAux now holds 16 unpacked 6-bit scales as bytes (indices 0..15)

            sbyte* scales = (sbyte*)pAux;

            int valid = Math.Min(QK_K, n - blockStart);
            int idx = 0;
            int qOff = 32; // qs starts at buf[32]

            // Two halves of 128 elements each.
            // Within each half the SAME 32 qs bytes are re-read with increasing
            // 2-bit shift (0,2,4,6) for the four sub-group pairs.
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    // Sub-group pair: gIdx1 covers [idx*16 .. idx*16+15],
                    //                 gIdx2 covers [(idx+1)*16 .. (idx+1)*16+15]
                    // gIdx1 elements use qs bytes at qOff+[0..15]
                    // gIdx2 elements use qs bytes at qOff+[16..31]
                    float s1 = scales[idx] - 32;
                    float s2 = scales[idx + 1] - 32;

                    int gIdx1 = idx;
                    int gIdx2 = idx + 1;

                    int lim1 = Math.Min(16, valid - gIdx1 * 16);
                    for (int l = 0; l < lim1; l++)
                    {
                        int relPos = gIdx1 * 16 + l;
                        int hmBit = (buf[relPos % 32] >> (relPos / 32)) & 1;
                        int q2 = (buf[qOff + l] >> shift) & 3;
                        data[blockStart + gIdx1 * 16 + l] = (s1 * (q2 - (hmBit != 0 ? 0 : 4))) * dAll;
                    }

                    int lim2 = Math.Min(16, valid - gIdx2 * 16);
                    for (int l = 0; l < lim2; l++)
                    {
                        int relPos = gIdx2 * 16 + l;
                        int hmBit = (buf[relPos % 32] >> (relPos / 32)) & 1;
                        int q2 = (buf[qOff + 16 + l] >> shift) & 3;
                        data[blockStart + gIdx2 * 16 + l] = (s2 * (q2 - (hmBit != 0 ? 0 : 4))) * dAll;
                    }

                    idx += 2;
                    shift += 2;
                }
                qOff += 32; // second half uses qs bytes at buf[64..95]
            }
        }
    }
}
