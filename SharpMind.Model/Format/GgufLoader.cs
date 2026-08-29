using SharpMind.Core;
using SharpMind.Core.Diagnostics;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text.RegularExpressions;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;

public sealed class GgufLoader(QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false) : IModelLoader
{
    private const uint Magic = 0x46554747;
    private readonly QuantizationOps _qOps = qOps ?? throw new ArgumentNullException(nameof(qOps));
    private readonly string _path = File.Exists(path)? path : throw new FileNotFoundException(path);
    private readonly ModelConfig _config = config ?? throw new ArgumentNullException(nameof(config));
    private readonly bool _useSafeIo = useSafeIo;
    private static object ReadValue(BinaryReader reader, uint valType) => valType switch
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

    // ── Static helpers (metadata / config / tokenizer loading) ────────────

    public static string[]? GetStringArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as string[];

    public static int[]? GetIntArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as int[];

    /// <summary>
    /// Reads an integer-array metadata key, accepting both GGUF INT32 arrays
    /// (etype 5) and UINT32 arrays (etype 4). Returns null when absent, empty,
    /// of an unexpected type, or containing values outside the int range.
    /// </summary>
    public static int[]? GetIntArrayNormalized(ModelMetaData meta, string key)
    {
        var value = meta.KvPairs.FirstOrDefault(p => p.Key == key).Value;
        if (value is int[] ia) return ia.Length == 0 ? null : ia;
        if (value is uint[] ua)
        {
            var result = new int[ua.Length];
            for (int i = 0; i < ua.Length; i++)
            {
                if (ua[i] > int.MaxValue) return null;
                result[i] = (int)ua[i];
            }
            return result.Length == 0 ? null : result;
        }
        return null;
    }

    public static float[]? GetFloatArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as float[];

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
                8 => ReadStringValue(reader),
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

                var dtype = (QuantDType)reader.ReadUInt32();
                var offset = reader.ReadUInt64();

                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            }
            catch (Exception ex) { SanityChecks.WriteLine($"GgufLoader: tensor metadata read failed: {ex.Message}"); break; }
        }

        uint alignment = (uint)meta.GetLong("general.alignment", 32);
        long pos = stream.Position;
        meta.DataOffset = (pos + alignment - 1) & ~(alignment - 1);

        return meta;
    }

    public static ModelConfig? LoadConfig(ModelMetaData meta)
    {
        string arch = meta.GetString("general.architecture");
        if (string.IsNullOrWhiteSpace(arch)) return null;

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

        hiddenDim = (int)meta.GetLong($"{arch}.embedding_length", hiddenDim);
        ffnDim = (int)meta.GetLong($"{arch}.feed_forward_length", ffnDim);
        maxSeqLen = (int)meta.GetLong($"{arch}.context_length", maxSeqLen);
        numHeads = (int)meta.GetLong($"{arch}.attention.head_count", numHeads);
        numKvHeads = (int)meta.GetLong($"{arch}.attention.head_count_kv", -1);
        if (numKvHeads <= 0) numKvHeads = numHeads;

        // Per-layer KV head counts ({arch}.attention.head_count_kv as an array).
        // Zero entries mark blocks without attention (e.g. LFM2 short-conv layers).
        // NumKvHeads is set to the maximum across all attention blocks.
        int[]? layerKvHeads = GetIntArrayNormalized(meta, $"{arch}.attention.head_count_kv");
        if (layerKvHeads is { Length: > 0 })
        {
            int maxLayerKvHeads = layerKvHeads.Max();
            if (maxLayerKvHeads > 0)
            {
                numKvHeads = maxLayerKvHeads;
            }
            else
            {
                // All zeros — treat the key as absent and keep the scalar path.
                layerKvHeads = null;
            }
        }

        long rawKeyLen = meta.GetLong($"{arch}.attention.key_length", -1);
        int? keyLength = rawKeyLen > 0 ? (int)rawKeyLen : null;
        long rawValLen = meta.GetLong($"{arch}.attention.value_length", -1);
        int? valueLength = rawValLen > 0 ? (int)rawValLen : null;

        numLayers = (int)meta.GetLong($"{arch}.block_count", numLayers);

        float ropeTheta = meta.GetFloat($"{arch}.rope.freq_base",
                          meta.GetFloat("rope_theta",
                          meta.GetFloat("rope.freq_base", 10_000f)));

        int tensorVocabSize = vocabSize; // from token_embd.weight shape
        int metaVocab = (int)meta.GetLong($"{arch}.vocab_size",
                         meta.GetLong("tokenizer.ggml.token_count",
                         meta.GetLong("vocab_size", vocabSize)));
        // Clamp to tensor dimension — metadata token count may include
        // added-token entries that the GGUF embedding tensor doesn't store.
        if (metaVocab > 0) vocabSize = Math.Min(metaVocab, tensorVocabSize);

        long rawHeadDim = meta.GetLong($"{arch}.head_dim", -1);
        int? headDimOverride = rawHeadDim > 0 ? (int)rawHeadDim : null;

        long rawRopeDim = meta.GetLong($"{arch}.rope.dimension_count", -1);
        int? ropeDim = rawRopeDim > 0 ? (int)rawRopeDim : null;
        // Gemma-3 uses partial RoPE (headDim/2) — default if metadata missing.
        // HeadDim = KeyLength ?? HeadDimOverride ?? HiddenDim / NumHeads
        if (ropeDim == null && arch.StartsWith("gemma3", StringComparison.OrdinalIgnoreCase))
            ropeDim = (keyLength ?? headDimOverride ?? hiddenDim / numHeads) / 2;

        string? ropeScalingType = meta.GetString($"{arch}.rope.scaling.type");
        float ropeFactor = meta.GetFloat($"{arch}.rope.scaling.factor", float.NaN);
        float? ropeScalingFactor = float.IsNaN(ropeFactor) ? null : ropeFactor;
        long rawRopeOrigCtx = meta.GetLong($"{arch}.rope.scaling.original_context_length", -1);
        int? ropeOriginalContextLength = rawRopeOrigCtx > 0 ? (int)rawRopeOrigCtx : null;

        float lowFreq = meta.GetFloat($"{arch}.rope.scaling.low_freq_factor", float.NaN);
        float? ropeLowFreqFactor = float.IsNaN(lowFreq) ? null : lowFreq;
        float highFreq = meta.GetFloat($"{arch}.rope.scaling.high_freq_factor", float.NaN);
        float? ropeHighFreqFactor = float.IsNaN(highFreq) ? null : highFreq;

        long rawTie = meta.GetLong($"{arch}.tie_word_embeddings", -1);
        bool? tieWordEmbeddings = rawTie >= 0 ? (rawTie != 0) : null;

        long rawNormType = meta.GetLong($"{arch}.norm_type", -1);
        int? normTypeOverride = rawNormType >= 0 ? (int)rawNormType : null;

        long rawExpertCount = meta.GetLong($"{arch}.expert_count", -1);
        int expertCount = rawExpertCount > 0 ? (int)rawExpertCount : 8;
        long rawTopK = meta.GetLong($"{arch}.expert_used_count", -1);
        int topKExperts = rawTopK > 0 ? (int)rawTopK : 2;

        long rawSlidingWindow = meta.GetLong($"{arch}.attention.sliding_window", -1);
        int slidingWindowSize = rawSlidingWindow > 0 ? (int)rawSlidingWindow : 0;

        // Fallback: ministral and mistral3 use sliding-window attention by
        // default. When the GGUF omits the key, assume a 4096-token window
        // so the KV cache is capped and the model can load without OOM.
        if (slidingWindowSize <= 0 &&
            (string.Equals(arch, "ministral", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(arch, "mistral3", StringComparison.OrdinalIgnoreCase)))
        {
            slidingWindowSize = Math.Min(4096, maxSeqLen);
        }

        return new ModelConfig
        {
            Architecture = arch,
            VocabSize = vocabSize,
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            LayerKvHeads = layerKvHeads,
            ShortConvCacheLength = (int)meta.GetLong($"{arch}.shortconv.l_cache", 3),
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
            RopeLowFreqFactor = ropeLowFreqFactor,
            RopeHighFreqFactor = ropeHighFreqFactor,
            TieWordEmbeddings = tieWordEmbeddings,
            NormTypeOverride = normTypeOverride,
            NumExperts = expertCount,
            TopKExperts = topKExperts,
            SlidingWindowSize = slidingWindowSize,
        };
    }

    public static Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta, int maxVocabSize = 0)
    {
        var tokens = GetStringArray(meta, "tokenizer.ggml.tokens");
        if (tokens == null || tokens.Length == 0) return null;

        // Cap the token list to maxVocabSize to prevent IDs beyond the GGUF
        // embedding tensor's capacity. The SentencePiece / BPE encoder won't
        // produce token IDs beyond this range, which would otherwise result in
        // zero-embedding lookups.
        if (maxVocabSize > 0 && tokens.Length > maxVocabSize)
            tokens = tokens.AsSpan(0, maxVocabSize).ToArray();

        var types = GetIntArray(meta, "tokenizer.ggml.token_type");
        var merges = GetStringArray(meta, "tokenizer.ggml.merges");
        var scores = GetFloatArray(meta, "tokenizer.ggml.scores");

        int bosId = (int)meta.GetLong("tokenizer.ggml.bos_token_id", 1);
        int eosId = (int)meta.GetLong("tokenizer.ggml.eos_token_id", 2);

        try
        {
            string arch = meta.GetString("general.architecture") ?? "";
            return Tokenizer.FromGguf(tokens, merges, types, bosId, eosId, scores, arch);
        }
        catch (Exception ex)
        {
            SanityChecks.WriteLine($"GgufLoader: GGUF tokenizer construction failed: {ex.Message}");
            return null;
        }
    }

    private static void InjectMissingTemplateTokens(
        ModelMetaData meta, ref ModelConfig config, Tokenizer tokenizer)
    {
        string? template = meta.GetChatTemplate();
        if (string.IsNullOrEmpty(template)) return;

        var candidates = new HashSet<string>();
        foreach (Match m in RegexGenerated.ChatTemplateRegex.Matches(template))
            candidates.Add(m.Value);

        if (candidates.Count == 0) return;

        int added = 0;
        foreach (string token in candidates)
        {
            // Register template tokens as specials so SplitOnSpecials matches
            // them (e.g. TinyLlama's <|user|>). Tokens already in the vocab get
            // their existing ID; genuinely missing tokens are appended and the
            // tensor is zero-padded to accommodate them during weight loading.
            if (!tokenizer.Vocab.Contains(token))
                config = config with { VocabSize = config.VocabSize + 1 };
            tokenizer.AddAdditionalToken(token);
            added++;
        }

        if (added > 0)
            SanityChecks.WriteLine($"GgufLoader: ensured {added} template tokens are registered as specials");
    }

    public static void Load(
        string ggufPath,
        string? tokenizerPath,
        out ModelMetaData meta,
        out ModelConfig config,
        out Tokenizer? tokenizer)
    {
        meta = LoadMeta(ggufPath);
        config = LoadConfig(meta)!;

        // Disabled: GGUF-exported rope_freqs.weight is often all-1.0 (bug),
        // causing all RoPE pairs to rotate by angle = pos (wrong). Theta-
        // based computation produces correct frequencies and is used instead.
        // float[]? ropeFreqs = LoadPrecomputedRopeFreqs(ggufPath, meta);
        // if (ropeFreqs != null) config = config with { PrecomputedRopeFreqs = ropeFreqs };

        // Extend VocabSize to cover the full GGUF token list.
        // Some GGUFs store control/special tokens beyond the embedding tensor
        // dimension (e.g. TinyLlama Chat's <|user|> tokens at index 32000+).
        // The tensor padding code in LoadSingleTensor zero-pads the extra rows.
        var allTokens = GetStringArray(meta, "tokenizer.ggml.tokens");
        if (allTokens != null && allTokens.Length > config.VocabSize)
            config = config with { VocabSize = allTokens.Length };

        tokenizer = LoadTokenizerFromMeta(meta, config.VocabSize);

        if (tokenizer == null && !string.IsNullOrEmpty(tokenizerPath) && File.Exists(tokenizerPath))
        {
            try
            {
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
                else if (arch.Contains("mistral", StringComparison.OrdinalIgnoreCase)
                      || arch.Contains("ministral", StringComparison.OrdinalIgnoreCase))
                {
                    tokenizer = Tokenizer.FromMistral(tokenizerPath);
                }
                else
                {
                    tokenizer = Tokenizer.FromFile(tokenizerPath);
                }
            }
            catch (Exception ex)
            {
                SanityChecks.WriteLine($"GgufLoader: external tokenizer file failed: {ex.Message}");
                tokenizer = null;
            }
        }

        if (tokenizer != null)
            InjectMissingTemplateTokens(meta, ref config, tokenizer);
    }

    private static float[]? LoadPrecomputedRopeFreqs(string ggufPath, ModelMetaData meta)
    {
        var ropeFreqsInfo = meta.Tensors.FirstOrDefault(t => t.Name == "rope_freqs.weight");
        if (ropeFreqsInfo.Shape == null || ropeFreqsInfo.Shape.Length == 0) return null;

        int count = 1;
        foreach (int d in ropeFreqsInfo.Shape) count *= d;
        if (count == 0 || ropeFreqsInfo.Dtype != QuantDType.F32) return null;

        float[] result = new float[count];
        try
        {
            using var mmf = MemoryMappedFile.CreateFromFile(ggufPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
            long dataPos = meta.DataOffset + ropeFreqsInfo.Offset;
            stream.Position = dataPos;
            using var reader = new BinaryReader(stream);
            for (int i = 0; i < count; i++)
                result[i] = reader.ReadSingle();
            return result;
        }
        catch
        {
            return null;
        }
    }

    // ── IModelLoader implementation ────────────────────────────────────────

    public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null)
    {
        Core.Memory.NativeBufferPool<float>.Clear();

        var meta = LoadMeta(_path);
        weights.GgufMeta = meta;
        weights.GgufPath = _path;
        weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));

        int total = meta.Tensors.Count;
        int loaded = 0;

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);
        using var reader = new BinaryReader(stream);

        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)loaded / total);
            LoadSingleTensor(weights, meta, stream, reader, info);
            loaded++;
        }
        progress?.Report(1f);
    }

    public void LoadLayerWeights(int layerIndex, TransformerWeights weights)
    {
        var meta = weights.GgufMeta;
        if (meta == null)
        {
            meta = LoadMeta(_path);
            weights.GgufMeta = meta;
            weights.GgufPath = _path;
            weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));
        }

        var targetBlock = layerIndex < weights.Blocks.Length ? weights.Blocks[layerIndex] : null;
        if (targetBlock == null) return;

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);
        using var reader = new BinaryReader(stream);

        foreach (var info in meta.Tensors)
        {
            var (_, block, _) = weights.ResolveTarget(info.Name);
            if (block == targetBlock)
            {
                LoadSingleTensor(weights, meta, stream, reader, info);
            }
        }
    }

    public void LoadGlobalTensors(TransformerWeights weights)
    {
        var meta = weights.GgufMeta ?? LoadMeta(_path);
        if (weights.GgufMeta == null)
        {
            weights.GgufMeta = meta;
            weights.GgufPath = _path;
            weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));
        }

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);
        using var reader = new BinaryReader(stream);

        foreach (var info in meta.Tensors)
        {
            var (target, block, _) = weights.ResolveTarget(info.Name);
            if (target != null && block == null)
            {
                LoadSingleTensor(weights, meta, stream, reader, info);
            }
        }
    }

    private void LoadSingleTensor(
        TransformerWeights weights, ModelMetaData meta,
        Stream stream, BinaryReader reader, TensorInfo info)
    {
        var (target, block, rawField) = weights.ResolveTarget(info.Name);

        // Must create LmHeadWeight BEFORE the early-return check below:
        // ResolveTarget returns (null, null, null) for output.weight when
        // LmHeadWeight is null (first encounter). Without this early creation,
        // the LM head tensor is never allocated and the weight data is skipped.
        if (!info.Name.Contains("blk.") && info.Name.Contains("output.weight") && weights.LmHeadWeight == null)
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
            weights.SetLmHead(new Tensor<float>(lmRows, lmCols));
            // Re-resolve now that LmHeadWeight is set
            (target, block, rawField) = weights.ResolveTarget(info.Name);
        }

        if (target == null && block == null) return;

        long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);

        // Record tensor metadata and top-level dtypes (consumed by SetWeights later)
        if (target != null && block == null && rawSize > 0)
        {
            if (target == weights.LmHeadWeight)
                weights.RawLmHeadDtype = info.Dtype;
            else if (target == weights.EmbeddingWeight)
                weights.RawEmbeddingDtype = info.Dtype;
        }
        if (block != null && rawField != null && rawSize > 0)
            SetTensorMeta(block, rawField, meta.DataOffset + info.Offset, TensorLoadHelper.CheckedInt(rawSize, "rawSize"), info.Dtype);

        long targetOffset = meta.DataOffset + info.Offset;
        if (targetOffset >= stream.Length) return;
        stream.Position = targetOffset;

        // long: a tensor can hold more elements than int can count (gemma-3n's
        // per-layer embeddings are 2.3e9), and silently wrapping negative here
        // surfaced far away as ArrayPool.Rent(-1946157056).
        long longCount = TensorLoadHelper.ComputeElementCount(info.Shape);
        if (longCount > int.MaxValue)
        {
            // Oversized tensor: raw bytes are already loaded above for both
            // block-level and top-level tensors. Skip dequant — the streaming
            // forward pass reads the raw quantized bytes directly.
            return;
        }
        int count = (int)longCount;

        // Load raw quantized data for block-level tensors
        if (block != null && rawField != null && rawSize > 0 && stream.Position + rawSize <= stream.Length)
        {
            // Fused QKV (e.g. Phi-3: "blk.N.attn_qkv.weight" → [out, 3*in]): split
            // into separate Q/K/V byte buffers so each InferenceLinearLayer gets the
            // exact [out, in] chunk its size guard expects.
            if (rawField == "RawWqkv")
            {
                if (rawSize % 3 != 0)
                    throw new InvalidDataException(
                        $"Fused QKV tensor '{info.Name}' has {rawSize} bytes which is not divisible by 3.");

                int partSize = (int)(rawSize / 3);
                byte[] fused = new byte[rawSize];
                stream.ReadExactly(fused);
                stream.Position -= rawSize;

                byte[] qPart = new byte[partSize];
                byte[] kPart = new byte[partSize];
                byte[] vPart = new byte[partSize];
                Buffer.BlockCopy(fused, 0, qPart, 0, partSize);
                Buffer.BlockCopy(fused, partSize, kPart, 0, partSize);
                Buffer.BlockCopy(fused, partSize * 2, vPart, 0, partSize);

                SetRawField(block, "RawWq", qPart, info.Dtype);
                SetRawField(block, "RawWk", kPart, info.Dtype);
                SetRawField(block, "RawWv", vPart, info.Dtype);

                // Record metadata for each split portion (offsets are approximate;
                // the streaming forward path reads RawWq/RawWk/RawWv directly).
                SetTensorMeta(block, "RawWq", meta.DataOffset + info.Offset, partSize, info.Dtype);
                SetTensorMeta(block, "RawWk", meta.DataOffset + info.Offset + partSize, partSize, info.Dtype);
                SetTensorMeta(block, "RawWv", meta.DataOffset + info.Offset + partSize * 2, partSize, info.Dtype);
                return;
            }

            byte[] rawData = new byte[rawSize];
            stream.ReadExactly(rawData);
            stream.Position -= rawSize;
            SetRawField(block, rawField, rawData, info.Dtype);
        }

        // Load raw quantized data for top-level tensors
        if (target != null && block == null && rawSize > 0 && stream.Position + rawSize <= stream.Length)
        {
            byte[] rawData;
            if (info.Shape.Length >= 2)
            {
                long tensorVocab = Math.Max(info.Shape[0], info.Shape[1]);
                long paddedVocab = _config.VocabSize;
                if (paddedVocab > tensorVocab)
                {
                    long colBytes = rawSize / tensorVocab;
                    int safeColBytes = TensorLoadHelper.CheckedInt(colBytes, "colBytes");
                    rawData = new byte[paddedVocab * colBytes];
                    for (long r = 0; r < tensorVocab; r++)
                        stream.ReadExactly(rawData, (int)(r * colBytes), safeColBytes);
                    stream.Position -= rawSize;
                }
                else
                {
                    rawData = new byte[rawSize];
                    stream.ReadExactly(rawData);
                    stream.Position -= rawSize;
                }
            }
            else
            {
                rawData = new byte[rawSize];
                stream.ReadExactly(rawData);
                stream.Position -= rawSize;
            }

            if (target == weights.LmHeadWeight)
            {
                weights.RawLmHead = rawData;
                weights.RawLmHeadDtype = info.Dtype;
            }
            else if (target == weights.EmbeddingWeight)
            {
                weights.RawEmbedding = rawData;
                weights.RawEmbeddingDtype = info.Dtype;
            }
        }

        // Dequantize to float — but only if something will consume the floats.
        // In streaming mode the 2D weight tensors are deliberately left null and
        // the quantized forward reads the raw bytes instead, so dequantizing here
        // did full work per layer per forward pass and discarded every value.
        // Each tensor is seeked to explicitly above, so skipping reads nothing
        // the next tensor depends on.
        Tensor<float>? blockFloatTarget = target == null && block != null
            ? weights.ResolveFloatTarget(info.Name)
            : null;
        if (target == null && blockFloatTarget == null) return;

        // For non-block tensors (embedding, lm_head, norms) read directly into
        // target.Data — no temp buffer needed.  This saves ~600 MB for Bonsai-8B's
        // token_embd.weight [151936, 1024] which previously allocated a duplicate
        // float buffer via ArrayPool power-of-2 bucketing on top of the target tensor.
        if (target != null && block == null)
        {
            target.Data.Clear();
            ReadTensorInto(reader, info.Dtype, info.Shape, target.Data);
            return;
        }

        float[] buffer = MemoryHelpers.RentArray<float>(count);
        try
        {
            ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));
            if (block != null)
            {
                var floatTarget = blockFloatTarget;
                if (floatTarget != null)
                {
                    // LFM2 short-conv kernel [l_cache, hidden] is not a linear-layer
                    // weight — it stays row-major [kernelRow, channel], so it must not
                    // go through the in/out transposition below.
                    if (info.Name.Contains("shortconv.conv.weight", StringComparison.OrdinalIgnoreCase))
                    {
                        floatTarget.Data.Clear();
                        buffer.AsSpan(0, count).CopyTo(floatTarget.Data);
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
        }
        finally
        {
            MemoryHelpers.ReturnArray(buffer);
        }
    }

    private void ReadTensorInto(BinaryReader stream, QuantDType dtype, int[] shape, Span<float> destination)
    {
        long longCount = 1;
        foreach (int d in shape) longCount *= d;
        if (longCount > int.MaxValue)
            throw new NotSupportedException(
                $"Tensor with shape [{string.Join(",", shape)}] has {longCount:N0} elements, " +
                $"more than a single float buffer can hold (max {int.MaxValue:N0}).");
        int count = (int)longCount;
        if (destination.Length < count) throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");
        _qOps.ReadFor(dtype, stream, destination, count);
    }
}
