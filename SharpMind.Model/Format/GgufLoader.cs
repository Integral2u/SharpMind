using SharpMind.Core;
using SharpMind.Core.Diagnostics;
using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.IO.MemoryMappedFiles;
using System.Text.RegularExpressions;

namespace SharpMind.Model.Format;

/// <summary>
/// Loads GGUF containers. The GGUF-specific work lives here: reading the binary
/// header/KV/tensor index (<see cref="LoadMeta"/>), deriving config and tokenizer
/// from it, and the row-aligned on-disk byte rule inherited from
/// <see cref="ModelLoaderBase.RawByteCount"/>. Everything after the index —
/// target resolution, raw-quantized-data handling, fused QKV/expert slicing,
/// dequantization — is shared with <see cref="SmmLoader"/> in
/// <see cref="ModelLoaderBase"/>.
/// </summary>
public sealed class GgufLoader(QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false,
    int maxParallelLoadDegree = 0) : ModelLoaderBase(qOps, path, config, useSafeIo, maxParallelLoadDegree)
{
    private const uint Magic = 0x46554747;

    protected override ModelMetaData LoadIndex() => LoadMeta(_path);

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
        var bytes = ReadBytes(reader, len, "string");
        return (len, System.Text.Encoding.UTF8.GetString(bytes));
    }

    private static string ReadStringValue(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        var bytes = ReadBytes(reader, len, "string value");
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private static byte[] ReadBytes(BinaryReader reader, ulong length, string description)
    {
        if (length > int.MaxValue)
            throw new InvalidDataException($"GGUF {description} length {length} exceeds supported limits.");

        long remaining = reader.BaseStream.Length - reader.BaseStream.Position;
        if ((long)length > remaining)
            throw new InvalidDataException($"GGUF {description} length {length} exceeds the remaining file data.");

        var bytes = reader.ReadBytes((int)length);
        if ((ulong)bytes.Length != length)
            throw new EndOfStreamException($"Unexpected end of file while reading GGUF {description}.");
        return bytes;
    }

    private static object? ReadArrayValue(BinaryReader reader)
    {
        var elemType = reader.ReadUInt32();
        var arrLen = reader.ReadUInt64();
        if (arrLen > int.MaxValue)
            throw new InvalidDataException($"GGUF array length {arrLen} exceeds supported limits.");
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
                    long byteCount;
                    try
                    {
                        byteCount = checked((long)len * elemSize);
                    }
                    catch (OverflowException ex)
                    {
                        throw new InvalidDataException("GGUF array byte count overflows a signed 64-bit value.", ex);
                    }
                    if (byteCount > reader.BaseStream.Length - reader.BaseStream.Position)
                        throw new InvalidDataException("GGUF array exceeds the remaining file data.");
                    reader.BaseStream.Position += byteCount;
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
        using var stream = ModelFileIo.OpenRead(path);
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

                var dtype = GgufTypeMap.FromGgufId(reader.ReadUInt32(), name);
                var offset = reader.ReadUInt64();

                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            }
            catch (NotSupportedException) { throw; }
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

        float ropeThetaSwa = meta.GetFloat($"{arch}.rope.freq_base_swa", float.NaN);
        float? ropeThetaSwaValue = float.IsNaN(ropeThetaSwa) ? null : ropeThetaSwa;

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
        // llama.cpp defaults n_rot to the key length (full rotary) when the
        // GGUF omits rope.dimension_count — Gemma-3 rotates the full 256-dim
        // head, not headDim/2. Only an explicit key narrows the rotary span.

        // GetString returns "" for an absent key; keep the config null so the SMM->GGUF
        // converter omits the key (llama.cpp treats an empty scaling type as an error).
        string? ropeScalingType = meta.GetString($"{arch}.rope.scaling.type") is { Length: > 0 } rst ? rst : null;
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

        // Read expert_weights_norm if present (deepseek2/exaone-moe). Missing key
        // means false for qwen2moe/olmoe semantics; keep default false unless explicitly set.
        bool normTopKProb = false;
        long rawNormW = meta.GetLong($"{arch}.expert_weights_norm", -2);
        if (rawNormW >= 0)
            normTopKProb = rawNormW != 0;

        // Qwen1.5-MoE sizes the routed experts and the shared expert separately:
        // feed_forward_length (5632) is the SHARED expert, while the routed
        // experts are expert_feed_forward_length (1408) wide. Falling back to
        // ffnDim for the experts would size every expert weight guard to 5632,
        // which the 1408-wide expert planes can never satisfy.
        long rawExpertFfn = meta.GetLong($"{arch}.expert_feed_forward_length", -1);
        int expertFfnDim = rawExpertFfn > 0 ? (int)rawExpertFfn : 0;
        long rawSharedFfn = meta.GetLong($"{arch}.expert_shared_feed_forward_length", -1);
        int sharedExpertFfnDim = rawSharedFfn > 0 ? (int)rawSharedFfn : 0;

        long rawSlidingWindow = meta.GetLong($"{arch}.attention.sliding_window", -1);
        int slidingWindowSize = rawSlidingWindow > 0 ? (int)rawSlidingWindow : 0;
        PositionalEncoding positionalEncoding = (arch.StartsWith("bert", StringComparison.OrdinalIgnoreCase) || arch.StartsWith("roberta", StringComparison.OrdinalIgnoreCase) || arch.StartsWith("xlnet", StringComparison.OrdinalIgnoreCase))
            ? PositionalEncoding.NoPE
            : PositionalEncoding.RoPE;

        // Fallback: ministral and mistral3 use sliding-window attention by
        // default. When the GGUF omits the key, assume a 4096-token window
        // so the KV cache is capped and the model can load without OOM.
        if (slidingWindowSize <= 0 &&
            (string.Equals(arch, "ministral", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(arch, "mistral3", StringComparison.OrdinalIgnoreCase)))
        {
            slidingWindowSize = Math.Min(4096, maxSeqLen);
        }

        // Gemma-3: sliding-window attention with full-attention layers at
        // il % period == period - 1 (layers 5, 11, 17 for 18 blocks), and a
        // distinct RoPE base for the windowed layers (10_000 vs 1_000_000).
        // llama.cpp defaults swa_period=6 and the SWA base to 10_000 when the
        // GGUF omits sliding_window_pattern / rope.freq_base_swa, which this
        // file (Unsloth gemma-3-270m) does — matching the model's own config
        // (rope_local_base_freq=10000, rope_theta=1000000).
        bool isGemma3Family = arch.StartsWith("gemma3", StringComparison.OrdinalIgnoreCase) ||
                              arch.StartsWith("gemma4", StringComparison.OrdinalIgnoreCase);
        int? slidingWindowPattern = null;
        if (slidingWindowSize > 0)
        {
            long rawPattern = meta.GetLong($"{arch}.attention.sliding_window_pattern", -1);
            slidingWindowPattern = rawPattern > 0
                ? (int)rawPattern
                : isGemma3Family ? 6 : null;

            if (ropeThetaSwaValue == null && isGemma3Family)
                ropeThetaSwaValue = 10_000f;
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
            RopeThetaSwa = ropeThetaSwaValue,
            SlidingWindowPattern = slidingWindowPattern,
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
            ExpertFfnDim = expertFfnDim,
            SharedExpertFfnDim = sharedExpertFfnDim,
            SlidingWindowSize = slidingWindowSize,
            IsHybridSlidingWindow = slidingWindowSize > 0 && isGemma3Family,
            NormTopKProb = normTopKProb,
            PositionalEncoding = positionalEncoding,
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
            var tokenizer = Tokenizer.FromGguf(tokens, merges, types, bosId, eosId, scores, arch);
            tokenizer.GgufPreTokenizer = meta.GetString("tokenizer.ggml.pre") is { Length: > 0 } pre ? pre : null;
            tokenizer.GgufTokenizerModel = meta.GetString("tokenizer.ggml.model") is { Length: > 0 } tm ? tm : null;
            return tokenizer;
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
}