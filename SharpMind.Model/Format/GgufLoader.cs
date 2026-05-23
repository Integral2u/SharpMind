using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text.RegularExpressions;

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

    // ?? KvPair array helpers ??????????????????????????????????????????????

    public static string[]? GetStringArray(GgufMeta meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as string[];

    public static float[]? GetFloatArray(GgufMeta meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as float[];

    public static int[]? GetIntArray(GgufMeta meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as int[];

    public static GgufMeta LoadMeta(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        var meta = new GgufMeta();

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

    public static ModelConfig? LoadConfig(GgufMeta meta)
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
        numKvHeads = (int)meta.GetLong($"{arch}.attention.head_count_kv", numKvHeads);

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
        };
    }

    // ?? Tokenizer from GGUF ???????????????????????????????????????????????

    /// <summary>
    /// Builds a <see cref="Tokenizer"/> directly from the GGUF metadata.
    /// This is the preferred path: the vocab stored in GGUF always matches the
    /// model weights, so it can never produce the OOB token-ID crashes that
    /// occur when an external tokenizer.json has the wrong vocab size.
    /// </summary>
    public static Tokenizer? LoadTokenizerFromMeta(GgufMeta meta)
    {
        var tokens = GetStringArray(meta, "tokenizer.ggml.tokens");
        if (tokens == null || tokens.Length == 0) return null;
        

        var scores = GetFloatArray(meta, "tokenizer.ggml.scores");
        var types = GetIntArray(meta, "tokenizer.ggml.token_type");
        var merges = GetStringArray(meta, "tokenizer.ggml.merges");

        int bosId = (int)meta.GetLong("tokenizer.ggml.bos_token_id", 1);
        int eosId = (int)meta.GetLong("tokenizer.ggml.eos_token_id", 2);

        try
        {
            return Tokenizer.FromGguf(tokens, merges, scores, types, bosId, eosId);
        }
        catch
        {
            return null;
        }
    }

    // ?? Template token injection ????????????????????????????????????????

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
        GgufMeta meta, ref ModelConfig config, Tokenizer tokenizer)
    {
        string? template = meta.GetChatTemplate();
        if (string.IsNullOrEmpty(template)) return;

        // Extract <...> patterns from the template
        var candidates = new HashSet<string>();
        foreach (Match m in Regex.Matches(template, @"<[^>]+>"))
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
        out GgufMeta meta,
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

    public static void LoadWeightsToModel(string path, GgufMeta meta, Transformer model, IProgress<float>? progress = null)
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
            if (IsQuantizedType(info.Dtype) && info.Shape.Length >= 2)
            {
                long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0 && savedPos + rawSize <= stream.Length)
                {
                    byte[] rawData = new byte[rawSize];
                    stream.Read(rawData, 0, rawData.Length);
                    stream.Position = savedPos; // seek back for dequant read
                    model.SetRawWeight(info.Name, rawData, info.Dtype);
                }
            }

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

        progress?.Report(1f);
    }

    private static bool IsQuantizedType(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q2_K or GgufDtype.Q3_K or GgufDtype.Q4_K or GgufDtype.Q5_K or GgufDtype.Q6_K
        or GgufDtype.Q4_0 or GgufDtype.Q4_1 or GgufDtype.Q5_0 or GgufDtype.Q5_1
        or GgufDtype.Q8_0 or GgufDtype.Q8_1 or GgufDtype.Q8_K => true,
        _ => false
    };

    private static long GetRawTensorByteCount(int[] shape, GgufDtype dtype)
    {

        int nRows = shape[0];
        int nCols = shape.Length > 1 ? shape[1] : 1;
        int blockSize, bytesPerBlock;

        switch (dtype)
        {
            case GgufDtype.Q3_K: blockSize = 256; bytesPerBlock = 110; break;  // hmask[32]+qs[64]+scales[12]+d[2]
            case GgufDtype.Q4_K: blockSize = 256; bytesPerBlock = 144; break;  // d[2]+dmin[2]+scales[12]+qs[128]
            case GgufDtype.Q5_K: blockSize = 256; bytesPerBlock = 176; break;  // d[2]+dmin[2]+scales[12]+qh[32]+qs[128]
            case GgufDtype.Q6_K: blockSize = 256; bytesPerBlock = 210; break;  // ql[128]+qh[64]+scales[16]+d[2]
            case GgufDtype.Q2_K: blockSize = 256; bytesPerBlock = 84;  break;  // d[2]+dmin[2]+scales[16]+qs[64]
            case GgufDtype.Q8_K: blockSize = 256; bytesPerBlock = 292; break;  // d[4]+qs[256]+bsums[32]
            case GgufDtype.Q8_0: blockSize = 32;  bytesPerBlock = 34;  break;
            case GgufDtype.Q8_1: blockSize = 32;  bytesPerBlock = 36;  break;  // d[2]+s[2]+qs[32]
            case GgufDtype.Q5_0: blockSize = 32;  bytesPerBlock = 22;  break;
            case GgufDtype.Q5_1: blockSize = 32;  bytesPerBlock = 24;  break;  // d[2]+m[2]+qh[4]+qs[16]
            case GgufDtype.Q4_0: blockSize = 32;  bytesPerBlock = 18;  break;
            case GgufDtype.Q4_1: blockSize = 32;  bytesPerBlock = 20;  break;  // d[2]+m[2]+qs[16]
            default: return 0;
        }
        // Correct: outermost dim (shape[1]) is the number of quantized "rows";
        // innermost dim (shape[0]) is what gets split into blocks.
        int nQuantRows = shape.Length > 1 ? shape[1] : 1;
        int nQuantCols = shape[0];
        int nBlocks = (nQuantCols + blockSize - 1) / blockSize;
        return (long)nQuantRows * nBlocks * bytesPerBlock;
        /*  --old
        int nBlocks = (nCols + blockSize - 1) / blockSize;
        return (long)nRows * nBlocks * bytesPerBlock;
        */
    }

    // ?? Tensor reading ????????????????????????????????????????????????????

    private static Tensor<float> ReadTensor(BinaryReader stream, GgufDtype dtype, int[] shape)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        var result = new Tensor<float>(shape);
        ReadTensorInto(stream, dtype, shape, result.Data);
        return result;
    }

    private static void ReadTensorInto(BinaryReader stream, GgufDtype dtype, int[] shape, Span<float> destination)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        if (destination.Length < count)
            throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");

        try
        {
            switch (dtype)
            {
                case GgufDtype.F32:
                    for (int i = 0; i < count; i++) destination[i] = stream.ReadSingle();
                    break;
                case GgufDtype.F16:
                    for (int i = 0; i < count; i++) destination[i] = HalfToFloat(stream.ReadUInt16());
                    break;
                case GgufDtype.Q4_0:
                    ReadQ4_0(stream, destination, count);
                    break;
                case GgufDtype.Q4_K:
                    ReadQ4K(stream, destination, count);
                    break;
                case GgufDtype.Q6_K:
                    ReadQ6K(stream, destination, count);
                    break;
                case GgufDtype.Q8_0:
                    ReadQ8_0(stream, destination, count);
                    break;
                case GgufDtype.Q5_K:
                    ReadQ5_K(stream, destination, count);
                    break;
                case GgufDtype.Q3_K:
                    ReadQ3_K(stream, destination, count);
                    break;
                case GgufDtype.Q4_1:
                    ReadQ4_1(stream, destination, count);
                    break;
                case GgufDtype.Q5_1:
                    ReadQ5_1(stream, destination, count);
                    break;
                case GgufDtype.Q8_1:
                    ReadQ8_1(stream, destination, count);
                    break;
                case GgufDtype.Q2_K:
                    ReadQ2K(stream, destination, count);
                    break;
                case GgufDtype.Q8_K:
                    ReadQ8K(stream, destination, count);
                    break;
                case GgufDtype.Q5_0:
                    ReadQ5_0(stream, destination, count);
                    break;
                default:
                    // Unknown/unhandled quant type — zero-fill to avoid garbage weights
                    destination[..count].Clear();
                    break;
            }
        }
        catch { }//partial tensor � leave zeros 
    }

    private static float HalfToFloat(ushort half)
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
            return 0f; // NaN and Inf from F16 → guarded to zero

        return (sign == 0 ? 1f : -1f) * MathF.Pow(2f, exp - 15) * (1f + mant / 1024f);
    }

    private static float SafeHalfToFloat(ushort half)
    {
        float v = HalfToFloat(half);
        return float.IsNaN(v) || float.IsInfinity(v) ? 0f : v;
    }

    internal static void ReadQ8_0(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            float d = HalfToFloat(reader.ReadUInt16());
            for (int j = 0; j < qk && blockStart + j < n; j++)
                data[blockStart + j] = reader.ReadSByte() * d;
        }
    }

    internal static void ReadQ4_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            float d = HalfToFloat(reader.ReadUInt16());
            float m = HalfToFloat(reader.ReadUInt16());
            byte[] packed = reader.ReadBytes(16);
            for (int j = 0; j < qk && blockStart + j < n; j++)
            {
                int q = (packed[j / 2] >> (4 * (j % 2))) & 0x0F;
                data[blockStart + j] = q * d + m;
            }
        }
    }

    internal static void ReadQ5_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            float d = HalfToFloat(reader.ReadUInt16());
            float m = HalfToFloat(reader.ReadUInt16());
            uint qh = reader.ReadUInt32();
            byte[] packed = reader.ReadBytes(16);
            for (int i = 0; i < qk && blockStart + i < n; i++)
            {
                int j = i % 16;
                int xh = i < 16
                    ? (int)((qh >> (j + 0)) & 1) << 4
                    : (int)((qh >> (j + 12)) & 1) << 4;
                int q = ((packed[j / 2] >> (4 * (j % 2))) & 0x0F) | xh;
                data[blockStart + i] = q * d + m;
            }
        }
    }

    internal static void ReadQ8_1(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            float d = HalfToFloat(reader.ReadUInt16());
            reader.ReadUInt16(); // skip s (d * sum(qs)) - only used for dot-product optimization
            for (int j = 0; j < qk && blockStart + j < n; j++)
                data[blockStart + j] = reader.ReadSByte() * d;
        }
    }

    internal static void ReadQ2K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            float dSuper = HalfToFloat(reader.ReadUInt16());
            float minSuper = HalfToFloat(reader.ReadUInt16());
            byte[] scales = reader.ReadBytes(16);
            byte[] qs = reader.ReadBytes(64);
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    byte sc0 = scales[isc];
                    float dl = dSuper * (sc0 & 0x0F);
                    float ml = minSuper * (sc0 >> 4);
                    int base_ = blockStart + n16 + j * 32;
                    for (int l = 0; l < 16 && base_ + l < n; l++)
                        data[base_ + l] = dl * ((qs[qOff + l] >> shift) & 3) - ml;

                    byte sc1 = scales[isc + 1];
                    dl = dSuper * (sc1 & 0x0F);
                    ml = minSuper * (sc1 >> 4);
                    for (int l = 0; l < 16 && base_ + 16 + l < n; l++)
                        data[base_ + 16 + l] = dl * ((qs[qOff + l + 16] >> shift) & 3) - ml;

                    shift += 2;
                }
                qOff += 32;
            }
        }
    }

    internal static void ReadQ8K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            float d = reader.ReadSingle();
            for (int j = 0; j < QK_K && blockStart + j < n; j++)
                data[blockStart + j] = reader.ReadSByte() * d;
            reader.ReadBytes(QK_K / 16 * 2); // skip bsums[16] int16
        }
    }

    internal static void ReadQ5_0(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            float d = HalfToFloat(reader.ReadUInt16());
            uint qh = reader.ReadUInt32();
            byte[] packed = reader.ReadBytes(16);

            for (int j = 0; j < qk / 2; j++)
            {
                int xh0 = ((int)(qh >> (j + 0)) << 4) & 0x10;
                int xh1 = ((int)(qh >> (j + 12))) & 0x10;
                int x0 = ((packed[j] & 0x0F) | xh0) - 16;
                int x1 = ((packed[j] >> 4) | xh1) - 16;
                if (blockStart + j < n)
                    data[blockStart + j] = x0 * d;
                if (blockStart + j + qk / 2 < n)
                    data[blockStart + j + qk / 2] = x1 * d;
            }
        }
    }

    internal static void ReadQ4_0(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        int nBlocks = (n + blockSize - 1) / blockSize;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, n);

            // Q4_0: 1 half-scale + 16 bytes packed 4-bit values = 18 bytes per 32 values
            float scale = HalfToFloat(reader.ReadUInt16());
            byte[] packed = reader.ReadBytes(16);

            for (int j = 0; j < blockSize && blockStart + j < blockEnd; j++)
            {
                int q = (packed[j / 2] >> (4 * (j % 2))) & 0x0F;
                data[blockStart + j] = (q - 8) * scale;
            }
        }
    }

    internal static void ReadQ4K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;

            // block_q4_K: d[2] + dmin[2] + scales[12] + qs[128] = 144 bytes
            float dSuper = HalfToFloat(reader.ReadUInt16());
            float minSuper = HalfToFloat(reader.ReadUInt16());

            byte[] scales = reader.ReadBytes(12);
            byte[] qs = reader.ReadBytes(128);

            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(idx + 0, scales, out byte sc0, out byte m0);
                GetScaleMinK4(idx + 1, scales, out byte sc1, out byte m1);

                float d1 = dSuper * sc0;
                float m1v = minSuper * m0;
                float d2 = dSuper * sc1;
                float m2v = minSuper * m1;

                int qIdx = (j / 64) * 32;
                for (int l = 0; l < 32 && blockStart + j + l < n; l++)
                    data[blockStart + j + l] = d1 * (qs[qIdx + l] & 0x0F) - m1v;
                for (int l = 0; l < 32 && blockStart + j + 32 + l < n; l++)
                    data[blockStart + j + 32 + l] = d2 * (qs[qIdx + l] >> 4) - m2v;

                idx += 2;
            }
        }
    }

    private static void GetScaleMinK4(int j, byte[] scales, out byte d, out byte m)
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

    internal static void ReadQ6K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            int valid = Math.Min(QK_K, n - blockStart);

            // Q6_K (QK_K=256): 128B ql + 64B qh + 16 int8 scales + 2B half d = 210 bytes per 256 values
            byte[] ql = reader.ReadBytes(128);
            byte[] qh = reader.ReadBytes(64);

            sbyte[] scales = new sbyte[16];
            for (int s = 0; s < 16; s++)
                scales[s] = reader.ReadSByte();

            float d = HalfToFloat(reader.ReadUInt16());

            // Current ggml Q6_K: process 128 values at a time
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

    internal static void ReadQ5_K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;

            // block_q5_K: d[2] + dmin[2] + scales[12] + qh[32] + qs[128] = 176 bytes
            float d = HalfToFloat(reader.ReadUInt16());
            float min = HalfToFloat(reader.ReadUInt16());
            byte[] scales = reader.ReadBytes(12);
            byte[] qh = reader.ReadBytes(32);
            byte[] qs = reader.ReadBytes(128);

            int idx = 0;
            int qIdx = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(idx + 0, scales, out byte sc0, out byte m0);
                GetScaleMinK4(idx + 1, scales, out byte sc1, out byte m1);
                float d1 = d * sc0; float m1v = min * m0;
                float d2 = d * sc1; float m2v = min * m1;

                for (int l = 0; l < 32 && blockStart + j + l < n; l++)
                {
                    int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                    data[blockStart + j + l] = d1 * val - m1v;
                }
                for (int l = 0; l < 32 && blockStart + j + 32 + l < n; l++)
                {
                    int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                    data[blockStart + j + 32 + l] = d2 * val - m2v;
                }
                qIdx += 32;
                idx += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
    }

    internal static void ReadQ3_K(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        int nBlocks = (n + QK_K - 1) / QK_K;

        Span<byte> scaleBuf = stackalloc byte[16];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;

            // block_q3_K: hmask[32] + qs[64] + scales[12] + d[2] = 110 bytes (no dmin)
            byte[] qh = reader.ReadBytes(32);    // high 1-bit per value
            byte[] qs = reader.ReadBytes(64);    // low 2 bits per value
            byte[] scalesRaw = reader.ReadBytes(12);
            ushort dBits = reader.ReadUInt16();   // d at byte offset 108
            float dAll = HalfToFloat(dBits);

            // Scale buffer: only 12 scale bytes needed; aux[3] is overwritten by bit-unpack
            for (int j = 0; j < 12; j++) scaleBuf[j] = scalesRaw[j];
            // bytes 12-15 are unused (DecodeQ3KScales overwrites all 16 bytes)

            int[] sc = DecodeQ3KScales(scaleBuf);

            for (int i = 0; i < QK_K && blockStart + i < n; i++)
            {
                // qs transposed: byte = (i/128)*32 + i%32, shift = ((i%128)/32)*2
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (qh[i % 32] >> (i / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                int sub = i / 16;
                float val = dAll * sc[sub] * actual;
                
                if (float.IsNaN(val) || float.IsInfinity(val)) val = 0f;
                data[blockStart + i] = val;
            }
        }
    }

    private static unsafe int[] DecodeQ3KScales(Span<byte> buf16)
    {
        var sc = new int[16];
        fixed (byte* p = buf16)
        {
            uint* aux = (uint*)p;
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
            sbyte* sc8 = (sbyte*)p;
            for (int j = 0; j < 16; j++) {
                sc[j] = sc8[j] - 32;
            }
        }
        return sc;
    }
}
