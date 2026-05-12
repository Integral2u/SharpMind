using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;

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
            object? val;

            switch (valType)
            {
                case 8: // STRING
                    val = ReadStringValue(reader);
                    break;
                case 9: // ARRAY � read all array types; tokenizer vocab lives here
                    val = ReadArrayValue(reader);
                    break;
                default:
                    val = ReadValue(reader, valType);
                    break;
            }

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

        // Prefer the explicit vocab_size key over the embedding tensor dimension,
        // as some models (e.g. LLaMA-3) pad the embedding table to a round number.
        int metaVocab = (int)meta.GetLong($"{arch}.vocab_size",
                         meta.GetLong("tokenizer.ggml.token_count",
                         meta.GetLong("vocab_size", vocabSize)));
        if (metaVocab > 0) vocabSize = metaVocab;

        return new ModelConfig
        {
            VocabSize = vocabSize,
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            FfnDim = ffnDim,
            MaxSeqLen = maxSeqLen,
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
        if (tokens == null || tokens.Length == 0)
        {
            Console.WriteLine("[GgufLoader] No tokenizer.ggml.tokens found in GGUF � tokenizer must be loaded from file.");
            return null;
        }

        var scores = GetFloatArray(meta, "tokenizer.ggml.scores");
        var types = GetIntArray(meta, "tokenizer.ggml.token_type");
        var merges = GetStringArray(meta, "tokenizer.ggml.merges");

        int bosId = (int)meta.GetLong("tokenizer.ggml.bos_token_id", 1);
        int eosId = (int)meta.GetLong("tokenizer.ggml.eos_token_id", 2);

        string tokModel = meta.GetString("tokenizer.ggml.model") ?? "bpe";
        Console.WriteLine($"[GgufLoader] Building tokenizer from GGUF: model={tokModel}, vocab={tokens.Length}, bos={bosId}, eos={eosId}");

        try
        {
            return Tokenizer.FromGguf(tokens, merges, scores, types, bosId, eosId);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GgufLoader] Tokenizer.FromGguf failed: {ex.Message}");
            return null;
        }
    }

    // ?? Main entry point ?????????????????????????????????????????????????

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

        Console.WriteLine($"[GgufLoader] Config: vocab={config.VocabSize}, hidden={config.HiddenDim}, " +
                          $"layers={config.NumLayers}, heads={config.NumHeads}, kvHeads={config.NumKvHeads}");

        // Prefer GGUF-embedded tokenizer � its vocab size is guaranteed to match the weights.
        tokenizer = LoadTokenizerFromMeta(meta);

        // Fall back to file only when GGUF has no vocab data.
        if (tokenizer == null && !string.IsNullOrEmpty(tokenizerPath) && File.Exists(tokenizerPath))
        {
            Console.WriteLine($"[GgufLoader] Falling back to tokenizer file: {tokenizerPath}");
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
            catch (Exception ex)
            {
                Console.WriteLine($"[GgufLoader] File tokenizer load failed: {ex.Message}");
                tokenizer = null;
            }
        }
    }

    // ?? Weight loading ????????????????????????????????????????????????????

    public static Dictionary<string, Tensor<float>> LoadWeights(string path)
    {
        var meta = LoadMeta(path);
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var result = new Dictionary<string, Tensor<float>>();

        foreach (var info in meta.Tensors)
        {
            stream.Position = info.Offset;
            var tensor = ReadTensor(reader, info.Dtype, info.Shape);
            result[info.Name] = tensor;
        }
        return result;
    }

    public static void LoadWeightsToModel(string path, GgufMeta meta, Transformer model)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        int loaded = 0, missing = 0, total = meta.Tensors.Count;

        foreach (var info in meta.Tensors)
        {
            long targetOffset = info.Offset;
            if (targetOffset >= stream.Length) continue;

            stream.Position = targetOffset;

            int count = 1;
            foreach (int d in info.Shape) count *= d;

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

        Console.WriteLine($"[GgufLoader] Loaded weights: {loaded}/{total} tensors (missing: {missing})");
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
                case GgufDtype.Q4_K: ReadQ4K(stream, destination, count); break;
                case GgufDtype.Q6_K: ReadQ6K(stream, destination, count); break;
                case GgufDtype.Q8_0: ReadQ8_0(stream, destination, count); break;
                case GgufDtype.Q5_K: ReadQ5_K(stream, destination, count); break;
                case GgufDtype.Q3_K: ReadQ3_K(stream, destination, count); break;
                case GgufDtype.Q5_0: ReadQ5_0(stream, destination, count); break;
            }
        }
        catch { /* partial tensor � leave zeros */ }
    }

    private static float HalfToFloat(ushort half)
    {
        int sign = (half >> 15) & 0x1;
        int exp = (half >> 10) & 0x1F;
        int mant = half & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            // Denormal
            float val = mant / 1024f;
            return (sign == 0 ? 1f : -1f) * val * MathF.Pow(2f, -14f);
        }
        if (exp == 31)
            return mant == 0
                ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity)
                : float.NaN;

        return (sign == 0 ? 1f : -1f) * MathF.Pow(2f, exp - 15) * (1f + mant / 1024f);
    }

    private static void ReadQ4K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        int nBlocks = (n + blockSize - 1) / blockSize;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, n);
            int blockCount = blockEnd - blockStart;

            var scales = new float[2];
            scales[0] = reader.ReadSingle();
            scales[1] = reader.ReadSingle();

            var mins = new float[2];
            mins[0] = reader.ReadSingle();
            mins[1] = reader.ReadSingle();

            for (int i = 0; i < blockCount; i++)
            {
                int q = i / 16;
                byte qb = reader.ReadByte();
                int low = (qb & 0x0F) - 8;
                data[blockStart + i] = low * scales[q] + mins[q];
            }
        }
    }

    private static void ReadQ6K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 64;
        int nBlocks = (n + blockSize - 1) / blockSize;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * blockSize;
            int blockEnd = Math.Min(blockStart + blockSize, n);
            int blockCount = blockEnd - blockStart;

            var scale = reader.ReadSingle();
            var q6 = new byte[blockCount];
            for (int i = 0; i < blockCount; i++) q6[i] = reader.ReadByte();
            for (int i = 0; i < blockCount; i++)
                data[blockStart + i] = (q6[i] - 32) * 0.25f * scale;
        }
    }

    private static void ReadQ8_0(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            var scale = reader.ReadSingle();
            for (int j = 0; j < blockSize && i + j < n; j++)
                data[i + j] = (reader.ReadByte() - 128) * scale;
        }
    }

    private static void ReadQ3_K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 64;
        for (int i = 0; i < n; i += blockSize)
        {
            float d1 = reader.ReadSingle();
            float d2 = reader.ReadSingle();

            int count = Math.Min(blockSize, n - i);
            int half = count / 2;

            for (int j = 0; j < count; j++)
            {
                if (j == 0 || j == half)
                {
                    byte bv = reader.ReadByte();
                    for (int k = 0; k < 8 && j + k < count && (j < half ? k < 8 : k < 8 + half - 16); k++)
                    {
                        int idx = j + k;
                        int qval = j < half ? ((bv >> k) & 0x7) : ((bv >> (k - half + 16)) & 0x7);
                        data[i + idx] = (qval - 4) * (idx < half ? d1 : d2);
                    }
                }
            }
        }
    }

    private static void ReadQ5_K(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            var scales = new[] {
                reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(),
                reader.ReadSingle(), reader.ReadSingle()
            };

            int count = Math.Min(blockSize, n - i);
            int groupSize = (count + 4) / 5;

            for (int j = 0; j < count; j++)
            {
                int groupIdx = Math.Min(j / groupSize, 4);
                if (j % 2 == 0)
                {
                    byte bv = reader.ReadByte();
                    data[i + j] = ((bv & 0x1F) - 16) * scales[groupIdx];
                }
                else if (j + 1 < count)
                {
                    byte bv = reader.ReadByte();
                    data[i + j] = (((bv >> 4) & 0x1F) - 16) * scales[groupIdx];
                }
            }
        }
    }

    private static void ReadQ5_0(BinaryReader reader, Span<float> data, int n)
    {
        int blockSize = 32;
        for (int i = 0; i < n; i += blockSize)
        {
            float d = HalfToFloat(reader.ReadUInt16());
            uint qh = reader.ReadUInt32();

            for (int j = 0; j < blockSize && i + j < n; j++)
            {
                sbyte qs = reader.ReadSByte();
                int q = (int)qs | (int)(((qh >> j) & 1) << 4);
                data[i + j] = q * d;
            }
        }
    }
}
