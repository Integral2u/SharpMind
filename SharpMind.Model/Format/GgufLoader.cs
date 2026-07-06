using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text.RegularExpressions;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;

public sealed class GgufLoader
{
    private const uint Magic = 0x46554747;
    private readonly QuantizationOps _qOps;

    public GgufLoader(QuantizationOps qOps)
    {
        _qOps = qOps ?? throw new ArgumentNullException(nameof(qOps));
    }

    private object ReadValue(BinaryReader reader, uint valType)
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

    private (ulong len, string str) ReadString(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        if (len > 10000) return (len, "");
        var bytes = reader.ReadBytes((int)len);
        return (len, System.Text.Encoding.UTF8.GetString(bytes));
    }

    private string ReadStringValue(BinaryReader reader)
    {
        var len = reader.ReadUInt64();
        var bytes = reader.ReadBytes((int)len);
        return System.Text.Encoding.UTF8.GetString(bytes);
    }

    private object? ReadArrayValue(BinaryReader reader)
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

    public string[]? GetStringArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as string[];

    public float[]? GetFloatArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as float[];

    public int[]? GetIntArray(ModelMetaData meta, string key)
        => meta.KvPairs.FirstOrDefault(p => p.Key == key).Value as int[];

    public ModelMetaData LoadMeta(string path)
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

                var dtype = (GgufDtype)reader.ReadUInt32();
                var offset = reader.ReadUInt64();

                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = (long)offset });
            }
            catch { break; }
        }

        uint alignment = (uint)meta.GetLong("general.alignment", 32);
        long pos = stream.Position;
        meta.DataOffset = (pos + alignment - 1) & ~(alignment - 1);

        return meta;
    }

    public ModelConfig? LoadConfig(ModelMetaData meta)
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

        long rawKeyLen = meta.GetLong($"{arch}.attention.key_length", -1);
        int? keyLength = rawKeyLen > 0 ? (int)rawKeyLen : null;
        long rawValLen = meta.GetLong($"{arch}.attention.value_length", -1);
        int? valueLength = rawValLen > 0 ? (int)rawValLen : null;

        numLayers = (int)meta.GetLong($"{arch}.block_count", numLayers);

        float ropeTheta = meta.GetFloat($"{arch}.rope.freq_base",
                          meta.GetFloat("rope_theta",
                          meta.GetFloat("rope.freq_base", 10_000f)));

        int metaVocab = (int)meta.GetLong($"{arch}.vocab_size",
                         meta.GetLong("tokenizer.ggml.token_count",
                         meta.GetLong("vocab_size", vocabSize)));
        if (metaVocab > 0) vocabSize = metaVocab;

        long rawHeadDim = meta.GetLong($"{arch}.head_dim", -1);
        int? headDimOverride = rawHeadDim > 0 ? (int)rawHeadDim : null;

        long rawRopeDim = meta.GetLong($"{arch}.rope.dimension_count", -1);
        int? ropeDim = rawRopeDim > 0 ? (int)rawRopeDim : null;

        string? ropeScalingType = meta.GetString($"{arch}.rope.scaling.type");
        float ropeFactor = meta.GetFloat($"{arch}.rope.scaling.factor", float.NaN);
        float? ropeScalingFactor = float.IsNaN(ropeFactor) ? null : ropeFactor;
        long rawRopeOrigCtx = meta.GetLong($"{arch}.rope.scaling.original_context_length", -1);
        int? ropeOriginalContextLength = rawRopeOrigCtx > 0 ? (int)rawRopeOrigCtx : null;

        long rawTie = meta.GetLong($"{arch}.tie_word_embeddings", -1);
        bool? tieWordEmbeddings = rawTie >= 0 ? (rawTie != 0) : null;

        long rawNormType = meta.GetLong($"{arch}.norm_type", -1);
        int? normTypeOverride = rawNormType >= 0 ? (int)rawNormType : null;

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

    public Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta)
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

    private void InjectMissingTemplateTokens(
        ModelMetaData meta, ref ModelConfig config, Tokenizer tokenizer)
    {
        string? template = meta.GetChatTemplate();
        if (string.IsNullOrEmpty(template)) return;

        var candidates = new HashSet<string>();
        foreach (Match m in RegexGenerated.ChatTemplateRegex.Matches(template))
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

    public void Load(
        string ggufPath,
        string? tokenizerPath,
        out ModelMetaData meta,
        out ModelConfig config,
        out Tokenizer? tokenizer)
    {
        meta = LoadMeta(ggufPath);
        config = LoadConfig(meta)!;

        tokenizer = LoadTokenizerFromMeta(meta);

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
                else if (arch.Contains("mistral", StringComparison.OrdinalIgnoreCase))
                {
                    tokenizer = Tokenizer.FromMistral(tokenizerPath);
                }
                else
                {
                    tokenizer = Tokenizer.FromFile(tokenizerPath);
                }
            }
            catch
            {
                tokenizer = null;
            }
        }

        if (tokenizer != null)
            InjectMissingTemplateTokens(meta, ref config, tokenizer);
    }

    public TransformerWeights LoadWeightsToTransformerWeights(string path, ModelConfig config, IProgress<float>? progress = null, LoadMode mode = LoadMode.Realtime)
    {
        SharpMind.Core.Memory.NativeBufferPool<float>.Clear();

        var meta = LoadMeta(path);
        var weights = ModelFactory.CreateWeights(config, config.ForModel(HardwareTier.Auto));
        weights.GgufMeta = meta;
        weights.GgufPath = path;

        weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));

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

            if (!info.Name.Contains("blk.") && info.Name.Contains("output.weight") && weights.LmHeadWeight == null)
            {
                long ggufIn = info.Shape[0];
                weights.SetLmHead(new Tensor<float>((int)config.VocabSize, (int)ggufIn));
            }

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
                    stream.Position -= rawSize;

                    SetRawField(block, rawField, rawData, info.Dtype);
                    loaded++;
                    if (mode != LoadMode.Full)
                        continue;
                }
            }

            if (IsQuantizedType(info.Dtype) && info.Shape.Length >= 2 && target != null && block == null)
            {
                long rawSize = GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0 && stream.Position + rawSize <= stream.Length)
                {
                    byte[] rawData;
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
                    bool isBadLayout = info.Dtype is GgufDtype.Q8_0 or GgufDtype.Q5_0
                        or GgufDtype.Q6_K or GgufDtype.Q6_K_S;
                    if (target == weights.LmHeadWeight && !isBadLayout)
                    {
                        weights.RawLmHead = rawData;
                        weights.RawLmHeadDtype = info.Dtype;
                    }
                    else if (target != weights.LmHeadWeight && !isBadLayout)
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

    public Dictionary<string, Tensor<float>> LoadWeights(string path, IProgress<float>? progress = null)
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

    internal bool IsQuantizedType(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q2_K or GgufDtype.Q3_K or GgufDtype.Q4_K or GgufDtype.Q5_K or GgufDtype.Q6_K
        or GgufDtype.Q2_K_S or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L
        or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M or GgufDtype.Q6_K_S
        or GgufDtype.Q4_0 or GgufDtype.Q4_1 or GgufDtype.Q5_0 or GgufDtype.Q5_1
        or GgufDtype.Q8_0 or GgufDtype.Q8_1 or GgufDtype.Q8_K
        or GgufDtype.IQ4_NL => true,
        _ => false
    };

    internal long GetRawTensorByteCount(int[] shape, GgufDtype dtype)
    {
        long totalElements = 1;
        foreach (int d in shape) totalElements *= d;

        switch (dtype)
        {
            case GgufDtype.F32: return totalElements * 4;
            case GgufDtype.F16: return totalElements * 2;
            case GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L: return ((totalElements + 255) / 256) * 110;
            case GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M: return ((totalElements + 255) / 256) * 144;
            case GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M: return ((totalElements + 255) / 256) * 176;
            case GgufDtype.Q6_K or GgufDtype.Q6_K_S: return ((totalElements + 255) / 256) * 210;
            case GgufDtype.Q2_K or GgufDtype.Q2_K_S: return ((totalElements + 255) / 256) * 84;
            case GgufDtype.Q8_K: return ((totalElements + 255) / 256) * 292;
            case GgufDtype.Q8_0: return ((totalElements + 31) / 32) * 34;
            case GgufDtype.Q8_1: return ((totalElements + 31) / 32) * 36;
            case GgufDtype.Q5_0: return ((totalElements + 31) / 32) * 22;
            case GgufDtype.Q5_1: return ((totalElements + 31) / 32) * 24;
            case GgufDtype.Q4_0: return ((totalElements + 31) / 32) * 18;
            case GgufDtype.IQ4_NL: return ((totalElements + 31) / 32) * 18;
            case GgufDtype.Q4_1: return ((totalElements + 31) / 32) * 20;
            default: return 0;
        }
    }

    private Tensor<float> ReadTensor(BinaryReader stream, GgufDtype dtype, int[] shape)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        var result = new Tensor<float>(shape);
        ReadTensorInto(stream, dtype, shape, result.Data);
        return result;
    }

    internal void ReadQBlockRow(BinaryReader stream, GgufDtype dtype, Span<float> dest, int count)
    {
        switch (dtype)
        {
            case GgufDtype.Q4_0: ReadQ4_0(stream, dest, count); break;
            case GgufDtype.IQ4_NL: ReadQ4_NL(stream, dest, count); break;
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

    internal void ReadTensorInto(BinaryReader stream, GgufDtype dtype, int[] shape, Span<float> destination)
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
                for (int i = 0; i < count; i++) destination[i] = _qOps.HalfToFloat(stream.ReadUInt16());
                break;
            default:
                ReadQBlockRow(stream, dtype, destination, count);
                break;
        }
    }

    public float HalfToFloat(ushort half) => _qOps.HalfToFloat(half);

    public void ReadQ8_0(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ8_0(reader, data, n);

    public void ReadQ4_1(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ4_1(reader, data, n);

    public void ReadQ5_1(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ5_1(reader, data, n);

    public void ReadQ8_1(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ8_1(reader, data, n);

    public void ReadQ2K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ2K(reader, data, n);

    public void ReadQ8K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ8K(reader, data, n);

    public void ReadQ5_0(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ5_0(reader, data, n);

    public void ReadQ4_0(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ4_0(reader, data, n);

    public void ReadQ4_NL(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ4_NL(reader, data, n);

    public void ReadQ4K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ4K(reader, data, n);

    public void ReadQ6K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ6K(reader, data, n);

    public void ReadQ5_K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ5K(reader, data, n);

    public void ReadQ3_K(BinaryReader reader, Span<float> data, int n)
        => _qOps.ReadQ3K(reader, data, n);
}
