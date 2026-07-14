using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text.RegularExpressions;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;

public sealed class GgufLoader(QuantizationOps qOps, string path, ModelConfig config) : IModelLoader
{
    private const uint Magic = 0x46554747;
    private readonly QuantizationOps _qOps = qOps ?? throw new ArgumentNullException(nameof(qOps));
    private readonly string _path = File.Exists(path)? path : throw new FileNotFoundException(path);
    private readonly ModelConfig _config = config ?? throw new ArgumentNullException(nameof(config));
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
            catch { break; }
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

    private static void InjectMissingTemplateTokens(
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

    public static void Load(
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

    // ── IModelLoader implementation ────────────────────────────────────────

    public void PreInit(TransformerWeights weights, IProgress<float>? progress = null)
    {
        Core.Memory.NativeBufferPool<float>.Clear();

        var meta = LoadMeta(_path);
        weights.GgufMeta = meta;
        weights.GgufPath = _path;
        weights.IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));

        int total = meta.Tensors.Count;
        int processed = 0;

        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)processed / total);

            var (target, block, rawField) = weights.ResolveTarget(info.Name);
            if (target == null && block == null) { processed++; continue; }

            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);

            // Record top-level tensor dtype (embedding, lm_head)
            if (target != null && block == null && rawSize > 0)
            {
                if (target == weights.LmHeadWeight)
                    weights.RawLmHeadDtype = info.Dtype;
                else if (target == weights.EmbeddingWeight)
                    weights.RawEmbeddingDtype = info.Dtype;
            }

            // Record tensor metadata for block-level weights
            if (block != null && rawField != null && rawSize > 0)
            {
                SetTensorMeta(block, rawField, meta.DataOffset + info.Offset, (int)rawSize, info.Dtype);
            }

            processed++;
        }
        progress?.Report(1f);
    }

    public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null)
    {
        var meta = weights.GgufMeta ?? LoadMeta(_path);
        int total = meta.Tensors.Count;
        int loaded = 0;

        using var mmf = MemoryMappedFile.CreateFromFile(_path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        foreach (var info in meta.Tensors)
        {
            progress?.Report((float)loaded / total);
            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) { loaded++; continue; }
            stream.Position = targetOffset;

            if (!info.Name.Contains("blk.") && info.Name.Contains("output.weight") && weights.LmHeadWeight == null)
            {
                long ggufIn = info.Shape[0];
                weights.SetLmHead(new Tensor<float>((int)_config.VocabSize, (int)ggufIn));
            }

            var (target, block, rawField) = weights.ResolveTarget(info.Name);
            if (target == null && block == null) { loaded++; continue; }

            int count = 1;
            foreach (int d in info.Shape) count *= d;
            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);

            // Load raw quantized data for block-level tensors
            if (block != null && rawField != null && rawSize > 0 && stream.Position + rawSize <= stream.Length)
            {
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
                }
                else
                {
                    rawData = new byte[rawSize];
                    stream.ReadExactly(rawData);
                    stream.Position -= rawSize;
                }

                if (target == weights.LmHeadWeight)
                {
                    //bool isBadLayout = info.Dtype is QuantDType.Q8_0 or QuantDType.Q5_0
                    //    or QuantDType.Q6_K or QuantDType.Q6_K_S;
                    //if (!isBadLayout)
                    {
                        weights.RawLmHead = rawData;
                        weights.RawLmHeadDtype = info.Dtype;
                    }
                }
                else if (target == weights.EmbeddingWeight)
                {
                    weights.RawEmbedding = rawData;
                    weights.RawEmbeddingDtype = info.Dtype;
                }
            }

            // Dequantize to float
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
    }

    public void LoadLayer(TransformerWeights weights, int layerIndex)
    {
        var meta = weights.GgufMeta ?? LoadMeta(_path);
        string prefix = $"blk.{layerIndex}.";
        var layerTensors = meta.Tensors
            .Where(t => t.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .ToList();
        if (layerTensors.Count == 0) return;

        using var mmf = MemoryMappedFile.CreateFromFile(_path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        var block = weights.Blocks[layerIndex];

        foreach (var info in layerTensors)
        {
            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;
            stream.Position = targetOffset;

            var (_, _, rawField) = weights.ResolveTarget(info.Name);
            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
            if (rawSize <= 0) continue;

            bool isQuantizedWeight = rawField != null;

            if (isQuantizedWeight)
            {
                byte[] rawData = new byte[rawSize];
                stream.ReadExactly(rawData);
                stream.Position -= rawSize;
                SetRawField(block, rawField!, rawData, info.Dtype);
            }

            int count = 1;
            foreach (int d in info.Shape) count *= d;
            float[] buffer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));
                var floatTarget = weights.ResolveFloatTarget(info.Name);
                if (floatTarget != null)
                {
                    if (floatTarget.ElementCount != count)
                    {
                        var newTensor = info.Shape.Length == 2
                            ? new Tensor<float>(info.Shape[0], info.Shape[1])
                            : new Tensor<float>(info.Shape);
                        SetBlockTensor(block, rawField, info.Name, newTensor);
                        floatTarget = newTensor;
                    }
                    floatTarget.Data.Clear();
                    buffer.AsSpan(0, count).CopyTo(floatTarget.Data);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
    }

    private static void SetBlockTensor(BlockWeights block, string? rawField, string name, Tensor<float> tensor)
    {
        if (rawField != null)
        {
            SetBlockFloatTensor(block, rawField, tensor);
            return;
        }

        if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            //if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            //    block.Norm1B = tensor;
            //else
                block.Norm1W = tensor;
            return;
        }
        if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            //if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            //    block.Norm2B = tensor;
            //else
                block.Norm2W = tensor;
            return;
        }
        if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) ||
                name.Contains("q_proj", StringComparison.OrdinalIgnoreCase))
                block.WqBias = tensor;
            else if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("k_proj", StringComparison.OrdinalIgnoreCase))
                block.WkBias = tensor;
            else if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
                block.WvBias = tensor;
            else if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("o_proj", StringComparison.OrdinalIgnoreCase))
                block.WoBias = tensor;
            else if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) ||
                     name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                block.Wf1Bias = tensor;
            else if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                block.Wf2Bias = tensor;
            return;
        }
    }

    public void LoadTopLevelTensors(TransformerWeights weights)
    {
        var meta = weights.GgufMeta ?? LoadMeta(_path);

        using var mmf = MemoryMappedFile.CreateFromFile(_path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        foreach (var info in meta.Tensors)
        {
            if (info.Name.Contains("blk.")) continue;

            // Dynamic lm_head allocation (matching LoadAllWeights behavior)
            if (info.Name.Contains("output.weight") && weights.LmHeadWeight == null)
            {
                long ggufIn = info.Shape[0];
                weights.SetLmHead(new Tensor<float>((int)weights.Config.VocabSize, (int)ggufIn));
            }

            var (target, block, _) = weights.ResolveTarget(info.Name);
            if (target == null || block != null) continue;

            long targetOffset = meta.DataOffset + info.Offset;
            if (targetOffset >= stream.Length) continue;
            stream.Position = targetOffset;

            int count = 1;
            foreach (int d in info.Shape) count *= d;
            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
            if (rawSize <= 0) continue;

            // Read raw quantized data with vocab padding
            byte[] rawData;
            if (info.Shape.Length >= 2)
            {
                long tensorVocab = Math.Max(info.Shape[0], info.Shape[1]);
                long paddedVocab = weights.Config.VocabSize;
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
            }
            else
            {
                rawData = new byte[rawSize];
                stream.ReadExactly(rawData);
                stream.Position -= rawSize;
            }

            if (target == weights.LmHeadWeight)
            {
                //bool isBadLayout = info.Dtype is QuantDType.Q8_0 or QuantDType.Q5_0
                 //   or QuantDType.Q6_K or QuantDType.Q6_K_S;
                //if (!isBadLayout)
                {
                    weights.RawLmHead = rawData;
                    weights.RawLmHeadDtype = info.Dtype;
                }
            }
            else if (target == weights.EmbeddingWeight)
            {
                weights.RawEmbedding = rawData;
                weights.RawEmbeddingDtype = info.Dtype;
            }

            // Dequantize to float
            float[] buffer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                ReadTensorInto(reader, info.Dtype, info.Shape, buffer.AsSpan(0, count));
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
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
    }

    private void ReadTensorInto(BinaryReader stream, QuantDType dtype, int[] shape, Span<float> destination)
    {
        int count = 1;
        foreach (int d in shape) count *= d;
        if (destination.Length < count) throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");
        _qOps.ReadFor(dtype, stream, destination, count);
    }

    /// <summary>Sets the float tensor property on a BlockWeights instance corresponding to the raw field name.</summary>
    private static void SetBlockFloatTensor(BlockWeights block, string rawField, Tensor<float> tensor)
    {
        switch (rawField)
        {
            case "RawWq": block.Wq = tensor; break;
            case "RawWk": block.Wk = tensor; break;
            case "RawWv": block.Wv = tensor; break;
            case "RawWo": block.Wo = tensor; break;
            case "RawWf1": block.Wf1 = tensor; break;
            case "RawWf2": block.Wf2 = tensor; break;
        }
    }
}
