using SharpMind.Core.Diagnostics;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using static SharpMind.Model.TransformerWeights;

namespace SharpMind.Model.Format;

/// <summary>
/// Loads SharpMind Model (.SMM) containers. Mirrors <see cref="GgufLoader"/>'s
/// target resolution, raw-quantized-data handling and GGUF transpose semantics
/// so a single loader serves both GGUF-converted and training-exported files.
/// The only differences are the container header/index.
/// </summary>
public sealed class SmmLoader(QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false) : IModelLoader
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    private readonly QuantizationOps _qOps = qOps ?? throw new ArgumentNullException(nameof(qOps));
    private readonly string _path = File.Exists(path) ? path : throw new FileNotFoundException(path);
    private readonly ModelConfig _config = config ?? throw new ArgumentNullException(nameof(config));
    private readonly bool _useSafeIo = useSafeIo;

    // ── Static helpers (metadata / config / tokenizer / plugins) ──────────

    public static ModelMetaData LoadMeta(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        return ReadIndex(reader, stream).Meta;
    }

    public static ModelConfig? LoadConfig(ModelMetaData meta)
    {
        string json = meta.GetString(SmmConstants.ConfigKey);
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            return JsonSerializer.Deserialize<ModelConfig>(json, JsonOpts);
        }
        catch (Exception ex)
        {
            SanityChecks.WriteLine($"SmmLoader: config JSON parse failed: {ex.Message}");
            return null;
        }
    }

    public static Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta)
    {
        string json = meta.GetString(SmmConstants.TokenizerKey);
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            return Tokenizer.FromJson(json);
        }
        catch (Exception ex)
        {
            SanityChecks.WriteLine($"SmmLoader: tokenizer JSON parse failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>Returns the embedded default system prompt, or <see langword="null"/> when absent.</summary>
    public static string? LoadSystemPromptFromMeta(ModelMetaData meta)
    {
        string value = meta.GetString(SmmConstants.SystemPromptKey);
        return string.IsNullOrWhiteSpace(value) ? null : value;
    }

    /// <summary>Returns the embedded skills (markdown documents), or an empty list when absent.</summary>
    public static List<string> LoadSkillsFromMeta(ModelMetaData meta)
    {
        string json = meta.GetString(SmmConstants.SkillsKey);
        if (string.IsNullOrWhiteSpace(json)) return [];
        try
        {
            var list = JsonSerializer.Deserialize<List<string>>(json);
            return list ?? [];
        }
        catch (Exception ex)
        {
            SanityChecks.WriteLine($"SmmLoader: skills JSON parse failed: {ex.Message}");
            return [];
        }
    }

    public static string? LoadSystemPrompt(string path) => LoadSystemPromptFromMeta(LoadMeta(path));

    public static List<string> LoadSkills(string path) => LoadSkillsFromMeta(LoadMeta(path));

    public static List<SmmPluginEntry> LoadPlugins(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        if (reader.ReadUInt32() != SmmConstants.Magic)
            throw new InvalidDataException("Not SMM: " + path);

        reader.ReadUInt32(); // version
        long metaLen = reader.ReadInt64();
        long tokenizerLen = reader.ReadInt64();
        long pluginAsmCount = reader.ReadInt64();
        reader.ReadInt64(); // tensorCount
        reader.ReadInt64(); // indexLen
        reader.ReadInt64(); // dataOffset
        reader.ReadInt64(); // reserved

        reader.BaseStream.Position += metaLen + tokenizerLen;

        var plugins = new List<SmmPluginEntry>((int)pluginAsmCount);
        for (long i = 0; i < pluginAsmCount; i++)
        {
            var (_, name) = ReadString(reader);
            bool recommended = reader.ReadBoolean();
            long asmLen = reader.ReadInt64();
            byte[] asm = reader.ReadBytes((int)asmLen);
            plugins.Add(new SmmPluginEntry { Name = name, AssemblyBytes = asm, Recommended = recommended });
        }
        return plugins;
    }

    /// <summary>
    /// Reads the tensor index of an .SMM container. Exposed so converters can
    /// stream raw tensor bytes out of the container (see <see cref="ReadTensorBytes"/>).
    /// </summary>
    public static List<SmmTensorIndexEntry> ReadTensorIndex(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException(path);
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        return ReadIndex(reader, stream).Entries;
    }

    /// <summary>
    /// Reads the raw bytes of a single tensor from an .SMM container — exactly
    /// the bytes GGUF would store on disk.
    /// </summary>
    public static byte[] ReadTensorBytes(string path, SmmTensorIndexEntry entry, long rawSize)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException(path);

        using var stream = File.OpenRead(path);

        // Re-derive the data offset from the header so converters don't need to
        // load the full index twice.
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: false);
        if (reader.ReadUInt32() != SmmConstants.Magic)
            throw new InvalidDataException("Not SMM: " + path);
        reader.ReadUInt32(); // version
        reader.ReadInt64();  // metaLen
        reader.ReadInt64();  // tokenizerLen
        reader.ReadInt64();  // pluginAsmCount
        reader.ReadInt64();  // tensorCount
        reader.ReadInt64();  // indexLen
        long dataOffset = reader.ReadInt64();

        long absolute = dataOffset + entry.Offset;
        if (absolute < 0 || absolute + rawSize > stream.Length)
            throw new InvalidDataException($"Tensor '{entry.Name}' range is beyond end of file.");
        stream.Position = absolute;

        var bytes = new byte[rawSize];
        stream.ReadExactly(bytes);
        return bytes;
    }

    public static void Load(
        string path,
        string? tokenizerPath,
        out ModelMetaData meta,
        out ModelConfig config,
        out Tokenizer? tokenizer)
    {
        meta = LoadMeta(path);
        config = LoadConfig(meta)
            ?? throw new InvalidDataException("SMM file is missing its model config (smm.config_json).");
        tokenizer = LoadTokenizerFromMeta(meta);
        if (tokenizer == null && !string.IsNullOrEmpty(tokenizerPath) && File.Exists(tokenizerPath))
        {
            try
            {
                tokenizer = Tokenizer.FromFile(tokenizerPath);
            }
            catch (Exception ex)
            {
                SanityChecks.WriteLine($"SmmLoader: external tokenizer file failed: {ex.Message}");
                tokenizer = null;
            }
        }
    }

    // ── IModelLoader implementation ────────────────────────────────────────

    public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null)
    {
        Core.Memory.NativeBufferPool<float>.Clear();

        var index = ReadIndex(_path);
        weights.GgufMeta = index.Meta;
        weights.GgufPath = _path;
        weights.IsMoE = index.Meta.Tensors.Any(t => t.Name.Contains(".exps."));

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);

        int total = index.Entries.Count;
        int loaded = 0;
        foreach (var entry in index.Entries)
        {
            progress?.Report((float)loaded / total);
            LoadSingleTensor(weights, index, stream, entry);
            loaded++;
        }
        progress?.Report(1f);
    }

    public void LoadLayerWeights(int layerIndex, TransformerWeights weights)
    {
        var index = weights.GgufMeta == null ? ReadIndex(_path) : ReadIndex(_path);
        if (weights.GgufMeta == null)
        {
            weights.GgufMeta = index.Meta;
            weights.GgufPath = _path;
            weights.IsMoE = index.Meta.Tensors.Any(t => t.Name.Contains(".exps."));
        }

        var targetBlock = layerIndex < weights.Blocks.Length ? weights.Blocks[layerIndex] : null;
        if (targetBlock == null) return;

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);

        foreach (var entry in index.Entries)
        {
            var (_, block, _) = weights.ResolveTarget(entry.Name);
            if (block == targetBlock)
                LoadSingleTensor(weights, index, stream, entry);
        }
    }

    public void LoadGlobalTensors(TransformerWeights weights)
    {
        var index = ReadIndex(_path);
        if (weights.GgufMeta == null)
        {
            weights.GgufMeta = index.Meta;
            weights.GgufPath = _path;
            weights.IsMoE = index.Meta.Tensors.Any(t => t.Name.Contains(".exps."));
        }

        using var stream = WeightStreamFactory.Open(_path, _useSafeIo);

        foreach (var entry in index.Entries)
        {
            var (target, block, _) = weights.ResolveTarget(entry.Name);
            if (target != null && block == null)
                LoadSingleTensor(weights, index, stream, entry);
        }
    }

    private void LoadSingleTensor(
        TransformerWeights weights, SmmFileIndex index,
        Stream stream, SmmTensorIndexEntry entry)
    {
        var (target, block, rawField) = weights.ResolveTarget(entry.Name);

        // Must create LmHeadWeight BEFORE the early-return check below (see
        // GgufLoader for the rationale).
        if (!entry.Name.Contains("blk.") && entry.Name.Contains("output.weight") && weights.LmHeadWeight == null)
        {
            // The input dim is whichever shape entry is not the vocab size — our own
            // exports declare [vocab, in], canonical GGUF-derived files [in, vocab].
            long ggufIn = entry.Shape[0];
            if (entry.Shape.Length > 1 && ggufIn == _config.VocabSize) ggufIn = entry.Shape[1];
            int lmRows = TensorLoadHelper.CheckedInt(_config.VocabSize, "VocabSize for LmHead");
            int lmCols = TensorLoadHelper.CheckedInt(ggufIn, "LmHead input dim");
            weights.SetLmHead(new Tensor<float>(lmRows, lmCols));
            (target, block, rawField) = weights.ResolveTarget(entry.Name);
        }

        if (target == null && block == null) return;

        long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
        if (rawSize <= 0) return;

        byte[] rawBytes = ReadTensorBytes(stream, index, entry, rawSize);

        // Record tensor metadata and top-level dtypes (consumed by SetWeights later)
        if (target != null && block == null)
        {
            if (target == weights.LmHeadWeight) weights.RawLmHeadDtype = entry.Dtype;
            else if (target == weights.EmbeddingWeight) weights.RawEmbeddingDtype = entry.Dtype;
        }
        if (block != null && rawField != null)
            SetTensorMeta(block, rawField, index.Meta.DataOffset + entry.Offset, TensorLoadHelper.CheckedInt(rawSize, "rawSize"), entry.Dtype);

        // Load raw quantized data
        if (block != null && rawField != null)
            SetRawField(block, rawField, rawBytes, entry.Dtype);

        if (target != null && block == null)
        {
            byte[] data = rawBytes;
            if (entry.Shape.Length >= 2)
            {
                long tensorVocab = Math.Max(entry.Shape[0], entry.Shape[1]);
                long paddedVocab = _config.VocabSize;
                if (paddedVocab > tensorVocab)
                {
                    long colBytes = rawSize / tensorVocab;
                    int safeColBytes = TensorLoadHelper.CheckedInt(colBytes, "colBytes");
                    var padded = new byte[paddedVocab * colBytes];
                    for (long r = 0; r < tensorVocab; r++)
                        Buffer.BlockCopy(rawBytes, (int)(r * colBytes), padded, (int)(r * colBytes), safeColBytes);
                    data = padded;
                }
            }

            if (target == weights.LmHeadWeight) { weights.RawLmHead = data; weights.RawLmHeadDtype = entry.Dtype; }
            else if (target == weights.EmbeddingWeight) { weights.RawEmbedding = data; weights.RawEmbeddingDtype = entry.Dtype; }
        }

        // Dequantize to float (same GGUF transpose semantics as GgufLoader)
        long longCount = TensorLoadHelper.ComputeElementCount(entry.Shape);
        if (longCount > int.MaxValue)
        {
            // Oversized tensor: raw bytes are already loaded above.
            // Skip dequant — the streaming forward pass reads them directly.
            return;
        }
        int count = (int)longCount;

        float[] buffer = MemoryHelpers.RentArray<float>(count);
        try
        {
            using var ms = new MemoryStream(rawBytes);
            using var reader = new BinaryReader(ms);
            ReadTensorInto(reader, entry.Dtype, entry.Shape, buffer.AsSpan(0, count));

            if (target != null)
            {
                target.Data.Clear();
                // Head data is [vocab, in] row-major in every file (see GgufLoader's
                // matching comment) — verbatim copy, same as the embedding.
                buffer.AsSpan(0, count).CopyTo(target.Data);
            }
            else if (block != null)
            {
                var floatTarget = weights.ResolveFloatTarget(entry.Name);
                if (floatTarget != null)
                {
                    if (entry.Shape.Length == 2)
                    {
                        int ggufIn = entry.Shape[0];
                        int ggufOut = entry.Shape[1];
                        bool isFfnUp = entry.Name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) &&
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
        int count = TensorLoadHelper.ComputeElementCountChecked(shape);
        if (destination.Length < count)
            throw new ArgumentException($"Destination buffer too small: {destination.Length} < {count}");
        _qOps.ReadFor(dtype, stream, destination, count);
    }

    // ── Index parsing ──────────────────────────────────────────────────────

    private sealed record SmmFileIndex(ModelMetaData Meta, List<SmmTensorIndexEntry> Entries);

    private static SmmFileIndex ReadIndex(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        return ReadIndex(reader, stream);
    }

    private static SmmFileIndex ReadIndex(BinaryReader reader, FileStream stream)
    {
        var meta = new ModelMetaData();

        uint magic = reader.ReadUInt32();
        if (magic != SmmConstants.Magic)
            throw new InvalidDataException("Not SMM: " + magic.ToString("X8"));
        uint version = reader.ReadUInt32();
        if (version != SmmConstants.Version)
            throw new InvalidDataException("Unsupported SMM version: " + version);

        long metaLen = reader.ReadInt64();
        long tokenizerLen = reader.ReadInt64();
        long pluginAsmCount = reader.ReadInt64();
        long tensorCount = reader.ReadInt64();
        long indexLen = reader.ReadInt64();
        long dataOffset = reader.ReadInt64();
        reader.ReadInt64(); // reserved

        meta.Version = version;
        meta.TensorCount = tensorCount;
        meta.DataOffset = dataOffset;

        // Meta JSON
        if (metaLen > 0)
        {
            byte[] metaBytes = reader.ReadBytes((int)metaLen);
            ParseMetaJson(Encoding.UTF8.GetString(metaBytes), meta);
        }

        // Tokenizer JSON
        if (tokenizerLen > 0)
        {
            byte[] tokBytes = reader.ReadBytes((int)tokenizerLen);
            meta.KvPairs.Add(new KvPair { Key = SmmConstants.TokenizerKey, Value = Encoding.UTF8.GetString(tokBytes) });
        }

        // Plugin manifest — skipped here (exposed via LoadPlugins)
        for (long i = 0; i < pluginAsmCount; i++)
        {
            var (_, _) = ReadString(reader);
            reader.ReadBoolean();
            long asmLen = reader.ReadInt64();
            reader.BaseStream.Position += asmLen;
        }

        // Tensor index is the last region of the file
        stream.Position = stream.Length - indexLen;
        var entries = new List<SmmTensorIndexEntry>();
        for (long i = 0; i < tensorCount; i++)
        {
            try
            {
                var (nameLen, name) = ReadString(reader);
                if (nameLen == 0 || nameLen > 500) break;

                var dtype = (QuantDType)reader.ReadInt32();
                int rank = reader.ReadInt32();
                if (rank < 0 || rank > 10) throw new InvalidDataException("Invalid tensor rank: " + rank);

                var shape = new int[rank];
                for (int j = 0; j < rank; j++) shape[j] = reader.ReadInt32();

                long offset = reader.ReadInt64();

                entries.Add(new SmmTensorIndexEntry(name, dtype, shape, offset));
                meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = shape, Offset = offset });
            }
            catch (Exception ex)
            {
                SanityChecks.WriteLine($"SmmLoader: tensor metadata read failed: {ex.Message}");
                break;
            }
        }

        return new SmmFileIndex(meta, entries);
    }

    private static void ParseMetaJson(string json, ModelMetaData meta)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            string arch = root.TryGetProperty("architecture", out var a) ? a.GetString() ?? "" : "";
            meta.KvPairs.Add(new KvPair { Key = "general.architecture", Value = arch });

            if (root.TryGetProperty("chat_template", out var ct) && ct.GetString() is { Length: > 0 } template)
                meta.KvPairs.Add(new KvPair { Key = "tokenizer.chat_template", Value = template });

            if (root.TryGetProperty("system_prompt", out var sp) && sp.GetString() is { Length: > 0 } systemPrompt)
                meta.KvPairs.Add(new KvPair { Key = SmmConstants.SystemPromptKey, Value = systemPrompt });

            if (root.TryGetProperty("skills", out var sk) && sk.ValueKind == JsonValueKind.Array)
            {
                var texts = sk.EnumerateArray()
                    .Select(e => e.GetString() ?? "")
                    .Where(s => s.Length > 0)
                    .ToList();
                if (texts.Count > 0)
                    meta.KvPairs.Add(new KvPair { Key = SmmConstants.SkillsKey, Value = JsonSerializer.Serialize(texts) });
            }

            if (root.TryGetProperty("config_json", out var cj) && cj.GetString() is { Length: > 0 } cfgJson)
                meta.KvPairs.Add(new KvPair { Key = SmmConstants.ConfigKey, Value = cfgJson });
        }
        catch (Exception ex)
        {
            SanityChecks.WriteLine($"SmmLoader: meta JSON parse failed: {ex.Message}");
        }
    }

    private static byte[] ReadTensorBytes(Stream stream, SmmFileIndex index, SmmTensorIndexEntry entry, long rawSize)
    {
        long absolute = index.Meta.DataOffset + entry.Offset;
        if (absolute < 0 || absolute + rawSize > stream.Length)
            throw new InvalidDataException($"Tensor '{entry.Name}' range is beyond end of file.");
        stream.Position = absolute;

        var bytes = new byte[rawSize];
        stream.ReadExactly(bytes);
        return bytes;
    }

    private static (int len, string value) ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0 || len > 100_000_000) throw new InvalidDataException("Invalid string length: " + len);
        return (len, Encoding.UTF8.GetString(reader.ReadBytes(len)));
    }
}
