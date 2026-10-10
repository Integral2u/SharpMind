using SharpMind.Core.Diagnostics;
using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Text;
using System.Text.Json;

namespace SharpMind.Model.Format;

/// <summary>
/// Loads SharpMind Model (.SMM) containers. The SMM-specific work lives here:
/// reading the JSON-based header/index (<see cref="ReadIndex"/>), and the flat
/// on-disk byte rule from <see cref="ModelLoaderBase.RawByteCount"/>. Everything
/// after the index — target resolution, raw-quantized-data handling, fused
/// QKV/expert slicing, dequantization — is shared with <see cref="GgufLoader"/>
/// in <see cref="ModelLoaderBase"/>.
/// </summary>
public sealed class SmmLoader(QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false,
    int maxParallelLoadDegree = 0) : ModelLoaderBase(qOps, path, config, useSafeIo, maxParallelLoadDegree)
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    protected override ModelMetaData LoadIndex() => LoadMeta(_path);

    /// <summary>
    /// SMM packs blocks flat over the quantized buffer, so it has no per-row
    /// padding rule (unlike GGUF's <see cref="QuantizationOps.GetRawTensorByteCount"/>).
    /// </summary>
    protected override long RawByteCount(TensorInfo info)
        => QuantizationOps.GetFlatTensorByteCount(info.Shape, info.Dtype);

    // ── Static helpers (metadata / config / tokenizer / plugins) ──────────

    public static ModelMetaData LoadMeta(string path)
    {
        using var stream = ModelFileIo.OpenRead(path);
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
        using var stream = ModelFileIo.OpenRead(path);
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
        using var stream = ModelFileIo.OpenRead(path);
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

        using var stream = ModelFileIo.OpenRead(path);

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

    // ── Index parsing ──────────────────────────────────────────────────────

    private sealed record SmmFileIndex(ModelMetaData Meta, List<SmmTensorIndexEntry> Entries);

    private static SmmFileIndex ReadIndex(string path)
    {
        using var stream = ModelFileIo.OpenRead(path);
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

    private static (int len, string value) ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0 || len > 100_000_000) throw new InvalidDataException("Invalid string length: " + len);
        return (len, Encoding.UTF8.GetString(reader.ReadBytes(len)));
    }
}