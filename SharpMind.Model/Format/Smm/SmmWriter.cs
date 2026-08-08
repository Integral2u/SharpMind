using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SharpMind.Model.Format;

/// <summary>
/// Writes a SharpMind Model (.SMM) container.
///
/// File layout (all integers little-endian):
/// <code>
/// [Header 64B]
///   u32 magic          "SMM1"
///   u32 version        1
///   u64 metaLen
///   u64 tokenizerLen
///   u64 pluginAsmCount
///   u64 tensorCount
///   u64 indexLen       (bytes of the tensor index region at end of file)
///   u64 dataOffset     (absolute offset of the tensor data region)
///   u64 reserved
/// [Meta JSON]          { architecture, chat_template?, system_prompt?, skills?, source, config_json }
/// [Tokenizer JSON]     SharpMind tokenizer JSON (SmmOutputs.Tokenizer)
/// [Plugin manifest]    pluginAsmCount × (name, recommended, len + assembly bytes)
/// [Data region]        each tensor's raw bytes, verbatim
/// [Tensor index]       tensorCount × (name, dtype, rank+shape, offset)
/// </code>
/// The data region is written before the index so each tensor's bytes are
/// fetched, quantized and flushed one at a time — a GGUF→SMM
/// converter never needs to hold the whole model in memory.
/// </summary>
public static class SmmWriter
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    /// <summary>
    /// Writes an .SMM file. <paramref name="tensors"/> must all use the same
    /// layout convention as GGUF (names like "blk.0.attn_q.weight" and 2D
    /// weights stored with dims [in, out] in transposed row-major order) so a
    /// single <see cref="SmmLoader"/> can serve both GGUF-converted and
    /// training-exported files.
    /// </summary>
    public static void Write(
        string path,
        ModelConfig config,
        Tokenizer? tokenizer,
        string? chatTemplate,
        IEnumerable<SmmTensorData> tensors,
        SmmWriteOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);

        options ??= new SmmWriteOptions();
        if (options.QuantizationLevel is not null && !TensorQuantizer.IsSupportedTarget(options.QuantizationLevel.Value))
            throw new NotSupportedException(
                $"Quantization level {options.QuantizationLevel} is not supported. Use F32, F16, Q8_0 or Q4_0.");

        using var fs = new FileStream(path, FileMode.Create, FileAccess.ReadWrite);
        using var writer = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        // ── Header placeholder (rewritten at the end once lengths are known) ──
        writer.Write(new byte[SmmConstants.HeaderSize]);

        // ── Meta JSON ──
        string metaJson = BuildMetaJson(config, chatTemplate, options);
        byte[] metaBytes = Encoding.UTF8.GetBytes(metaJson);
        writer.Write(metaBytes);

        // ── Tokenizer JSON ──
        bool includeTokenizer = (options.Outputs & SmmOutputs.Tokenizer) != 0 && tokenizer is not null;
        byte[] tokenizerBytes = includeTokenizer ? Encoding.UTF8.GetBytes(tokenizer!.ToJson()) : [];
        writer.Write(tokenizerBytes);

        // ── Plugin manifest ──
        bool includePlugins = (options.Outputs & SmmOutputs.Plugins) != 0;
        IReadOnlyList<SmmPluginEntry>? plugins = includePlugins ? options.Plugins : null;
        long pluginAsmCount = plugins?.Count ?? 0;
        if (plugins is not null)
        {
            foreach (var plugin in plugins)
            {
                WriteString(writer, plugin.Name);
                writer.Write(plugin.Recommended);
                writer.Write((long)plugin.AssemblyBytes.Length);
                writer.Write(plugin.AssemblyBytes);
            }
        }

        // ── Align to the data region ──
        long dataOffset = Align(fs.Position, options.Alignment);
        if (dataOffset > fs.Position)
            writer.Write(new byte[dataOffset - fs.Position]);

        // ── Tensor data (streamed one tensor at a time) ──
        var index = new List<SmmTensorIndexEntry>();
        long dataCursor = 0;
        foreach (var tensor in tensors)
        {
            var (dtype, bytes) = PrepareTensor(tensor, options);
            writer.Write(bytes);
            index.Add(new SmmTensorIndexEntry(tensor.Name, dtype, tensor.Shape, dataCursor));
            dataCursor += bytes.Length;
        }

        // ── Tensor index (end of file) ──
        long indexStart = fs.Position;
        foreach (var entry in index)
        {
            WriteString(writer, entry.Name);
            writer.Write((int)entry.Dtype);
            writer.Write(entry.Shape.Length);
            foreach (int dim in entry.Shape) writer.Write(dim);
            writer.Write(entry.Offset);
        }
        long indexLen = fs.Position - indexStart;

        // ── Rewrite the header with final values ──
        fs.Position = 0;
        writer.Write(SmmConstants.Magic);
        writer.Write(SmmConstants.Version);
        writer.Write((long)metaBytes.Length);
        writer.Write((long)tokenizerBytes.Length);
        writer.Write(pluginAsmCount);
        writer.Write((long)index.Count);
        writer.Write(indexLen);
        writer.Write(dataOffset);
        writer.Write(0L); // reserved
        writer.Flush();
    }

    private static (QuantDType dtype, byte[] bytes) PrepareTensor(SmmTensorData tensor, SmmWriteOptions options)
    {
        var target = options.QuantizationLevel;
        if (target is null || target == QuantDType.F32 || target == tensor.Dtype)
            return (tensor.Dtype, tensor.GetBytes());

        // Only F32 sources are quantized — already-quantized tensors (e.g.
        // from a GGUF conversion) are passed through verbatim.
        if (tensor.Dtype != QuantDType.F32)
            return (tensor.Dtype, tensor.GetBytes());

        byte[] raw = tensor.GetBytes();
        int count = 1;
        foreach (int dim in tensor.Shape) count *= dim;
        if (raw.Length != count * 4)
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' claims F32 ({tensor.Shape.Length}-D, {count} elements) but has {raw.Length} bytes.");

        var values = new float[count];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return (target.Value, TensorQuantizer.Quantize(values, tensor.Shape, target.Value));
    }

    private static string BuildMetaJson(ModelConfig config, string? chatTemplate, SmmWriteOptions options)
    {
        bool includeTemplate = (options.Outputs & SmmOutputs.ChatTemplate) != 0;
        var obj = new JsonObject
        {
            ["architecture"] = config.Architecture ?? "",
            ["source"] = options.Source ?? "smm",
            ["config_json"] = JsonSerializer.Serialize(config, JsonOpts),
        };
        if (includeTemplate && !string.IsNullOrWhiteSpace(chatTemplate))
            obj["chat_template"] = chatTemplate;
        if (options.Skills is { Count: > 0 })
            obj["skills"] = new JsonArray([.. options.Skills.Select(s => (JsonNode?)JsonValue.Create(s))]);
        if (!string.IsNullOrWhiteSpace(options.SystemPrompt))
            obj["system_prompt"] = options.SystemPrompt;
        return obj.ToJsonString(JsonOpts);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static long Align(long position, int alignment)
        => (position + alignment - 1) & ~(alignment - 1L);
}

/// <summary>A single tensor index entry inside an .SMM container.</summary>
public readonly record struct SmmTensorIndexEntry(
    string Name,
    QuantDType Dtype,
    int[] Shape,
    long Offset);
