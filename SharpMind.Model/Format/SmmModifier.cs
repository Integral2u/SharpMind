using System.Text;
using System.Text.Json.Nodes;

namespace SharpMind.Model.Format;

/// <summary>
/// The editable side-channel content of an .SMM container: the embedded default
/// system prompt, the embedded skills (markdown documents), and the embedded
/// plugin assemblies. Populated by <see cref="SmmModifier.Read"/> and written
/// back by <see cref="SmmModifier.Write"/>.
/// </summary>
public sealed class SmmModelDocument
{
    /// <summary>Current embedded system prompt, or an empty string when absent.</summary>
    public string SystemPrompt { get; set; } = "";

    /// <summary>Embedded skills, one markdown document per entry.</summary>
    public List<string> Skills { get; } = [];

    /// <summary>Embedded plugin assemblies.</summary>
    public List<SmmPluginEntry> Plugins { get; } = [];
}

/// <summary>
/// Adds, removes or alters the side-channel metadata (system prompt / skills /
/// plugin assemblies) of an existing .SMM file without touching the model:
/// the meta JSON is rebuilt in place, the tokenizer JSON is copied verbatim,
/// and the tensor data region plus tensor index are streamed through unchanged.
///
/// The write goes to a <c>.tmp</c> file and is atomically moved over the
/// original, so a failure (or cancelled operation) never trashes the model.
/// </summary>
public static class SmmModifier
{
    /// <summary>Reads the current system prompt, skills and plugins from an .SMM file.</summary>
    public static SmmModelDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException("SMM file not found", path);

        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        ReadHeader(reader, out long metaLen, out long tokenizerLen, out long pluginAsmCount, out _, out _, out _);

        var doc = new SmmModelDocument();

        byte[] metaBytes = reader.ReadBytes(checked((int)metaLen));
        if (JsonNode.Parse(Encoding.UTF8.GetString(metaBytes)) is JsonObject root)
        {
            if (root["system_prompt"] is JsonValue prompt && prompt.TryGetValue<string>(out var value))
                doc.SystemPrompt = value ?? "";
            if (root["skills"] is JsonArray skills)
            {
                foreach (var item in skills)
                {
                    if (item is JsonValue skill && skill.TryGetValue<string>(out var text))
                        doc.Skills.Add(text);
                }
            }
        }

        reader.BaseStream.Position += tokenizerLen;

        for (long i = 0; i < pluginAsmCount; i++)
        {
            string name = ReadString(reader);
            bool recommended = reader.ReadBoolean();
            long asmLen = reader.ReadInt64();
            byte[] asm = reader.ReadBytes(checked((int)asmLen));
            doc.Plugins.Add(new SmmPluginEntry { Name = name, AssemblyBytes = asm, Recommended = recommended });
        }

        return doc;
    }

    /// <summary>
    /// Rewrites <paramref name="path"/> with the edited document, atomically.
    /// Everything the caller didn't touch — tokenizer JSON, tensor data, tensor
    /// index, and any other meta keys (config, chat template, source) — is
    /// preserved byte-for-byte.
    ///
    /// When the document already matches what is on disk (a no-op edit) the
    /// file is left completely untouched; the rewrite is skipped entirely.
    /// </summary>
    public static void Write(string path, SmmModelDocument doc)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(doc);
        if (!File.Exists(path)) throw new FileNotFoundException("SMM file not found", path);

        if (DocsEqual(Read(path), doc))
            return; // Nothing changed: do not rewrite (avoids the costly 535 MB copy and the replace-deny).

        string tmpPath = path + ".tmp";
        try
        {
            WriteInternal(path, tmpPath, doc);
            MoveWithRetry(tmpPath, path);
        }
        catch
        {
            TryDelete(tmpPath);
            throw;
        }
    }

    private static bool DocsEqual(SmmModelDocument a, SmmModelDocument b)
    {
        if (!string.Equals(a.SystemPrompt, b.SystemPrompt, StringComparison.Ordinal))
            return false;
        if (a.Skills.Count != b.Skills.Count)
            return false;
        for (int i = 0; i < a.Skills.Count; i++)
        {
            if (!string.Equals(a.Skills[i], b.Skills[i], StringComparison.Ordinal))
                return false;
        }
        if (a.Plugins.Count != b.Plugins.Count)
            return false;
        for (int i = 0; i < a.Plugins.Count; i++)
        {
            var pa = a.Plugins[i];
            var pb = b.Plugins[i];
            if (!string.Equals(pa.Name, pb.Name, StringComparison.Ordinal))
                return false;
            if (pa.Recommended != pb.Recommended)
                return false;
            if (!pa.AssemblyBytes.AsSpan().SequenceEqual(pb.AssemblyBytes))
                return false;
        }
        return true;
    }

    private static void MoveWithRetry(string sourcePath, string destPath)
    {
        const int maxAttempts = 6;
        const int delayMs = 100;
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                File.Move(sourcePath, destPath, overwrite: true);
                return;
            }
            catch (IOException) when (attempt < maxAttempts)
            {
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }

    private static void WriteInternal(string sourcePath, string tmpPath, SmmModelDocument doc)
    {
        using var src = File.OpenRead(sourcePath);
        using var srcReader = new BinaryReader(src);
        ReadHeader(srcReader, out long metaLen, out long tokenizerLen, out _, out long tensorCount, out long indexLen, out long dataOffset);

        byte[] metaBytes = srcReader.ReadBytes(checked((int)metaLen));
        byte[] tokenizerBytes = srcReader.ReadBytes(checked((int)tokenizerLen));

        if (JsonNode.Parse(Encoding.UTF8.GetString(metaBytes)) is not JsonObject root)
            throw new InvalidDataException("Not SMM: malformed meta JSON in " + sourcePath);

        if (string.IsNullOrWhiteSpace(doc.SystemPrompt))
            root.Remove("system_prompt");
        else
            root["system_prompt"] = doc.SystemPrompt;

        if (doc.Skills.Count == 0)
            root.Remove("skills");
        else
            root["skills"] = new JsonArray([.. doc.Skills.Select(s => (JsonNode?)JsonValue.Create(s))]);

        byte[] newMetaBytes = Encoding.UTF8.GetBytes(root.ToJsonString());

        using var outFs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(outFs);

        long metaPos = outFs.Position;
        writer.Write(new byte[SmmConstants.HeaderSize]);
        writer.Write(newMetaBytes);
        writer.Write(tokenizerBytes);

        // Plugin manifest (rebuilt from the edited list).
        if (doc.Plugins is { Count: > 0 })
        {
            foreach (var plugin in doc.Plugins)
            {
                WriteString(writer, plugin.Name);
                writer.Write(plugin.Recommended);
                writer.Write((long)plugin.AssemblyBytes.Length);
                writer.Write(plugin.AssemblyBytes);
            }
        }

        // Align to the same block size, then stream the original data region +
        // tensor index through untouched.
        long newDataOffset = Align(outFs.Position, SmmConstants.DefaultAlignment);
        if (newDataOffset > outFs.Position)
            writer.Write(new byte[newDataOffset - outFs.Position]);

        src.Position = dataOffset;
        src.CopyTo(outFs);

        // Patch header.
        long end = outFs.Position;
        outFs.Position = metaPos;
        writer.Write(SmmConstants.Magic);
        writer.Write(SmmConstants.Version);
        writer.Write((long)newMetaBytes.Length);
        writer.Write((long)tokenizerBytes.Length);
        writer.Write((long)doc.Plugins.Count);
        writer.Write(tensorCount);
        writer.Write(indexLen);
        writer.Write(newDataOffset);
        writer.Write(0L); // reserved
        writer.Flush();
        _ = end;
    }

    private static void ReadHeader(
        BinaryReader reader,
        out long metaLen, out long tokenizerLen, out long pluginAsmCount,
        out long tensorCount, out long indexLen, out long dataOffset)
    {
        uint magic = reader.ReadUInt32();
        if (magic != SmmConstants.Magic)
            throw new InvalidDataException("Not SMM: " + magic.ToString("X8"));
        uint version = reader.ReadUInt32();
        if (version != SmmConstants.Version)
            throw new InvalidDataException("Unsupported SMM version: " + version);

        metaLen = reader.ReadInt64();
        tokenizerLen = reader.ReadInt64();
        pluginAsmCount = reader.ReadInt64();
        tensorCount = reader.ReadInt64();
        indexLen = reader.ReadInt64();
        dataOffset = reader.ReadInt64();
        reader.ReadInt64(); // reserved
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadString(BinaryReader reader)
    {
        int len = reader.ReadInt32();
        if (len < 0 || len > 100_000_000) throw new InvalidDataException("Invalid string length: " + len);
        return Encoding.UTF8.GetString(reader.ReadBytes(len));
    }

    private static long Align(long position, int alignment)
        => (position + alignment - 1) & ~(alignment - 1L);

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort cleanup */ }
    }
}