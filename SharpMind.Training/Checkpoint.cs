using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Training.Autograd;
using SharpMind.Training.Optimizers;

namespace SharpMind.Training;

/// <summary>
/// Saves and loads training state so runs can be interrupted and resumed.
///
/// A checkpoint directory contains:
///   model.bin       — raw float32 parameter tensors, one file per parameter
///   optimizer.json  — AdamW moment vectors + step count + LR
///   meta.json       — step number, loss, timestamp, config snapshot
///
/// Weights are stored as raw binary (little-endian float32) rather than JSON
/// for speed and size — a 7B model checkpoint is ~28GB, JSON would be unusable.
/// </summary>
public static class Checkpoint
{
    // ── Save ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Saves all parameters and optimizer state to <paramref name="directory"/>.
    /// Creates the directory if it does not exist.
    /// </summary>
    public static void Save(
        string                directory,
        IEnumerable<Parameter> parameters,
        AdamW?                optimizer = null,
        int                   step      = 0,
        float                 loss      = float.NaN,
        string?               note      = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory.CreateDirectory(directory);

        var paramList = parameters.ToList();

        // ── Model weights ─────────────────────────────────────────────────
        string modelPath = Path.Combine(directory, "model.bin");
        using var modelStream = new FileStream(
            modelPath, FileMode.Create, FileAccess.Write,
            FileShare.None, bufferSize: 1 << 20, useAsync: false);
        using var writer = new BinaryWriter(modelStream);

        // Header: number of parameters
        writer.Write(paramList.Count);
        foreach (var p in paramList)
        {
            // Name length + name bytes
            byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(p.Name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            // Shape rank + dims
            writer.Write(p.Data.Shape.Rank);
            foreach (int d in p.Data.Shape.Dims) writer.Write(d);
            // Data as raw float32
            var data = p.Data.Data;
            for (int i = 0; i < data.Length; i++)
                writer.Write(data[i]);
        }

        // ── Optimizer state ───────────────────────────────────────────────
        if (optimizer is not null)
            SaveOptimizerState(Path.Combine(directory, "optimizer.bin"), optimizer, paramList);

        // ── Metadata ──────────────────────────────────────────────────────
        var meta = new JsonObject
        {
            ["step"]      = step,
            ["loss"]      = float.IsNaN(loss) ? null : (JsonNode?)JsonValue.Create(loss),
            ["note"]      = note,
            ["savedUtc"]  = DateTime.UtcNow.ToString("o"),
            ["paramCount"] = paramList.Count,
        };
        File.WriteAllText(
            Path.Combine(directory, "meta.json"),
            meta.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    // ── Load ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Loads parameter values from a checkpoint directory into the supplied
    /// parameter list. Parameters are matched by name — any name present in
    /// the checkpoint but not in <paramref name="parameters"/> is skipped.
    /// </summary>
    public static CheckpointMeta Load(
        string                directory,
        IEnumerable<Parameter> parameters,
        AdamW?                optimizer = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Checkpoint directory not found: {directory}");

        var paramDict = parameters.ToDictionary(p => p.Name, StringComparer.Ordinal);

        string modelPath = Path.Combine(directory, "model.bin");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"model.bin not found in {directory}");

        using var modelStream = new FileStream(
            modelPath, FileMode.Open, FileAccess.Read,
            FileShare.Read, bufferSize: 1 << 20, useAsync: false);
        using var reader = new BinaryReader(modelStream);

        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            int    nameLen  = reader.ReadInt32();
            string name     = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
            int    rank     = reader.ReadInt32();
            int    elemCount = 1;
            for (int d = 0; d < rank; d++) { int dim = reader.ReadInt32(); elemCount *= dim; }

            if (paramDict.TryGetValue(name, out var param))
            {
                var data = param.Data.Data;
                for (int j = 0; j < elemCount; j++)
                    data[j] = reader.ReadSingle();
            }
            else
            {
                // Skip — parameter exists in checkpoint but not in current model
                reader.BaseStream.Seek(elemCount * sizeof(float), SeekOrigin.Current);
            }
        }

        if (optimizer is not null)
        {
            string optPath = Path.Combine(directory, "optimizer.bin");
            if (File.Exists(optPath))
                LoadOptimizerState(optPath, optimizer, [.. paramDict.Values]);
        }

        // Read metadata
        string metaPath = Path.Combine(directory, "meta.json");
        if (!File.Exists(metaPath))
            return new CheckpointMeta();

        using var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
        var root = metaDoc.RootElement;
        return new CheckpointMeta
        {
            Step = root.TryGetProperty("step", out var s)    ? s.GetInt32()  : 0,
            Loss = root.TryGetProperty("loss", out var l) && l.ValueKind != JsonValueKind.Null
                       ? l.GetSingle() : float.NaN,
            Note     = root.TryGetProperty("note", out var n) ? n.GetString() : null,
            SavedUtc = root.TryGetProperty("savedUtc", out var t)
                           ? DateTime.Parse(t.GetString()!) : DateTime.MinValue,
        };
    }

    // ── Optimizer state helpers ───────────────────────────────────────────

    private static void SaveOptimizerState(string path, AdamW optimizer, List<Parameter> parameters)
    {
        // Use reflection to access private moment vectors — avoids adding
        // public API surface to AdamW just for checkpointing.
        var mField = typeof(AdamW).GetField("_m", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var vField = typeof(AdamW).GetField("_v", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var stepField = typeof(AdamW).GetField("_step", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var lrField = typeof(AdamW).GetField("_lr", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (mField?.GetValue(optimizer) is not float[][] m) return;
        if (vField?.GetValue(optimizer) is not float[][] v) return;
        int step = stepField?.GetValue(optimizer) is int s ? s : 0;
        float lr = lrField?.GetValue(optimizer) is float l ? l : 0f;

        using var fs     = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(fs);
        writer.Write(step);
        writer.Write(lr);
        for (int i = 0; i < m.Length && i < parameters.Count; i++)
        {
            foreach (float f in m[i]) writer.Write(f);
            foreach (float f in v[i]) writer.Write(f);
        }
    }

    private static void LoadOptimizerState(string path, AdamW optimizer, List<Parameter> parameters)
    {
        var mField    = typeof(AdamW).GetField("_m",    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var vField    = typeof(AdamW).GetField("_v",    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var stepField = typeof(AdamW).GetField("_step", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var lrField   = typeof(AdamW).GetField("_lr",   System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (mField?.GetValue(optimizer) is not float[][] m) return;
        if (vField?.GetValue(optimizer) is not float[][] v) return;

        using var fs     = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);

        int   step = reader.ReadInt32();
        float lr   = reader.ReadSingle();
        stepField?.SetValue(optimizer, step);
        lrField?.SetValue(optimizer, lr);

        for (int i = 0; i < m.Length && i < parameters.Count; i++)
        {
            for (int j = 0; j < m[i].Length; j++) m[i][j] = reader.ReadSingle();
            for (int j = 0; j < v[i].Length; j++) v[i][j] = reader.ReadSingle();
        }
    }
}

/// <summary>Metadata returned when loading a checkpoint.</summary>
public sealed record CheckpointMeta
{
    public int      Step     { get; init; }
    public float    Loss     { get; init; } = float.NaN;
    public string?  Note     { get; init; }
    public DateTime SavedUtc { get; init; }
}
