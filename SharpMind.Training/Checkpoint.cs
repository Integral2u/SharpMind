using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Core.Training;

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
        IOptimizer?           optimizer = null,
        int                   step      = 0,
        float                 loss      = float.NaN,
        string?               note      = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        Directory.CreateDirectory(directory);

        var paramList = parameters.ToList();

        // ── Model weights ─────────────────────────────────────────────────
        string modelPath = Path.Combine(directory, "model.bin");
        using (var modelStream = new FileStream(
            modelPath, FileMode.Create, FileAccess.Write,
            FileShare.None, bufferSize: 1 << 20, useAsync: false))
        using (var writer = new BinaryWriter(modelStream))
        {
            // Header: number of parameters
            writer.Write(paramList.Count);
            foreach (var p in paramList)
            {
                byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(p.Name);
                writer.Write(nameBytes.Length);
                writer.Write(nameBytes);
                writer.Write(p.Data.Shape.Rank);
                foreach (int d in p.Data.Shape.Dims) writer.Write(d);
                var data = p.Data.Data;
                for (int i = 0; i < data.Length; i++)
                    writer.Write(data[i]);
            }
        }

        // ── Optimizer state ───────────────────────────────────────────────
        if (optimizer is not null)
        {
            string optPath = Path.Combine(directory, "optimizer.bin");
            using var optStream = new FileStream(optPath, FileMode.Create, FileAccess.Write);
            using var optWriter = new BinaryWriter(optStream);
            optimizer.SaveState(optWriter);
        }

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
        IOptimizer?           optimizer = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Checkpoint directory not found: {directory}");

        var paramDict = parameters.ToDictionary(p => p.Name, StringComparer.Ordinal);

        string modelPath = Path.Combine(directory, "model.bin");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"model.bin not found in {directory}");

        using (var modelStream = new FileStream(
            modelPath, FileMode.Open, FileAccess.Read,
            FileShare.Read, bufferSize: 1 << 20, useAsync: false))
        using (var reader = new BinaryReader(modelStream))
        {
            int count = reader.ReadInt32();
            for (int i = 0; i < count; i++)
            {
                int nameLen = reader.ReadInt32();
                string name = System.Text.Encoding.UTF8.GetString(reader.ReadBytes(nameLen));
                int rank = reader.ReadInt32();
                int elemCount = 1;
                for (int d = 0; d < rank; d++) { int dim = reader.ReadInt32(); elemCount *= dim; }

                if (paramDict.TryGetValue(name, out var param))
                {
                    var data = param.Data.Data;
                    for (int j = 0; j < elemCount; j++)
                        data[j] = reader.ReadSingle();
                }
                else
                {
                    reader.BaseStream.Seek(elemCount * sizeof(float), SeekOrigin.Current);
                }
            }
        }

        // Read metadata first (needed for optimizer step)
        string metaPath = Path.Combine(directory, "meta.json");
        CheckpointMeta meta;
        if (!File.Exists(metaPath))
            meta = new CheckpointMeta();
        else
        {
            using var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
            var root = metaDoc.RootElement;
            meta = new CheckpointMeta
            {
                Step = root.TryGetProperty("step", out var s) ? s.GetInt32() : 0,
                Loss = root.TryGetProperty("loss", out var l) && l.ValueKind != JsonValueKind.Null
                           ? l.GetSingle() : float.NaN,
                Note = root.TryGetProperty("note", out var n) ? n.GetString() : null,
                SavedUtc = root.TryGetProperty("savedUtc", out var t)
                               ? DateTime.Parse(t.GetString()!) : DateTime.MinValue,
            };
        }

        // Load optimizer state (using step from meta)
        if (optimizer is not null)
        {
            string optPath = Path.Combine(directory, "optimizer.bin");
            if (File.Exists(optPath))
            {
                using var optStream = new FileStream(optPath, FileMode.Open, FileAccess.Read);
                using var optReader = new BinaryReader(optStream);
                optimizer.LoadState(optReader, meta.Step);
            }
        }

        return meta;
    }
}
