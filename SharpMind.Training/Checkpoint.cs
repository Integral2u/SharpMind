using System.Runtime.InteropServices;
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
    // Save

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

        // Model weights
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
                writer.Write(MemoryMarshal.AsBytes(data));
            }
        }

        // Optimizer state
        if (optimizer is not null)
        {
            string optPath = Path.Combine(directory, "optimizer.bin");
            using var optStream = new FileStream(optPath, FileMode.Create, FileAccess.Write);
            using var optWriter = new BinaryWriter(optStream);
            optimizer.SaveState(optWriter);
        }

        // Metadata
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

    // Load

    /// <summary>
    /// Reads just <c>meta.json</c> from a checkpoint directory without touching
    /// the weights — used to identify how far a checkpoint got before resuming.
    /// </summary>
    public static CheckpointMeta ReadMeta(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);

        string metaPath = Path.Combine(directory, "meta.json");
        if (!File.Exists(metaPath)) return new CheckpointMeta();

        using var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
        var root = metaDoc.RootElement;
        return new CheckpointMeta
        {
            Step = root.TryGetProperty("step", out var s) ? s.GetInt32() : 0,
            Loss = root.TryGetProperty("loss", out var l) && l.ValueKind != JsonValueKind.Null
                       ? l.GetSingle() : float.NaN,
            Note = root.TryGetProperty("note", out var n) ? n.GetString() : null,
            SavedUtc = root.TryGetProperty("savedUtc", out var t)
                           ? DateTime.Parse(t.GetString()!) : DateTime.MinValue,
        };
    }

    /// <summary>
    /// Returns the path of the most advanced checkpoint under
    /// <paramref name="checkpointDir"/>, or null when the directory has no
    /// step-marked checkpoints. "Most advanced" = highest parsed step, with a
    /// tie broken by directory name (name order matches step order for
    /// zero-padded names). Used to auto-resume the latest interrupted run.
    /// </summary>
    public static string? FindLatest(string? checkpointDir)
    {
        if (string.IsNullOrWhiteSpace(checkpointDir) || !Directory.Exists(checkpointDir))
            return null;

        string? best = null;
        long bestStep = long.MinValue;
        foreach (var dir in Directory.GetDirectories(checkpointDir, "step-*"))
        {
            long step = ParseStep(Path.GetFileName(dir));
            if (step > bestStep || (step == bestStep && best is not null &&
                string.CompareOrdinal(Path.GetFileName(dir), Path.GetFileName(best)) > 0))
            {
                bestStep = step;
                best = dir;
            }
        }
        return best;
    }

    /// <summary>Parses the step number from "step-0000123" / "step-0000123-final" names.</summary>
    private static long ParseStep(string name)
    {
        if (!name?.StartsWith("step-", StringComparison.Ordinal) == true) return 0;
        int end = name.IndexOf('-', 5);
        var number = end < 0 ? name[5..] : name[5..end];
        return long.TryParse(number, out var v) ? v : 0;
    }

    /// <summary>
    /// Loads parameter values from a checkpoint directory into the supplied
    /// parameter list. Parameters are matched by name — any name present in
    /// the checkpoint but not in <paramref name="parameters"/> is skipped.
    ///
    /// Names are not required to be unique (e.g. NormLayer yields
    /// <c>LayerNormLayer.weight</c> without a layer index). When several
    /// parameters share a name they are matched positionally, in parameter-list
    /// order, which mirrors the order <see cref="Save"/> wrote them in.
    ///
    /// A shape mismatch between a checkpoint tensor and the matching parameter
    /// throws <see cref="InvalidDataException"/> naming the tensor — the safety
    /// check keeps a stale checkpoint from silently corrupting the model.
    /// </summary>
    public static CheckpointMeta Load(
        string                directory,
        IEnumerable<Parameter> parameters,
        IOptimizer?           optimizer = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Checkpoint directory not found: {directory}");

        var parametersList = parameters.ToList();
        // name -> FIFO of parameter indices, so duplicate names preserve order.
        var paramIndices = new Dictionary<string, Queue<int>>(StringComparer.Ordinal);
        for (int i = 0; i < parametersList.Count; i++)
        {
            if (!paramIndices.TryGetValue(parametersList[i].Name, out var q))
            {
                q = new Queue<int>();
                paramIndices.Add(parametersList[i].Name, q);
            }
            q.Enqueue(i);
        }

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

                Parameter? param = null;
                if (paramIndices.TryGetValue(name, out var q) && q.TryDequeue(out int idx))
                    param = parametersList[idx];

                if (param is not null)
                {
                    var data = param.Data.Data;
                    if (elemCount != data.Length)
                        throw new InvalidDataException(
                            $"Checkpoint tensor '{name}' has {elemCount} elements but the " +
                            $"model parameter '{param.Name}' has {data.Length}.");
                    reader.Read(MemoryMarshal.AsBytes(data));
                }
                else
                {
                    reader.BaseStream.Seek(elemCount * sizeof(float), SeekOrigin.Current);
                }
            }
        }

        // Read metadata first (needed for optimizer step)
        CheckpointMeta meta = ReadMeta(directory);

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
