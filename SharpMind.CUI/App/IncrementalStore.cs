using System.Text.Json;

namespace SharpMind.CUI.App;

/// <summary>
/// Persists a training job's per-file corpus hashes
/// (<see cref="TrainJobSettings.SourceFileHashes"/>) next to its checkpoints so
/// an incremental run can detect exactly which files are new or changed — even
/// across an app restart before the job file itself has been manually saved.
/// The sidecar lives at <c>&lt;CheckpointDir&gt;/incremental-file-hashes.json</c>
/// and is a plain JSON serialization of the map; missing/unreadable files are
/// treated as "no previous map" (the next run then sees everything as new).
/// </summary>
public static class IncrementalStore
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public static string PathFor(TrainJobSettings job)
        => System.IO.Path.Combine(job.CheckpointDir, "incremental-file-hashes.json");

    /// <summary>Reads the previously persisted per-file hash map, or an empty map.</summary>
    public static Dictionary<string, Dictionary<string, string>> Load(TrainJobSettings job)
    {
        try
        {
            string path = PathFor(job);
            return File.Exists(path)
                ? JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(File.ReadAllText(path), Options) ?? []
                : [];
        }
        catch
        {
            return [];
        }
    }

    /// <summary>Writes <see cref="TrainJobSettings.SourceFileHashes"/> to the sidecar.</summary>
    public static void Save(TrainJobSettings job)
    {
        try
        {
            Directory.CreateDirectory(job.CheckpointDir);
            File.WriteAllText(PathFor(job), JsonSerializer.Serialize(job.SourceFileHashes, Options));
        }
        catch
        {
            // Best-effort: an unwritable checkpoint dir shouldn't kill the run.
        }
    }
}