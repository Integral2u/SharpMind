using System.Security.Cryptography;
using System.Text;
using SharpMind.Data.Sources;

namespace SharpMind.CUI.App;

/// <summary>
/// Computes content fingerprints for a training job's configured sources so a
/// later run can tell whether the corpus changed (tokenizer cache staleness)
/// and stamp the trained export's metadata with its training data fingerprint.
/// </summary>
public static class SourceHasher
{
    /// <summary>
    /// Resolves the file-based <c>path</c> arg of each source through
    /// <see cref="GlobResolver"/> (already deterministic, lexicographic) and
    /// returns a per-source SHA-256 over the sorted resolved files' content
    /// bytes, in lexicographic file order. Content-only, so the same corpus on
    /// different machines (or paths) produces the same fingerprint — ideal for
    /// stamping exported model metadata. Sources without a resolvable path
    /// (e.g. remote HuggingFace) are skipped entirely.
    /// </summary>
    public static Dictionary<string, string?> Compute(IEnumerable<JobSource> sources)
    {
        var result = new Dictionary<string, string?>(StringComparer.Ordinal);
        foreach (var source in sources)
        {
            string? path = PathArg(source.Component.Args);
            if (string.IsNullOrWhiteSpace(path)) continue;

            string[] files = GlobResolver.Resolve(path);
            if (files.Length == 0) continue;

            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            foreach (string file in files)
                hash.AppendData(File.ReadAllBytes(file));
            result[source.DisplayName ?? source.Component.TypeName] = Convert.ToHexString(hash.GetHashAndReset());
        }
        return result;
    }

    /// <summary>
    /// Combined one-line fingerprint of all hashable sources, for the exported
    /// model's metadata (null when nothing is hashable).
    /// </summary>
    public static string? Combined(IEnumerable<JobSource> sources)
    {
        var perSource = Compute(sources);
        if (perSource.Count == 0) return null;

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var kv in perSource.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            hash.AppendData(Encoding.UTF8.GetBytes($"{kv.Key}="));
            hash.AppendData(Encoding.UTF8.GetBytes(kv.Value ?? ""));
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary>
    /// Computes a per-file SHA-256 map for every file-based source so a later
    /// incremental run can tell exactly which files are new or changed and train
    /// only on those deltas. Keys are the same source display names used by
    /// <see cref="Compute"/>; values map each resolved absolute file path (in
    /// lexicographic order) to its content hash. Sources without a resolvable
    /// path (e.g. remote HuggingFace) are absent, exactly like <see cref="Compute"/>.
    /// </summary>
    public static Dictionary<string, Dictionary<string, string>> ComputeFileHashes(IEnumerable<JobSource> sources)
    {
        var result = new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        foreach (var source in sources)
        {
            string? path = PathArg(source.Component.Args);
            if (string.IsNullOrWhiteSpace(path)) continue;

            string[] files = GlobResolver.Resolve(path);
            if (files.Length == 0) continue;

            var perFile = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string file in files)
            {
                using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                hash.AppendData(File.ReadAllBytes(file));
                perFile[file] = Convert.ToHexString(hash.GetHashAndReset());
            }
            result[source.DisplayName ?? source.Component.TypeName] = perFile;
        }
        return result;
    }

    /// <summary>
    /// The union of delta files across a job's configured sources for an
    /// incremental run: every file whose content hash is absent from (new) or
    /// different than (changed) the persisted map. Files that are unchanged keep
    /// the knowledge already baked into the resumed weights and are skipped.
    /// </summary>
    public static Dictionary<string, string[]> ComputeDeltas(
        Dictionary<string, Dictionary<string, string>> current,
        Dictionary<string, Dictionary<string, string>> previous)
    {
        var deltas = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (var (sourceKey, files) in current)
        {
            previous.TryGetValue(sourceKey, out var prior);
            var delta = new List<string>();
            foreach (var (file, hash) in files)
            {
                if (prior is null || !prior.TryGetValue(file, out var priorHash) || !string.Equals(priorHash, hash, StringComparison.Ordinal))
                    delta.Add(file);
            }
            if (delta.Count > 0)
                deltas[sourceKey] = delta.ToArray();
        }
        return deltas;
    }

    /// <summary>Extracts the wizard-supplied "path" constructor arg, if any.</summary>
    private static string? PathArg(IReadOnlyDictionary<string, string> args)
        => args.TryGetValue("path", out var raw) ? raw : null;
}