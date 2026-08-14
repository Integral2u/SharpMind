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

    /// <summary>Extracts the wizard-supplied "path" constructor arg, if any.</summary>
    private static string? PathArg(IReadOnlyDictionary<string, string> args)
        => args.TryGetValue("path", out var raw) ? raw : null;
}