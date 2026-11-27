namespace SharpMind.Model.Audio;

/// <summary>
/// Locates a speech-output projector next to a backbone. The projector name is
/// <c>mmproj-*.gguf</c> in the backbone folder or a sibling folder named after the
/// backbone stem, matching the existing LFM3 projector layout.
/// </summary>
public static class TtsProjector
{
    private static readonly string[] QuantSuffixes =
        ["-Q8_0", "-Q4_K_M", "-Q4_0", "-F16", "-BF16", "-Q6_K", "-Q5_K_M"];

    /// <summary>The first projector that either TTS runtime recognises, or null.</summary>
    public static string? Find(string backbonePath) =>
        Find(backbonePath, static path =>
            Zonos3TtsRuntime.IsTtsProjector(path) || Lfm3TtsRuntime.IsTtsProjector(path));

    /// <summary>Finds a projector next to <paramref name="backbonePath"/> matching a predicate.</summary>
    public static string? Find(string backbonePath, Func<string, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        string? directory = Path.GetDirectoryName(backbonePath);
        if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory)) return null;

        string stem = Path.GetFileNameWithoutExtension(backbonePath);
        foreach (string suffix in QuantSuffixes)
        {
            if (stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                stem = stem[..^suffix.Length];
                break;
            }
        }

        var candidates = new List<string>();
        candidates.AddRange(Directory.EnumerateFiles(directory, "mmproj-*.gguf", SearchOption.TopDirectoryOnly));
        foreach (string child in Directory.EnumerateDirectories(directory))
        {
            if (!Path.GetFileName(child).StartsWith(stem, StringComparison.OrdinalIgnoreCase)) continue;
            candidates.AddRange(Directory.EnumerateFiles(child, "mmproj-*.gguf", SearchOption.TopDirectoryOnly));
        }
        if (candidates.Count == 0)
            candidates.AddRange(Directory.EnumerateFiles(directory, "mmproj-*.gguf", SearchOption.AllDirectories));

        return candidates
            .Where(predicate)
            .OrderByDescending(path => Path.GetFileName(path).Contains(stem, StringComparison.OrdinalIgnoreCase))
            .ThenByDescending(PrecisionRank)
            .ThenBy(path => path, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    /// <summary>Full-precision projector files win when both F32 and F16 copies are present.</summary>
    private static int PrecisionRank(string path)
    {
        string name = Path.GetFileName(path);
        if (name.Contains("F32", StringComparison.OrdinalIgnoreCase)) return 3;
        if (name.Contains("BF16", StringComparison.OrdinalIgnoreCase)) return 2;
        if (name.Contains("F16", StringComparison.OrdinalIgnoreCase)) return 1;
        return 0;
    }
}
