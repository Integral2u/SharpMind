using System.Security.Cryptography;
using System.Text;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>
/// Deduplicates documents within a rolling window of <see cref="WindowSize"/> items.
/// Uses a hash set — memory grows with window size.
/// Exact global deduplication across large corpora should be done offline.
/// </summary>
[ComponentKind("Deduplicate", "Drops repeats within a rolling window.")]
public sealed class DeduplicateFilter : ICleaningStage
{
    private readonly HashSet<string> _seen;
    private readonly Queue<string> _window;
    private readonly int _windowSize;

    public DeduplicateFilter(
        [MinMaxDefault(1, 10_000_000, 100_000, 10_000), Tooltip("Rolling dedupe window size.")] int windowSize = 100_000)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowSize);
        _windowSize = windowSize;
        _seen = new HashSet<string>(windowSize, StringComparer.Ordinal);
        _window = new Queue<string>(windowSize);
    }

    public string Name => $"Deduplicate(window={_windowSize})";

    public string? Process(string document)
    {
        // GetHashCode is randomly seeded per process and collision-prone; a collision would drop a distinct document.
        string hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(document)));
        if (!_seen.Add(hash)) return null;

        _window.Enqueue(hash);
        if (_window.Count > _windowSize)
            _seen.Remove(_window.Dequeue());

        return document;
    }
}