namespace SharpMind.Data.Pipeline.Stages;
/// <summary>
/// Deduplicates documents within a rolling window of <see cref="WindowSize"/> items.
/// Uses a hash set — memory grows with window size.
/// Exact global deduplication across large corpora should be done offline.
/// </summary>
public sealed class DeduplicateFilter : ICleaningStage
{
    private readonly HashSet<int> _seen;
    private readonly Queue<int> _window;
    private readonly int _windowSize;

    public DeduplicateFilter(int windowSize = 100_000)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(windowSize);
        _windowSize = windowSize;
        _seen = new HashSet<int>(windowSize);
        _window = new Queue<int>(windowSize);
    }

    public string Name => $"Deduplicate(window={_windowSize})";

    public string? Process(string document)
    {
        int hash = document.GetHashCode();
        if (!_seen.Add(hash)) return null;

        _window.Enqueue(hash);
        if (_window.Count > _windowSize)
            _seen.Remove(_window.Dequeue());

        return document;
    }
}