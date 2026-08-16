namespace SharpMind.Tests;

/// <summary>
/// Creates temporary files for tests and cleans them up on dispose.
/// </summary>
internal sealed class TempDirectory : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(
        System.IO.Path.GetTempPath(),
        System.IO.Path.GetRandomFileName());

    public TempDirectory() => Directory.CreateDirectory(Path);

    /// <summary>Writes text to a file inside the temp directory and returns its path.</summary>
    public string Write(string filename, string content)
    {
        string path = System.IO.Path.Combine(Path, filename);
        File.WriteAllText(path, content);
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(Path))
            Directory.Delete(Path, recursive: true);
    }
}

internal static class TestTokens
{
    /// <summary>
    /// Maps a word to a token id in [0, vocabSize). Uses FNV-1a rather than
    /// string.GetHashCode(), which is seeded per process — that made every test
    /// run train on different data, so any assertion on a numeric training
    /// outcome (e.g. "loss descends") passed or failed at random.
    /// </summary>
    public static int Id(string word, int vocabSize)
    {
        uint hash = 2166136261u;
        foreach (char c in word)
        {
            hash ^= c;
            hash *= 16777619u;
        }
        return (int)(hash % (uint)vocabSize);
    }

    /// <summary>Splits on spaces and maps each word with <see cref="Id"/>.</summary>
    public static int[] Encode(string text, int vocabSize) =>
        [.. text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => Id(w, vocabSize))];
}

internal static class AsyncEnumerableExtensions
{
    /// <summary>Collects all items from an async enumerable into a list.</summary>
    public static async Task<List<T>> ToListAsync<T>(
        this IAsyncEnumerable<T> source,
        CancellationToken cancellationToken = default)
    {
        var list = new List<T>();
        await foreach (var item in source.WithCancellation(cancellationToken))
            list.Add(item);
        return list;
    }
}
