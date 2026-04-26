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
