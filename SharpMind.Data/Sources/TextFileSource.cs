using System.Runtime.CompilerServices;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Sources;

/// <summary>
/// Reads plain text files as a stream of documents.
///
/// Two modes:
///   <see cref="DocumentMode.LinePerDoc"/> — each non-empty line is one document.
///   <see cref="DocumentMode.FilePerDoc"/> — the entire file is one document.
///
/// Accepts a single path, explicit list of paths, or a glob pattern
/// (e.g. <c>data/**/*.txt</c>). Files are read in lexicographic order.
/// </summary>
[ComponentKind("Text File", "Plain text file(s); one line or one whole file per document.")]
public sealed class TextFileSource : IDataSource
{
    private readonly string[] _paths;
    private readonly DocumentMode _mode;

    public enum DocumentMode { LinePerDoc, FilePerDoc }

    public TextFileSource(
        [FileChooser("*.txt;*.md", "Text file or glob pattern, e.g. data/**/*.txt")] string path,
        [DefaultValue("LinePerDoc")] DocumentMode mode = DocumentMode.LinePerDoc)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _mode = mode;
        _paths = GlobResolver.Resolve(path);

        if (_paths.Length == 0)
            throw new FileNotFoundException($"No files matched: {path}");
    }

    public TextFileSource(
        IEnumerable<string> paths,
        DocumentMode mode = DocumentMode.LinePerDoc)
    {
        ArgumentNullException.ThrowIfNull(paths);
        _mode = mode;
        _paths = GlobResolver.ResolveMany(paths);

        if (_paths.Length == 0)
            throw new ArgumentException("Path list must not be empty.", nameof(paths));
    }

    public long? EstimatedCount => _mode == DocumentMode.FilePerDoc ? _paths.Length : null;

    public string Description =>
        _paths.Length == 1
            ? $"TextFile({Path.GetFileName(_paths[0])}, {_mode})"
            : $"TextFiles({_paths.Length} files, {_mode})";

    public async IAsyncEnumerable<string> ReadAsync(
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (string path in _paths)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_mode == DocumentMode.FilePerDoc)
            {
                string content = await File.ReadAllTextAsync(path, cancellationToken)
                                            .ConfigureAwait(false);
                if (!string.IsNullOrWhiteSpace(content))
                    yield return content;
            }
            else
            {
                using var reader = new StreamReader(path, detectEncodingFromByteOrderMarks: true);

                string? line;
                while ((line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false)) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                        yield return line;
                }
            }
        }   
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

}
