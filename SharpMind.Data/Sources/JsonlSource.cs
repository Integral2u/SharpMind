using System.Runtime.CompilerServices;
using System.Text.Json;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Sources;

/// <summary>
/// Streams documents from JSONL (newline-delimited JSON) files.
///
/// Extracts a single text field by name — defaulting to <c>"text"</c>,
/// the convention used by HuggingFace datasets.
/// Supports dot notation for nested fields: <c>"meta.content"</c>.
/// Malformed lines are skipped; count tracked in <see cref="SkippedLines"/>.
/// </summary>
[ComponentKind("JSONL", "Newline-delimited JSON; the text field of each record is one document.")]
public sealed class JsonlSource : IDataSource
{
    private readonly string[] _paths;
    private readonly string[] _fieldPath;
    private long _skippedLines;

    public JsonlSource(
        [FileChooser("*.jsonl;*.json", "JSONL file or glob pattern")] string path,
        [DefaultValue("text")] string textField = "text")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(textField);
        _fieldPath = textField.Split('.', StringSplitOptions.RemoveEmptyEntries);
        _paths = GlobResolver.Resolve(path);

        if (_paths.Length == 0)
            throw new FileNotFoundException($"No files matched: {path}");
    }

    public JsonlSource(IEnumerable<string> paths, string textField = "text")
    {
        ArgumentNullException.ThrowIfNull(paths);
        ArgumentException.ThrowIfNullOrWhiteSpace(textField);
        _fieldPath = textField.Split('.', StringSplitOptions.RemoveEmptyEntries);
        _paths = GlobResolver.ResolveMany(paths);

        if (_paths.Length == 0)
            throw new ArgumentException("Path list must not be empty.", nameof(paths));
    }

    public long? EstimatedCount => null;
    public long SkippedLines => Volatile.Read(ref _skippedLines);

    public string Description =>
        _paths.Length == 1
            ? $"Jsonl({Path.GetFileName(_paths[0])}, field={string.Join('.', _fieldPath)})"
            : $"Jsonl({_paths.Length} files, field={string.Join('.', _fieldPath)})";

    public IAsyncEnumerator<string> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        => ReadAsync(cancellationToken).GetAsyncEnumerator(cancellationToken);

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var opts = new JsonDocumentOptions { AllowTrailingCommas = true };

        foreach (string path in _paths)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var stream = new FileStream(
                path, FileMode.Open, FileAccess.Read,
                FileShare.Read, bufferSize: 65_536, useAsync: true);
            using var reader = new StreamReader(stream, detectEncodingFromByteOrderMarks: true);

            while (!reader.EndOfStream)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string? line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
                if (string.IsNullOrWhiteSpace(line)) continue;

                string? text = ExtractField(line, opts);
                if (text is null)
                {
                    Interlocked.Increment(ref _skippedLines);
                    continue;
                }

                if (!string.IsNullOrWhiteSpace(text))
                    yield return text;
            }
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    private string? ExtractField(string line, JsonDocumentOptions opts)
    {
        try
        {
            using var doc = JsonDocument.Parse(line, opts);
            JsonElement current = doc.RootElement;
            foreach (string key in _fieldPath)
            {
                if (current.ValueKind != JsonValueKind.Object ||
                    !current.TryGetProperty(key, out current))
                    return null;
            }
            return current.ValueKind == JsonValueKind.String ? current.GetString() : null;
        }
        catch (JsonException) { return null; }
    }
}
