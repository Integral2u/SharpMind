using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using SharpMind.Data.Sources;

namespace SharpMind.Data.Sources;

/// <summary>
/// Streams documents from Fusechat JSON files.
/// Format: Array of objects containing a "conversations" list of {from, value} turns.
/// </summary>
public sealed class FusechatSource : IDataSource
{
    private readonly string[] _paths;

    public FusechatSource(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _paths = GlobResolver.Resolve(path);

        if (_paths.Length == 0)
            throw new FileNotFoundException($"No files matched: {path}");
    }

    public FusechatSource(IEnumerable<string> paths)
    {
        ArgumentNullException.ThrowIfNull(paths);
        _paths = GlobResolver.ResolveMany(paths);

        if (_paths.Length == 0)
            throw new ArgumentException("Path list must not be empty.", nameof(paths));
    }

    public long? EstimatedCount => null;

    public string Description =>
        _paths.Length == 1
            ? $"Fusechat({Path.GetFileName(_paths[0])})"
            : $"Fusechat({_paths.Length} files)";

    public IAsyncEnumerator<string> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        => ReadAsync(cancellationToken).GetAsyncEnumerator(cancellationToken);

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (string path in _paths)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var stream = new FileStream(
                path, FileMode.Open, FileAccess.Read,
                FileShare.Read, bufferSize: 65_536, useAsync: true);

            // Since Fusechat files are large JSON arrays, we use Utf8JsonReader for efficiency
            // or just JsonDocument if size allows. The sample was ~1.7M records.
            // For 1.7M records, we definitely need a streaming approach.
            
            using var document = await JsonDocument.ParseAsync(stream, default, cancellationToken).ConfigureAwait(false);
            JsonElement root = document.RootElement;

            if (root.ValueKind != JsonValueKind.Array)
                continue;

            foreach (JsonElement entry in root.EnumerateArray())
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (entry.TryGetProperty("conversations", out JsonElement conversations) && 
                    conversations.ValueKind == JsonValueKind.Array)
                {
                    string text = FormatConversations(conversations);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        yield return text;
                    }
                }
            }
        }
    }

    private string FormatConversations(JsonElement conversations)
    {
        var sb = new StringBuilder();
        foreach (JsonElement turn in conversations.EnumerateArray())
        {
            if (turn.TryGetProperty("from", out JsonElement from) && 
                turn.TryGetProperty("value", out JsonElement value))
            {
                sb.AppendLine($"{from.GetString()}: {value.GetString()}");
            }
        }
        return sb.ToString().TrimEnd();
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
