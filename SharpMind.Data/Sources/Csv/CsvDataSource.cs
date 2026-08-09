using System.Runtime.CompilerServices;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Sources.Csv;

[ComponentKind("CSV", "CSV/TSV file; one row (text column) per document.")]
public sealed class CsvDataSource : IDataSource
{
    private readonly string _path;
    private readonly string _textColumn;
    private readonly bool _hasHeader;
    private readonly char _delimiter;

    public CsvDataSource(
        [FileChooser("*.csv", "CSV file path")] string path,
        [DefaultValue("text")] string textColumn = "text",
        [DefaultValue("true")] bool hasHeader = true,
        [DefaultValue(",")] char delimiter = ',')
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        
        _path = path;
        _textColumn = textColumn;
        _hasHeader = hasHeader;
        _delimiter = delimiter;
    }

    public long? EstimatedCount => null;

    public string Description => $"CSV: {_path} (column: {_textColumn})";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var reader = new StreamReader(_path);
        
        var header = _hasHeader ? await reader.ReadLineAsync(cancellationToken) : null;
        
        int textColIndex = 0;
        if (_hasHeader && header is not null)
        {
            var headers = header.Split(_delimiter);
            textColIndex = Array.IndexOf(headers, _textColumn);
            if (textColIndex < 0)
                textColIndex = 0;
        }
        
        string? line;
        while ((line = await reader.ReadLineAsync(cancellationToken)) is not null)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            if (string.IsNullOrWhiteSpace(line))
                continue;

            var fields = ParseCsvLine(line);
            
            if (textColIndex < fields.Length)
            {
                string text = fields[textColIndex];
                if (!string.IsNullOrWhiteSpace(text))
                    yield return text;
            }
        }
    }

    private string[] ParseCsvLine(string line)
    {
        var fields = new List<string>();
        var current = new System.Text.StringBuilder();
        bool inQuotes = false;
        
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            
            if (c == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                {
                    current.Append('"');
                    i++;
                }
                else
                {
                    inQuotes = !inQuotes;
                }
            }
            else if (c == _delimiter && !inQuotes)
            {
                fields.Add(current.ToString());
                current.Clear();
            }
            else
            {
                current.Append(c);
            }
        }
        
        fields.Add(current.ToString());
        return [.. fields];
    }

    public ValueTask DisposeAsync()
    {
        return ValueTask.CompletedTask;
    }
}

public sealed class CsvDataSource<T>(string path) : IDataSource
{
    private readonly string _path = path;
    public long? EstimatedCount => null;
    public string Description => $"CSV<T>: {_path}";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var reader = new StreamReader(_path);
        
        await reader.ReadLineAsync(cancellationToken);
        
        string? line;
        while ((line = await reader.ReadLineAsync(cancellationToken)) is not null)
        {
            if (cancellationToken.IsCancellationRequested)
                break;
            
            yield return line;
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}