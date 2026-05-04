using System.Runtime.CompilerServices;
using Parquet;
using Parquet.Data;
using SharpMind.Data.Sources;

namespace SharpMind.Data.Parquet.Sources;

/// <summary>
/// Streams documents from Parquet files.
/// Extracts a specific column as the text content.
/// </summary>
public sealed class ParquetSource : IDataSource
{
    private readonly string[] _paths;
    private readonly string _textField;

    public ParquetSource(string path, string textField = "text")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(textField);
        _textField = textField;
        _paths = GlobResolver.Resolve(path);

        if (_paths.Length == 0)
            throw new FileNotFoundException($"No files matched: {path}");
    }

    public ParquetSource(IEnumerable<string> paths, string textField = "text")
    {
        ArgumentNullException.ThrowIfNull(paths);
        ArgumentException.ThrowIfNullOrWhiteSpace(textField);
        _textField = textField;
        _paths = GlobResolver.ResolveMany(paths);

        if (_paths.Length == 0)
            throw new ArgumentException("Path list must not be empty.", nameof(paths));
    }

    public long? EstimatedCount => null;

    public string Description =>
        _paths.Length == 1
            ? $"Parquet({Path.GetFileName(_paths[0])}, field={_textField})"
            : $"Parquet({_paths.Length} files, field={_textField})";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (string path in _paths)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var stream = File.OpenRead(path);
            using var reader = await ParquetReader.CreateAsync(stream, null, true, cancellationToken);

            // Find the field by name
            var field = reader.Schema.DataFields.FirstOrDefault(f => f.Name == _textField);
            if (field == null) continue;

            // Parquet files are divided into row groups. We must read each one.
            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var rowGroupReader = reader.OpenRowGroupReader(i);
                DataColumn column = await rowGroupReader.ReadColumnAsync(field, cancellationToken);

                for (int j = 0; j < column.Data.Length; j++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    object? value = column.Data.GetValue(j);
                    if (value == null) continue;

                    string? text = FormatValue(value);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        yield return text;
                    }
                }
            }
        }
    }

    private static string? FormatValue(object value)
    {
        if (value is string s) return s;

        // Handle lists/arrays (e.g. conversations)
        if (value is System.Collections.IEnumerable enumerable)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var item in enumerable)
            {
                if (item == null) continue;
                sb.AppendLine(item.ToString());
            }
            return sb.ToString().TrimEnd();
        }

        return value.ToString();
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
