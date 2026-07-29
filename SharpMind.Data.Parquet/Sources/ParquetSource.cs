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
            var reader = await ParquetReader.CreateAsync(stream, null, true, cancellationToken);
            
            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var rowGroupReader = reader.OpenRowGroupReader(i);
                var dataFields = reader.Schema.GetDataFields();

                // Find the data field schemas by their names
                var fromField = dataFields.FirstOrDefault(f => f.Name == "from");
                var valueField = dataFields.FirstOrDefault(f => f.Name == "value");
                var sourceField = dataFields.FirstOrDefault(f => f.Name == "source");

                int rowCount = (int)rowGroupReader.RowCount;

                // 1. Read "from" column
                object[]? fromCol = null;
                if (fromField != null)
                {
                    // Swap string[] for whatever type 'from' is (e.g., int[], long[])
                    string[] buffer = new string[rowCount];
                    await rowGroupReader.ReadAsync(fromField, buffer,null, cancellationToken);
                    fromCol = [.. buffer.Cast<object>()];
                }

                // 2. Read "value" column
                object[]? valueCol = null;
                if (valueField != null)
                {
                    // Swap double[] for whatever type 'value' is (e.g., decimal[], float[])
                    double[] buffer = new double[rowCount];
                    await rowGroupReader.ReadAsync<double>(valueField, buffer, null, cancellationToken);
                    valueCol = [.. buffer.Cast<object>()];
                }

                // 3. Read "source" column
                object[]? sourceCol = null;
                if (sourceField != null)
                {
                    string[] buffer = new string[rowCount];
                    await rowGroupReader.ReadAsync(sourceField, buffer, null, cancellationToken);
                    sourceCol = [.. buffer.Cast<object>()];
                }

                if (fromCol != null && valueCol != null)
                {
                    var froms = (string[])fromCol;
                    var values = (string[])valueCol;

                    for (int j = 0; j < froms.Length; j++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        yield return $"{froms[j]}: {values[j]}";
                    }
                }
                else if (_textField == "source" && sourceCol != null)
                {
                    var sources = (string[])sourceCol;
                    foreach (var s in sources)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        if (!string.IsNullOrWhiteSpace(s))
                            yield return s;
                    }
                }
            }
            await reader.DisposeAsync();
        }
    }
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

}
