using System.Globalization;
using System.Runtime.CompilerServices;
using Parquet;
using Parquet.Schema;
using SharpMind.Data.Sources;

namespace SharpMind.Data.Parquet.Sources;

/// <summary>
/// Streams documents from Parquet files.
/// Extracts a specific column as the text content. Values are read with the
/// schema-typed <see cref="ParquetRowGroupReader.ReadAsync{T}"/> overloads, so
/// the CLR buffers always match the column's declared type, then stringified
/// with the invariant culture ("R" round-trip format for floats).
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
            await using var reader = await ParquetReader.CreateAsync(stream, null, true, cancellationToken);

            DataField[] dataFields = reader.Schema.DataFields;
            DataField field = dataFields.FirstOrDefault(
                f => string.Equals(f.Name, _textField, StringComparison.OrdinalIgnoreCase))
                ?? throw new InvalidOperationException(
                    $"Parquet file '{path}' has no '{_textField}' column. " +
                    $"Available columns: {string.Join(", ", dataFields.Select(f => f.Name))}.");

            for (int i = 0; i < reader.RowGroupCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                using var rowGroupReader = reader.OpenRowGroupReader(i);
                string[] rows = await ReadColumnToStrings(rowGroupReader, field, cancellationToken);

                foreach (string row in rows)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!string.IsNullOrWhiteSpace(row))
                        yield return row;
                }
            }
        }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    /// <summary>
    /// Reads one column into an array typed by the schema's <see cref="DataField.ClrType"/>
    /// and stringifies every value. Unsupported column types surface as
    /// <see cref="NotSupportedException"/>.
    /// </summary>
    private static async Task<string[]> ReadColumnToStrings(
        ParquetRowGroupReader reader, DataField field, CancellationToken ct)
    {
        int n = (int)reader.RowCount;
        Type type = field.ClrType;

        if (type == typeof(string))
        {
            var buffer = new string[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return buffer;
        }
        if (type == typeof(byte[]))
        {
            var buffer = new byte[n][];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(bool))
        {
            var buffer = new bool[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(byte))
        {
            var buffer = new byte[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(sbyte))
        {
            var buffer = new sbyte[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(short))
        {
            var buffer = new short[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(ushort))
        {
            var buffer = new ushort[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(int))
        {
            var buffer = new int[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(uint))
        {
            var buffer = new uint[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(long))
        {
            var buffer = new long[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(ulong))
        {
            var buffer = new ulong[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(float))
        {
            var buffer = new float[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(double))
        {
            var buffer = new double[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(decimal))
        {
            var buffer = new decimal[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(DateTime))
        {
            var buffer = new DateTime[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(DateTimeOffset))
        {
            var buffer = new DateTimeOffset[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(TimeSpan))
        {
            var buffer = new TimeSpan[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }
        if (type == typeof(Guid))
        {
            var buffer = new Guid[n];
            await reader.ReadAsync(field, buffer.AsMemory(), null, ct);
            return BufferToStrings(buffer);
        }

        throw new NotSupportedException(
            $"Parquet column '{field.Name}' has unsupported CLR type '{type}'.");
    }

    private static string[] BufferToStrings<T>(T[] buffer) where T : notnull
        => [.. buffer.Select(v => RowToString(v))];

    private static string RowToString(object? value) => value switch
    {
        null => "",
        string s => s,
        byte[] bytes => Convert.ToBase64String(bytes),
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        decimal m => m.ToString(CultureInfo.InvariantCulture),
        DateTime dt => dt.ToString("O", CultureInfo.InvariantCulture),
        DateTimeOffset dto => dto.ToString("O", CultureInfo.InvariantCulture),
        TimeSpan ts => ts.ToString("c", CultureInfo.InvariantCulture),
        Guid g => g.ToString("D", CultureInfo.InvariantCulture),
        bool b => b ? "true" : "false",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? ""
    };
}