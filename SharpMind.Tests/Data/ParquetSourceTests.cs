using Parquet;
using Parquet.Schema;
using SharpMind.Data.Parquet.Sources;

namespace SharpMind.Tests.Data;

[Collection("Non-Parallel")]
public sealed class ParquetSourceTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private readonly DataField<long> _fromField = new("from");
    private readonly DataField<float> _valueField = new("value");
    private readonly DataField<string> _textField = new("text");

    private async Task<string> WriteSampleFile()
    {
        string path = Path.Combine(_dir.Path, "sample.parquet");
        var schema = new ParquetSchema(_fromField, _valueField, _textField);

        await using var stream = File.Create(path);
        await using var writer = await ParquetWriter.CreateAsync(schema, stream);
        using var rg = writer.CreateRowGroup();

        await rg.WriteAsync(_fromField, (ReadOnlyMemory<long>)new long[] { 1, 2, 3, 4 });
        await rg.WriteAsync(_valueField, (ReadOnlyMemory<float>)new float[] { 1.5f, 0.000000123456789f, -0.125f, -42.75f });
        await rg.WriteAsync(_textField, new string[] { "hello", "world", "third", "fourth" });
        return path;
    }

    private static async Task<string[]> ReadAll(string path, string field)
    {
        var source = new ParquetSource(path, field);
        var rows = new List<string>();
        await foreach (string row in source.ReadAsync())
            rows.Add(row);
        return [.. rows];
    }

    [Fact]
    public async Task TextColumn_YieldsStringRows()
    {
        string path = await WriteSampleFile();

        string[] rows = await ReadAll(path, "text");

        Assert.Equal(["hello", "world", "third", "fourth"], rows);
    }

    [Fact]
    public async Task LongColumn_YieldsInvariantStrings()
    {
        string path = await WriteSampleFile();

        string[] rows = await ReadAll(path, "from");

        Assert.Equal(["1", "2", "3", "4"], rows);
    }

    [Fact]
    public async Task FloatColumn_YieldsRoundTripStrings()
    {
        string path = await WriteSampleFile();

        string[] rows = await ReadAll(path, "value");

        Assert.Equal(["1.5", "1.2345679E-07", "-0.125", "-42.75"], rows);
    }

    [Fact]
    public async Task MissingColumn_Throws()
    {
        string path = await WriteSampleFile();

        var source = new ParquetSource(path, "nope");

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            async () =>
            {
                await foreach (string _ in source.ReadAsync()) { }
            });
        Assert.Contains("nope", ex.Message);
        Assert.Contains("from", ex.Message);
        Assert.Contains("value", ex.Message);
        Assert.Contains("text", ex.Message);
    }

    [Fact]
    public async Task FieldLookup_IsCaseInsensitive()
    {
        string path = await WriteSampleFile();

        string[] rows = await ReadAll(path, "TEXT");

        Assert.Equal(["hello", "world", "third", "fourth"], rows);
    }
}