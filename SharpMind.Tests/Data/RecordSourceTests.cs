using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Records;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.Data;

/// <summary>
/// B4: the typed record path runs alongside the string path. Existing string
/// sources and stages cross into records and back through adapters, so a record
/// pipeline can reuse everything that already exists.
/// </summary>
public sealed class RecordSourceTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private string Source()
        => _dir.Write("corpus.txt", "alpha bravo\ncharlie delta\nHI\n");

    private static async Task<List<string>> ReadTexts(IRecordSource source)
    {
        var list = new List<string>();
        await foreach (var record in source.ReadRecordsAsync())
            list.Add(record.Text);
        return list;
    }

    private static async Task<List<string>> Drain(IDataSource source)
    {
        var list = new List<string>();
        await foreach (string doc in source.ReadAsync())
            list.Add(doc);
        return list;
    }

    [Fact]
    public async Task StringSource_AdaptsToRecords()
    {
        var source = new TextFileSource(Source()).ToRecordSource("corpus");

        var records = new List<DataRecord>();
        await foreach (var record in source.ReadRecordsAsync())
            records.Add(record);

        Assert.Equal(new[] { "alpha bravo", "charlie delta", "HI" }, records.Select(r => r.Text));
        Assert.All(records, r => Assert.Equal("corpus", r.Source));
        Assert.All(records, r => Assert.Null(r.Turns));
        Assert.All(records, r => Assert.Empty(r.Tags));
    }

    [Fact]
    public async Task Records_RoundTripBackToStrings()
    {
        var texts = await Drain(new TextFileSource(Source()).ToRecordSource().ToStringSource());

        Assert.Equal(new[] { "alpha bravo", "charlie delta", "HI" }, texts);
    }

    [Fact]
    public async Task ExistingStringStage_RunsOnRecords_Unchanged()
    {
        var records = new TextFileSource(Source())
            .ToRecordSource("corpus")
            .Pipe(new LowerCase());

        Assert.Equal(new[] { "alpha bravo", "charlie delta", "hi" }, await ReadTexts(records));
    }

    [Fact]
    public async Task DroppingStage_DropsTheWholeRecord()
    {
        // LowerCase keeps everything; MinLength(8) drops "hi" (and, after
        // lowercasing, nothing else).
        var records = new TextFileSource(Source())
            .ToRecordSource()
            .Pipe(new LowerCase())
            .Pipe(new MinLengthFilter(8));

        Assert.Equal(new[] { "alpha bravo", "charlie delta" }, await ReadTexts(records));
    }

    [Fact]
    public async Task Pipe_PreservesStructuredFields_WhenOnlyTextChanges()
    {
        var source = new ListRecordSource(
        [
            new DataRecord
            {
                Text = "Hello World",
                Turns = [new ConversationTurn("user", "Hello"), new ConversationTurn("assistant", "World")],
                Tags = ["sft", "english"],
                Source = "unit",
            },
        ]);

        var record = (await ReadAll(source.Pipe(new LowerCase())))[0];

        Assert.Equal("hello world", record.Text);
        Assert.Equal(2, record.Turns!.Count);
        Assert.Equal("user", record.Turns![0].Role);
        Assert.Equal(new[] { "sft", "english" }, record.Tags);
        Assert.Equal("unit", record.Source);
    }

    [Fact]
    public async Task RecordSource_ToStringSource_UsesText()
    {
        var source = new ListRecordSource(
        [
            new DataRecord
            {
                Text = "rendered transcript",
                Turns = [new ConversationTurn("user", "hi")],
            },
        ]);

        Assert.Equal(new[] { "rendered transcript" }, await Drain(source.ToStringSource()));
    }

    [Fact]
    public async Task RecordTransform_CanAttachTags()
    {
        var records = new TextFileSource(Source())
            .ToRecordSource()
            .Pipe("tag", r => r with { Tags = [.. r.Tags, "corpus"] });

        var all = await ReadAll(records);
        Assert.All(all, r => Assert.Contains("corpus", r.Tags));
    }

    private static async Task<List<DataRecord>> ReadAll(IRecordSource source)
    {
        var list = new List<DataRecord>();
        await foreach (var record in source.ReadRecordsAsync())
            list.Add(record);
        return list;
    }

    private sealed class ListRecordSource(IEnumerable<DataRecord> records) : IRecordSource
    {
        private readonly IReadOnlyList<DataRecord> _records = [.. records];

        public long? EstimatedCount => _records.Count;

        public string Description => "List";

        public async IAsyncEnumerable<DataRecord> ReadRecordsAsync(
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var record in _records)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return record;
            }
            await Task.CompletedTask;
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
