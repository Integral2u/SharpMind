using System.Runtime.CompilerServices;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Records;
using SharpMind.Data.Sources;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;
using SharpMind.Training.Sft;
using Xunit;

namespace SharpMind.Tests.Training;

public sealed class RecordDataLoaderTests
{
    private const string ChatMlTokens =
        "<|im_start|><|im_end|><|system|><|user|><|assistant|>";

    private static async Task<Tokenizer> CharTokenizerAsync()
    {
        var docs = new ListSource(
        [
            "abcxyz ABCXYZ 0123 \n\t<>|/_-.:!?",
            "hello world hi pong q1 q2 a1 a2 s user assistant model system",
        ]);
        return await TokenizationPipeline.TrainCharacterAsync(docs);
    }

    [Fact]
    public async Task LoadAsync_ProducesSftBatchesAndResumesAfterASkip()
    {
        Tokenizer tokenizer = await CharTokenizerAsync();
        var conversation = new ConversationTokenizer(
            new ChatMLFormatter(ChatMlTokens), tokenizer, addBos: false);
        var batcher = new SftBatcher(batchSize: 1, maxSeqLen: 128, padTokenId: tokenizer.PadId);

        var records = new ListRecordSource(
        [
            new DataRecord
            {
                Text = "",
                Turns =
                [
                    new ConversationTurn("user", "hi"),
                    new ConversationTurn("assistant", "pong"),
                ],
            },
            new DataRecord
            {
                Text = "",
                Turns =
                [
                    new ConversationTurn("user", "hello"),
                    new ConversationTurn("assistant", "world"),
                ],
            },
        ]);

        var loader = new RecordDataLoader(records, batcher, conversation.Encode, maxBatches: 2);
        var batches = new List<TrainingBatch>();
        await foreach (TrainingBatch batch in loader.LoadAsync())
            batches.Add(batch);

        Assert.Equal(2, batches.Count);
        Assert.Contains(batches, batch => batch.Labels.Data.ToArray().Any(value => value != -100));
        foreach (TrainingBatch batch in batches) batch.Dispose();

        var skipped = new List<TrainingBatch>();
        await foreach (TrainingBatch batch in loader.LoadAsync(skipBatches: 1))
            skipped.Add(batch);

        Assert.Single(skipped);
        foreach (TrainingBatch batch in skipped) batch.Dispose();
    }

    private sealed class ListRecordSource(IEnumerable<DataRecord> records) : IRecordSource
    {
        public long? EstimatedCount => null;
        public string Description => "test records";

        public async IAsyncEnumerable<DataRecord> ReadRecordsAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (DataRecord record in records)
            {
                await Task.Yield();
                yield return record;
            }
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class ListSource(IEnumerable<string> docs) : IDataSource
    {
        public long? EstimatedCount => null;
        public string Description => "test";

        public async IAsyncEnumerable<string> ReadAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (string doc in docs)
            {
                await Task.Yield();
                yield return doc;
            }
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
