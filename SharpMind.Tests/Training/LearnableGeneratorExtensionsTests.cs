using SharpMind.Core.Tensors;
using SharpMind.Data.Batching;
using SharpMind.Data.Sources.PseudoLanguage;

namespace SharpMind.Tests.Training;

/// <summary>
/// Covers the <see cref="LearnableGenerator"/> → <see cref="TrainingBatch"/>
/// adapter: noun-verb-noun samples are 3 tokens, so with seqLen 4 the last real
/// position and the padding column must be masked and labelled -100.
/// </summary>
public class LearnableGeneratorExtensionsTests
{
    [Fact]
    public async Task ToTrainingBatches_ProducesShiftedLabelsMaskAndPadding()
    {
        var generator = new LearnableGenerator(new LearnableConfig(), new Random(1));

        await using var enumerator = generator.ToTrainingBatches(batchSize: 2, seqLen: 4).GetAsyncEnumerator();
        Assert.True(await enumerator.MoveNextAsync());
        TrainingBatch batch = enumerator.Current;

        try
        {
            Assert.Equal((2, 4), (batch.TokenIds.Shape.Rows, batch.TokenIds.Shape.Cols));
            Assert.Equal((2, 4), (batch.Labels.Shape.Rows, batch.Labels.Shape.Cols));
            Assert.Equal((2, 4), (batch.AttentionMask.Shape.Rows, batch.AttentionMask.Shape.Cols));

            for (int b = 0; b < 2; b++)
            {
                int take = 3; // noun-verb-noun samples are exactly 3 tokens
                for (int s = 0; s < take; s++)
                {
                    int id = batch.TokenIds.Data[b * 4 + s];
                    Assert.InRange(id, 0, 63);
                    Assert.Equal(1f, batch.AttentionMask.Data[b * 4 + s]);
                    int expected = s + 1 < take ? batch.TokenIds.Data[b * 4 + s + 1] : -100;
                    Assert.Equal(expected, batch.Labels.Data[b * 4 + s]);
                }

                // Padding column: token 0, ignored label, no mask.
                Assert.Equal(0, batch.TokenIds.Data[b * 4 + 3]);
                Assert.Equal(-100, batch.Labels.Data[b * 4 + 3]);
                Assert.Equal(0f, batch.AttentionMask.Data[b * 4 + 3]);
            }

            Assert.Equal(6, batch.RealTokenCount);
        }
        finally
        {
            batch.Dispose();
        }
    }
}
