using SharpMind.Core.Embeddings;

namespace SharpMind.Tests.Core
{
    public sealed class EmbeddingTableTests
    {
        [Fact]
        public void Forward_ReturnsCorrectRows()
        {
            var table = new EmbeddingTable(vocabSize: 4, embeddingDim: 3);
            table.LoadWeights([
                1f, 2f, 3f,   // token 0
            4f, 5f, 6f,   // token 1
            7f, 8f, 9f,   // token 2
            10f, 11f, 12f // token 3
            ]);

            using var r = table.Forward([2, 0, 3]);
            Assert.Equal([3, 3], r.Shape.Dims.ToArray());
            Assert.Equal([7f, 8f, 9f], r.RowSpan(0).ToArray());
            Assert.Equal([1f, 2f, 3f], r.RowSpan(1).ToArray());
            Assert.Equal([10f, 11f, 12f], r.RowSpan(2).ToArray());
            table.Dispose();
        }

        [Fact]
        public void Forward_OutOfRangeId_Throws()
        {
            var table = new EmbeddingTable(4, 3);
            Assert.Throws<ArgumentOutOfRangeException>(() => table.Forward([0, 4]));
            table.Dispose();
        }

        [Fact]
        public void Dispose_Safe()
        {
            var table = new EmbeddingTable(10, 8);
            var ex = Record.Exception(() => { table.Dispose(); table.Dispose(); });
            Assert.Null(ex);
        }
    }
}
