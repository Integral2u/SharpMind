using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    public sealed class TensorShapeTests
    {
        [Fact]
        public void ElementCount_CorrectForAllRanks()
        {
            Assert.Equal(6, new TensorShape(2, 3).ElementCount);
            Assert.Equal(24, new TensorShape(2, 3, 4).ElementCount);
            Assert.Equal(120, new TensorShape(2, 3, 4, 5).ElementCount);
        }

        [Fact]
        public void Strides_RowMajor()
        {
            var s = new TensorShape(3, 4, 5);
            Assert.Equal([20, 5, 1], s.Strides.ToArray());
        }

        [Fact]
        public void GetOffset_2D_MatchesRowMajor()
        {
            var s = new TensorShape(4, 5);
            Assert.Equal(7, s.GetOffset(1, 2));   // 1*5 + 2
            Assert.Equal(17, s.GetOffset(3, 2));   // 3*5 + 2
        }

        [Fact]
        public void NegativeIndex_LastDim()
        {
            var s = new TensorShape(3, 4, 5);
            Assert.Equal(5, s[-1]);
            Assert.Equal(4, s[-2]);
            Assert.Equal(3, s[-3]);
        }

        [Theory]
        [InlineData(new[] { 2, 3 }, new[] { 6 })]
        [InlineData(new[] { 2, 3 }, new[] { 3, 2 })]
        [InlineData(new[] { 6 }, new[] { 2, 3 })]
        public void Reshape_PreservesElementCount(int[] from, int[] to)
        {
            var s = new TensorShape(from);
            var r = s.Reshape(to);
            Assert.Equal(s.ElementCount, r.ElementCount);
        }

        [Fact]
        public void Reshape_InferredDim()
        {
            var s = new TensorShape(2, 6);
            var r = s.Reshape(4, -1);
            Assert.Equal(3, r[1]);
        }

        [Fact]
        public void Reshape_IncompatibleCount_Throws()
        {
            var s = new TensorShape(2, 3);
            Assert.Throws<ArgumentException>(() => s.Reshape(5));
        }

        [Fact]
        public void Unsqueeze_InsertsUnitDim()
        {
            var s = new TensorShape(3, 4);
            var u = s.Unsqueeze(0);
            Assert.Equal(new TensorShape(1, 3, 4), u);
        }

        [Fact]
        public void Squeeze_RemovesUnitDim()
        {
            var s = new TensorShape(1, 3, 4);
            var q = s.Squeeze(0);
            Assert.Equal(new TensorShape(3, 4), q);
        }

        [Fact]
        public void AssertSameShape_ThrowsOnMismatch()
        {
            var a = new TensorShape(2, 3);
            var b = new TensorShape(3, 2);
            Assert.Throws<ArgumentException>(() => TensorShape.AssertSameShape(a, b));
        }

        [Fact]
        public void AssertMatMulCompatible_ThrowsOnInnerDimMismatch()
        {
            Assert.Throws<ArgumentException>(() =>
                TensorShape.AssertMatMulCompatible(new TensorShape(2, 3), new TensorShape(4, 5)));
        }
    }
}
