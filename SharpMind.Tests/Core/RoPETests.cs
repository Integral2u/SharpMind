using SharpMind.Core.Embeddings;
using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    public sealed class RoPETests
    {
        [Fact]
        public void Apply_PreservesShape()
        {
            var rope = new RoPE(headDim: 8, maxSeqLen: 16);
            using var x = Tensor<float>.Ones(4, 2, 8);   // [SeqLen=4, NumHeads=2, HeadDim=8]
            rope.Apply(x);
            Assert.Equal([4, 2, 8], x.Shape.Dims.ToArray());
        }

        [Fact]
        public void Apply_ZeroPosition_CosSin0()
        {
            // At position 0, angle = 0 → cos=1, sin=0 → vector unchanged
            var rope = new RoPE(headDim: 4, maxSeqLen: 8);
            using var x = Tensor<float>.From([1f, 2f, 3f, 4f], 1, 1, 4);
            var before = x.Data.ToArray().ToArray();
            rope.Apply(x, positionOffset: 0);
            Assert.Equal(before[0], x[0], precision: 5);  // x0 unchanged at pos=0
            Assert.Equal(before[1], x[1], precision: 5);
        }

        [Fact]
        public void Apply_OutputNotAllSame()
        {
            // Different positions should produce different rotations
            var rope = new RoPE(headDim: 8, maxSeqLen: 16);
            using var xA = Tensor<float>.Ones(1, 1, 8);
            using var xB = Tensor<float>.Ones(1, 1, 8);
            rope.Apply(xA, positionOffset: 0);
            rope.Apply(xB, positionOffset: 5);
            bool anyDiff = false;
            for (int i = 0; i < 8; i++)
                if (MathF.Abs(xA[i] - xB[i]) > 1e-5f) { anyDiff = true; break; }
            Assert.True(anyDiff, "RoPE at different positions should differ.");
        }

        [Fact]
        public void Apply_ExceedsMaxSeqLen_Throws()
        {
            var rope = new RoPE(headDim: 4, maxSeqLen: 8);
            using var x = Tensor<float>.Ones(5, 1, 4);   // seqLen=5
            Assert.Throws<ArgumentOutOfRangeException>(() => rope.Apply(x, positionOffset: 4));
        }

        [Fact]
        public void Apply_WrongHeadDim_Throws()
        {
            var rope = new RoPE(headDim: 8, maxSeqLen: 16);
            using var x = Tensor<float>.Ones(1, 1, 4);   // headDim=4, expected 8
            Assert.Throws<ArgumentException>(() => rope.Apply(x));
        }
    }
}
