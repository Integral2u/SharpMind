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
        public void Apply_AdjacentPairing_MatchesLlamaCppConvention()
        {
            // llama.cpp pairs adjacent dims (2i, 2i+1) with freq theta^{-2i/headDim}.
            // headDim=4, theta=10000: pair0 angle = 1*1.0, pair1 angle = 1*theta^{-1/2}.
            var rope = new RoPE(headDim: 4, maxSeqLen: 8, theta: 10_000f);
            using var x = Tensor<float>.From([1f, 2f, 3f, 4f], 1, 1, 4);
            rope.Apply(x, positionOffset: 1);

            double th1 = 1.0;                      // theta^{-2*0/4}
            double th2 = MathF.Pow(10_000f, -0.5f); // theta^{-2*1/4}
            Assert.Equal(1.0 * Math.Cos(th1) - 2.0 * Math.Sin(th1), x[0], precision: 5);
            Assert.Equal(2.0 * Math.Cos(th1) + 1.0 * Math.Sin(th1), x[1], precision: 5);
            Assert.Equal(3.0 * Math.Cos(th2) - 4.0 * Math.Sin(th2), x[2], precision: 5);
            Assert.Equal(4.0 * Math.Cos(th2) + 3.0 * Math.Sin(th2), x[3], precision: 5);
        }

        [Fact]
        public void Apply_AdjacentPairing_AVX2Path_MatchesLlamaCppConvention()
        {
            // headDim=8 exercises the AVX2 vectorized path (ropePairs=4).
            // Pair i uses adjacent dims (2i, 2i+1) and freq theta^{-2i/8}.
            var rope = new RoPE(headDim: 8, maxSeqLen: 8, theta: 10_000f);
            using var x = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], 1, 1, 8);
            rope.Apply(x, positionOffset: 1);

            for (int i = 0; i < 4; i++)
            {
                double th = MathF.Pow(10_000f, -2.0f * i / 8.0f);
                double x0 = x[2 * i], x1 = x[2 * i + 1];
                double ex0 = (2 * i + 1) * Math.Cos(th) - (2 * i + 2) * Math.Sin(th);
                double ex1 = (2 * i + 2) * Math.Cos(th) + (2 * i + 1) * Math.Sin(th);
                Assert.Equal(ex0, x0, precision: 4);
                Assert.Equal(ex1, x1, precision: 4);
            }
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
