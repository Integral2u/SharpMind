using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    /// <summary>
    /// Creates a scalar-only TensorOps for all correctness tests so results
    /// are deterministic regardless of host CPU capabilities. A separate
    /// parity test verifies that the AVX2/FMA paths produce identical output.
    /// </summary>
    [Collection("Non-Parallel")]
    public sealed class TensorOpsTests : IDisposable
    {
        private readonly TensorOps _ops;

        public TensorOpsTests()
        {
            var config = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
            _ops = TensorOpsFactory.Create(config);
        }

        public void Dispose() { }

        // MatMul

        [Fact]
        public void MatMul_Identity_ReturnsInputUnchanged()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
            using var id = Tensor<float>.Eye(3);
            using var c = _ops.MatMul(a, id);
            AssertClose(a.Data, c.Data);
        }

        [Fact]
        public void MatMul_KnownValues()
        {
            // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
            // [3 4] × [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f], 2, 2);
            using var b = Tensor<float>.From([5f, 6f, 7f, 8f], 2, 2);
            using var c = _ops.MatMul(a, b);
            Assert.Equal([19f, 22f, 43f, 50f], c.Data.ToArray());
        }

        [Fact]
        public void MatMul_NonSquare()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);   // [2,3]
            using var b = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 3, 2);   // [3,2]
            using var c = _ops.MatMul(a, b);
            Assert.Equal([2, 2], c.Shape.Dims.ToArray());
            Assert.Equal(22f, c[0, 0]);  // 1+4+9
            Assert.Equal(28f, c[0, 1]);  // 2+8+16 → actually 1*2+2*4+3*6=2+8+18=28
            Assert.Equal(49f, c[1, 0]);  // 4*1+5*3+6*5=4+15+30=49
        }

        [Fact]
        public void MatMulInto_WritesToPreallocated()
        {
            using var a = Tensor<float>.From([1f, 0f, 0f, 1f], 2, 2);  // identity
            using var b = Tensor<float>.From([3f, 7f, 2f, 5f], 2, 2);
            using var c = new Tensor<float>(2, 2);
            _ops.MatMulInto(a, b, c);
            Assert.Equal(3f, c[0, 0]);
            Assert.Equal(5f, c[1, 1]);
        }

        [Fact]
        public void BatchedMatMul_Shape()
        {
            using var a = Tensor<float>.Ones(2, 3, 4);   // [B=2, M=3, K=4]
            using var b = Tensor<float>.Ones(2, 4, 5);   // [B=2, K=4, N=5]
            using var c = _ops.BatchedMatMul(a, b);
            Assert.Equal([2, 3, 5], c.Shape.Dims.ToArray());
        }

        [Fact]
        public void BatchedMatMul_Values()
        {
            // Each 2×2 block: ones × ones with K=2 → each cell = 2
            using var a = Tensor<float>.Ones(3, 2, 2);
            using var b = Tensor<float>.Ones(3, 2, 2);
            using var c = _ops.BatchedMatMul(a, b);
            Assert.All(c.Data.ToArray(), v => Assert.Equal(2f, v));
        }

        [Fact]
        public void BatchedMatMul_IncompatibleDims_Throws()
        {
            using var a = Tensor<float>.Ones(2, 3, 4);
            using var b = Tensor<float>.Ones(2, 5, 6);  // K mismatch
            Assert.Throws<ArgumentException>(() => _ops.BatchedMatMul(a, b));
        }

        // Elementwise

        [Fact]
        public void Add_KnownValues()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f], 3);
            using var b = Tensor<float>.From([4f, 5f, 6f], 3);
            using var r = TensorOps.Add(a, b);
            Assert.Equal([5f, 7f, 9f], r.Data.ToArray());
        }

        [Fact]
        public void Subtract_KnownValues()
        {
            using var a = Tensor<float>.From([5f, 7f, 9f], 3);
            using var b = Tensor<float>.From([1f, 2f, 3f], 3);
            using var r = TensorOps.Subtract(a, b);
            Assert.Equal([4f, 5f, 6f], r.Data.ToArray());
        }

        [Fact]
        public void Multiply_Elementwise()
        {
            using var a = Tensor<float>.From([2f, 3f, 4f], 3);
            using var b = Tensor<float>.From([5f, 6f, 7f], 3);
            using var r = TensorOps.Multiply(a, b);
            Assert.Equal([10f, 18f, 28f], r.Data.ToArray());
        }

        [Fact]
        public void Scale_ScalesAllElements()
        {
            using var a = Tensor<float>.From([1f, 2f, 4f], 3);
            using var r = TensorOps.Scale(a, 3f);
            Assert.Equal([3f, 6f, 12f], r.Data.ToArray());
        }

        [Fact]
        public void Clamp_ClampsToRange()
        {
            using var a = Tensor<float>.From([-2f, 0f, 1f, 5f], 4);
            using var r = TensorOps.Clamp(a, 0f, 3f);
            Assert.Equal([0f, 0f, 1f, 3f], r.Data.ToArray());
        }

        [Fact]
        public void Sqrt_Correct()
        {
            using var a = Tensor<float>.From([1f, 4f, 9f, 16f], 4);
            using var r = TensorOps.Sqrt(a);
            Assert.Equal([1f, 2f, 3f, 4f], r.Data.ToArray());
        }

        [Fact]
        public void Abs_NegativeValues()
        {
            using var a = Tensor<float>.From([-1f, 2f, -3f], 3);
            using var r = TensorOps.Abs(a);
            Assert.Equal([1f, 2f, 3f], r.Data.ToArray());
        }

        [Fact]
        public void MaskedFill_FillsSelectedPositions()
        {
            using var a = Tensor<float>.Ones(4);
            bool[] mask = [false, true, false, true];
            TensorOps.MaskedFill(a, mask, float.NegativeInfinity);
            Assert.Equal(1f, a[0]);
            Assert.Equal(float.NegativeInfinity, a[1]);
            Assert.Equal(1f, a[2]);
            Assert.Equal(float.NegativeInfinity, a[3]);
        }

        [Fact]
        public void Sum_Correct()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f], 4);
            Assert.Equal(10f, TensorOps.Sum(a));
        }

        [Fact]
        public void Mean_Correct()
        {
            using var a = Tensor<float>.From([2f, 4f, 6f, 8f], 4);
            Assert.Equal(5f, TensorOps.Mean(a));
        }

        [Fact]
        public void Variance_Correct()
        {
            using var a = Tensor<float>.From([2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f], 8);
            float v = TensorOps.Variance(a);
            Assert.Equal(4f, v, precision: 4);  // known variance = 4
        }

        [Fact]
        public void ArgMax_ReturnsCorrectIndex()
        {
            using var a = Tensor<float>.From([1f, 5f, 3f, 2f], 4);
            Assert.Equal(1, TensorOps.ArgMax(a));
        }

        [Fact]
        public void ArgTopK_ReturnsKIndices()
        {
            using var a = Tensor<float>.From([1f, 5f, 3f, 9f, 2f], 5);
            var topK = TensorOps.ArgTopK(a, 2);
            Assert.Equal(2, topK.Length);
            Assert.Contains(3, topK);  // value 9
            Assert.Contains(1, topK);  // value 5
        }

        [Fact]
        public void Transpose_2D()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
            using var t = TensorOps.Transpose(a);
            Assert.Equal([3, 2], t.Shape.Dims.ToArray());
            Assert.Equal(1f, t[0, 0]);
            Assert.Equal(4f, t[0, 1]);
            Assert.Equal(2f, t[1, 0]);
        }

        // Kernel parity

        [Fact]
        public void ScalarAndAvx2MatMul_ProduceIdenticalResults()
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
                return;  // skip if CPU lacks AVX2

            using var a = Tensor<float>.Ones(32, 64);
            using var b = Tensor<float>.Ones(64, 32);

            var scalarOps = TensorOpsFactory.Create(
                SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });
            var avx2Ops = TensorOpsFactory.Create(
                SharpMindConfig.Gpt with { Hardware = HardwareTier.AVX2 });

            using var cScalar = scalarOps.MatMul(a, b);
            using var cAvx2 = avx2Ops.MatMul(a, b);

            AssertClose(cScalar.Data, cAvx2.Data, tolerance: 1e-4f);
        }

        // Helpers

        private static void AssertClose(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tolerance = 1e-5f)
        {
            Assert.Equal(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a[i], b[i], tolerance);
        }
    }
}
