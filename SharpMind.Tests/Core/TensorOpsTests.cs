using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    [Collection("Non-Parallel")]
    public sealed class TensorOpsTests
    {

        // Elementwise

        [Fact]
        public void Add_KnownValues()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f], 3);
            using var b = Tensor<float>.From([4f, 5f, 6f], 3);
            using var r = a.Add(b);
            Assert.Equal([5f, 7f, 9f], r.Data.ToArray());
        }

        [Fact]
        public void Subtract_KnownValues()
        {
            using var a = Tensor<float>.From([5f, 7f, 9f], 3);
            using var b = Tensor<float>.From([1f, 2f, 3f], 3);
            using var r = a.Subtract(b);
            Assert.Equal([4f, 5f, 6f], r.Data.ToArray());
        }

        [Fact]
        public void Multiply_Elementwise()
        {
            using var a = Tensor<float>.From([2f, 3f, 4f], 3);
            using var b = Tensor<float>.From([5f, 6f, 7f], 3);
            using var r = a.Multiply(b);
            Assert.Equal([10f, 18f, 28f], r.Data.ToArray());
        }

        [Fact]
        public void Scale_ScalesAllElements()
        {
            using var a = Tensor<float>.From([1f, 2f, 4f], 3);
            using var r = a.Scale(3f);
            Assert.Equal([3f, 6f, 12f], r.Data.ToArray());
        }

        [Fact]
        public void Clamp_ClampsToRange()
        {
            using var a = Tensor<float>.From([-2f, 0f, 1f, 5f], 4);
            using var r = a.Clamp(0f, 3f);
            Assert.Equal([0f, 0f, 1f, 3f], r.Data.ToArray());
        }

        [Fact]
        public void Sqrt_Correct()
        {
            using var a = Tensor<float>.From([1f, 4f, 9f, 16f], 4);
            using var r = a.Sqrt();
            Assert.Equal([1f, 2f, 3f, 4f], r.Data.ToArray());
        }

        [Fact]
        public void Abs_NegativeValues()
        {
            using var a = Tensor<float>.From([-1f, 2f, -3f], 3);
            using var r = a.Abs();
            Assert.Equal([1f, 2f, 3f], r.Data.ToArray());
        }

        [Fact]
        public void MaskedFill_FillsSelectedPositions()
        {
            using var a = Tensor<float>.Ones(4);
            bool[] mask = [false, true, false, true];
            a.MaskedFill(mask, float.NegativeInfinity);
            Assert.Equal(1f, a[0]);
            Assert.Equal(float.NegativeInfinity, a[1]);
            Assert.Equal(1f, a[2]);
            Assert.Equal(float.NegativeInfinity, a[3]);
        }

        [Fact]
        public void Sum_Correct()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f], 4);
            Assert.Equal(10f, a.Sum());
        }

        [Fact]
        public void Mean_Correct()
        {
            using var a = Tensor<float>.From([2f, 4f, 6f, 8f], 4);
            Assert.Equal(5f, a.Mean());
        }

        [Fact]
        public void Variance_Correct()
        {
            using var a = Tensor<float>.From([2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f], 8);
            float v = a.Variance();
            Assert.Equal(4f, v, precision: 4);
        }

        [Fact]
        public void ArgMax_ReturnsCorrectIndex()
        {
            using var a = Tensor<float>.From([1f, 5f, 3f, 2f], 4);
            Assert.Equal(1, a.ArgMax());
        }

        [Fact]
        public void Transpose_2D()
        {
            using var a = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
            using var t = a.Transpose();
            Assert.Equal([3, 2], t.Shape.Dims.ToArray());
            Assert.Equal(1f, t[0, 0]);
            Assert.Equal(4f, t[0, 1]);
            Assert.Equal(2f, t[1, 0]);
        }

    }
}
