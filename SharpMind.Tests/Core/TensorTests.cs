using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    public sealed class TensorTests
    {
        [Fact]
        public void Zeros_AllZero()
        {
            using var t = Tensor<float>.Zeros(3, 4);
            Assert.Equal(0f, t.Data.ToArray().Sum());
        }

        [Fact]
        public void Ones_AllOne()
        {
            using var t = Tensor<float>.Ones(2, 5);
            Assert.All(t.Data.ToArray(), v => Assert.Equal(1f, v));
        }

        [Fact]
        public void From_CopiesData()
        {
            float[] src = [1, 2, 3, 4, 5, 6];
            using var t = Tensor<float>.From(src, 2, 3);
            Assert.Equal(src, t.Data.ToArray());
        }

        [Fact]
        public void Reshape_ZeroCopy_SharesBuffer()
        {
            using var t = Tensor<float>.From([1f, 2f, 3f, 4f], 2, 2);
            using var v = t.Reshape(4);
            v[0] = 99f;
            Assert.Equal(99f, t[0, 0]);  // mutation visible through original
        }

        /* [Fact]
         public void RowView_ZeroCopy()
         {
             using var t = Tensor<float>.From([1f, 2f, 3f, 4f], 2, 2);
             using var row = t.RowView(1);
             row[0] = 77f;
             Assert.Equal(77f, t[1, 0]);
         }

         [Fact]
         public void Dispose_ViewIndependentOfOriginal()
         {
             var t = Tensor<float>.Ones(4, 4);
             var row = t.RowView(0);
             row.Dispose();              // view disposed — original still alive
             var ex = Record.Exception(() => { float _ = t[0, 0]; });
             Assert.Null(ex);
             t.Dispose();
         }*/

        [Fact]
        public void NegativeIndex_WorksForLastDim()
        {
            using var t = Tensor<float>.From([1f, 2f, 3f], 3);
            Assert.Equal(3f, t[t.Shape[-1] - 1]);
        }

        [Fact]
        public unsafe void Slice_DataPtr_PointsAtViewStart()
        {
            using var t = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
            using var row = t.Slice(1);

            Assert.Equal(4f, row.DataPtr[0]);
            Assert.Equal(6f, row.DataPtr[2]);
        }

        [Fact]
        public unsafe void Slice_OfSlice_DataPtr_AccumulatesOffsets()
        {
            using var t = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], 2, 2, 2);
            using var batch = t.Slice(1);
            using var row = batch.Slice(1);

            Assert.Equal(7f, row.DataPtr[0]);
        }
    }
}
