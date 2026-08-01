using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Tests.Core
{
    [Collection("Non-Parallel")]
    public sealed class GradientMappingTests
    {
        private readonly GradientMapping _mapping;

        public GradientMappingTests()
        {
            _mapping = GradientMappingFactory.Create(
                SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar });
        }

        [Fact]
        public void Assembles_AllGradientSlots()
        {
            // Every abstract slot must have been wired to a kernel by JigSaw.
            Assert.NotNull(_mapping);
        }

        [Fact]
        public void Linear_ReturnsDInput_AndAccumulatesGradients()
        {
            int B = 2, Out = 3, In = 4;
            using var dOutput = Tensor<float>.From(
                [1f, 2f, 3f, 4f, 5f, 6f], B, Out);
            using var input = Tensor<float>.From(
                [1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f], B, In);
            using var wData = Tensor<float>.Ones(Out, In);
            using var bData = Tensor<float>.Ones(Out);
            using var weight = new Parameter("w", wData);
            using var bias = new Parameter("b", bData);

            using var dInput = _mapping.Linear(dOutput, input, weight, bias);

            Assert.Equal(B, dInput.Shape.Rows);
            Assert.Equal(In, dInput.Shape.Cols);
            // weight.Grad[o,i] += sum_b dOutput[b,o] * input[b,i]
            Assert.Equal(9f, weight.Grad[0, 0], precision: 4);
            Assert.Equal(15f, weight.Grad[2, 3], precision: 4);
            // bias.Grad[o] += sum_b dOutput[b,o]
            Assert.Equal(5f, bias.Grad[0], precision: 4);
            Assert.Equal(9f, bias.Grad[2], precision: 4);
        }

        [Fact]
        public void RMSNorm_ReturnsDInput_AndAccumulatesWeightGrad()
        {
            int T = 2, D = 3;
            using var dOutput = Tensor<float>.Ones(T, D);
            using var xNorm = Tensor<float>.Ones(T, D);
            float[] rmsInv = [2f, 2f];
            using var wData = Tensor<float>.Ones(D);
            using var weight = new Parameter("w", wData);

            using var dInput = _mapping.RMSNorm(dOutput, xNorm, rmsInv, weight);

            Assert.Equal(T, dInput.Shape.Rows);
            Assert.Equal(D, dInput.Shape.Cols);
            Assert.Equal(2f, weight.Grad[0], precision: 4);
        }

        [Fact]
        public void LayerNorm_ReturnsDInput_AndAccumulatesGradients()
        {
            int T = 2, D = 3;
            using var dOutput = Tensor<float>.Ones(T, D);
            using var input = Tensor<float>.From(
                [1f, 1f, 1f, 2f, 2f, 2f], T, D);
            using var wData = Tensor<float>.Ones(D);
            using var bData = Tensor<float>.Ones(D);
            using var weight = new Parameter("w", wData);
            using var bias = new Parameter("b", bData);

            using var dInput = _mapping.LayerNorm(dOutput, input, weight, bias);

            Assert.Equal(T, dInput.Shape.Rows);
            Assert.Equal(D, dInput.Shape.Cols);
            Assert.Equal(2f, bias.Grad[0], precision: 4);
        }

        [Fact]
        public void Attention_ReturnsDQDKAndDV()
        {
            int S = 2, D = 3;
            using var dOut = Tensor<float>.Ones(S, D);
            using var q = Tensor<float>.Ones(S, D);
            using var k = Tensor<float>.Ones(S, D);
            using var v = Tensor<float>.Ones(S, D);
            using var probs = Tensor<float>.From(
                [0.5f, 0.5f, 0.25f, 0.75f], S, S);

            var grads = _mapping.Attention(dOut, q, k, v, probs, scale: 0.5f);

            using (grads.DQ) using (grads.DK) using (grads.DV)
            {
                Assert.Equal(S, grads.DQ.Shape.Rows);
                Assert.Equal(D, grads.DQ.Shape.Cols);
                Assert.Equal(S, grads.DK.Shape.Rows);
                Assert.Equal(S, grads.DV.Shape.Rows);
            }
        }

        [Fact]
        public void Embedding_AccumulatesIntoSelectedRows()
        {
            int T = 2, D = 3, V = 5;
            using var dOutput = Tensor<float>.Ones(T, D);
            using var tokenIds = Tensor<int>.From([1, 3], T);
            using var wData = Tensor<float>.Zeros(V, D);
            using var weight = new Parameter("embedding.weight", wData);

            _mapping.Embedding(dOutput, tokenIds, weight);

            Assert.Equal(1f, weight.Grad[1, 0], precision: 4);
            Assert.Equal(0f, weight.Grad[0, 0], precision: 4);
            Assert.Equal(1f, weight.Grad[3, 2], precision: 4);
        }

        [Fact]
        public void ActivationSiLU_ReturnsDInput()
        {
            using var dOutput = Tensor<float>.From([1f, 2f, 3f], 3);
            using var preAct = Tensor<float>.From([1f, 1f, 1f], 3);

            using var dInput = _mapping.ActivationSiLU(dOutput, preAct);

            Assert.Equal(3, dInput.Shape.Cols);
            Assert.False(float.IsNaN(dInput[0]));
        }

        [Fact]
        public void ActivationGELU_ReturnsDInput()
        {
            using var dOutput = Tensor<float>.From([1f, 2f, 3f], 3);
            using var preAct = Tensor<float>.From([1f, 1f, 1f], 3);

            using var dInput = _mapping.ActivationGELU(dOutput, preAct);

            Assert.Equal(3, dInput.Shape.Cols);
            Assert.False(float.IsNaN(dInput[0]));
        }
    }
}
