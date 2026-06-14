using SharpMind.Core.Activations;
using SharpMind.Core.Tensors;
using SharpMind.Core.Ops;
using SharpMind;
using SharpMind.GPU;

namespace SharpMind.Tests.GPU
{
    [Collection("Non-Parallel")]
    public sealed class ActivationComparisonTests
    {
        [Fact(Skip = "No CUDA GPU available on this system")]
        public void ReLU_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void GELU_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void SiLU_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void GeGLU_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void SwiGLU_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void Softmax_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void RMSNorm_CPUAndGPU_ProduceSameResults() { }

        [Fact(Skip = "No CUDA GPU available on this system")]
        public void MatMul_CPUAndGPU_ProduceSameResults() { }
    }
}
