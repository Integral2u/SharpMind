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
        private readonly ActivationOps _cpuOps;
        private readonly ActivationOps _gpuOps;

        public ActivationComparisonTests()
        {
            _cpuOps = ActivationFactory.Create(SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });

            var gpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
            var mapping = gpuConfig.ToJigSawMapping();
            mapping["pointwise"] = "relugpu";
            mapping["gate"] = "geglugpu";
            mapping["softmax"] = "softmaxgpu";
            mapping["rmsnorm"] = "rmsnormgpu";
            _gpuOps = ActivationFactory.Create(gpuConfig);
        }

        [Fact]
        public void ReLU_CPUAndGPU_ProduceSameResults()
        {
            var data = new float[] { 1f, -2f, 3f, -4f, 5f, -6f, 0f, 0.5f };
            using var src = Tensor<float>.From(data, 8);

            var cpuDst = new float[data.Length];
            var gpuDst = new float[data.Length];

            _cpuOps.ApplyPointwise(src.Data, cpuDst);
            _gpuOps.ApplyPointwise(src.Data, gpuDst);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 4);
        }

        [Fact]
        public void GELU_CPUAndGPU_ProduceSameResults()
        {
            var data = new float[] { 0.1f, -0.1f, 0.5f, -0.5f, 1f, -1f };
            using var src = Tensor<float>.From(data, 6);

            var cpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Activation = ActivationKind.GELU };
            var gpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Activation = ActivationKind.GELU };
            var gpuMapping = gpuConfig.ToJigSawMapping();
            gpuMapping["pointwise"] = "gelugpu";

            var cpuOps = ActivationFactory.Create(cpuConfig);
            var gpuOps = ActivationFactory.Create(gpuConfig);

            var cpuDst = new float[data.Length];
            var gpuDst = new float[data.Length];

            cpuOps.ApplyPointwise(src.Data, cpuDst);
            gpuOps.ApplyPointwise(src.Data, gpuDst);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 3);
        }

        [Fact]
        public void SiLU_CPUAndGPU_ProduceSameResults()
        {
            var data = new float[] { 0.1f, -0.1f, 0.5f, -0.5f, 1f, -1f };
            using var src = Tensor<float>.From(data, 6);

            var cpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Activation = ActivationKind.SiLU };
            var gpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Activation = ActivationKind.SiLU };
            var gpuMapping = gpuConfig.ToJigSawMapping();
            gpuMapping["pointwise"] = "silugpu";

            var cpuOps = ActivationFactory.Create(cpuConfig);
            var gpuOps = ActivationFactory.Create(gpuConfig);

            var cpuDst = new float[data.Length];
            var gpuDst = new float[data.Length];

            cpuOps.ApplyPointwise(src.Data, cpuDst);
            gpuOps.ApplyPointwise(src.Data, gpuDst);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 3);
        }

        [Fact]
        public void GeGLU_CPUAndGPU_ProduceSameResults()
        {
            var gate = new float[] { 0.1f, -0.1f, 0.5f, -0.5f };
            var up = new float[] { 1f, 2f, 3f, 4f };
            using var gateTensor = Tensor<float>.From(gate, 4);
            using var upTensor = Tensor<float>.From(up, 4);

            var cpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Gate = GateKind.GeGLU };
            var gpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar, Gate = GateKind.GeGLU };
            var gpuMapping = gpuConfig.ToJigSawMapping();
            GPUSharpMindConfig.AddGPUMappings(gpuMapping);

            var cpuOps = ActivationFactory.Create(cpuConfig);
            var gpuOps = ActivationFactory.Create(gpuConfig);

            var cpuDst = new float[gate.Length];
            var gpuDst = new float[gate.Length];

            cpuOps.ApplyGate(gateTensor.Data, upTensor.Data, cpuDst);
            gpuOps.ApplyGate(gateTensor.Data, upTensor.Data, gpuDst);

            for (int i = 0; i < gate.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 3);
        }

        [Fact]
        public void SwiGLU_CPUAndGPU_ProduceSameResults()
        {
            var gate = new float[] { 0.1f, -0.1f, 0.5f, -0.5f };
            var up = new float[] { 1f, 2f, 3f, 4f };
            using var gateTensor = Tensor<float>.From(gate, 4);
            using var upTensor = Tensor<float>.From(up, 4);

            var cpuConfig = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
            var gpuConfig = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };

            var cpuOps = ActivationFactory.Create(cpuConfig);
            var gpuOps = ActivationFactory.Create(gpuConfig);

            var cpuDst = new float[gate.Length];
            var gpuDst = new float[gate.Length];

            cpuOps.ApplyGate(gateTensor.Data, upTensor.Data, cpuDst);
            gpuOps.ApplyGate(gateTensor.Data, upTensor.Data, gpuDst);

            for (int i = 0; i < gate.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 3);
        }

        [Fact]
        public void Softmax_CPUAndGPU_ProduceSameResults()
        {
            var data = new float[] { 1f, 2f, 3f, 4f, 5f };
            using var src = Tensor<float>.From(data, 5);

            var cpuConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

            var cpuOps = ActivationFactory.Create(cpuConfig);

            var cpuDst = new float[data.Length];
            var gpuDst = new float[data.Length];

            cpuOps.ApplySoftmaxRow(src.Data, cpuDst);
            _gpuOps.ApplySoftmaxRow(src.Data, gpuDst);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 4);
        }

        [Fact]
        public void RMSNorm_CPUAndGPU_ProduceSameResults()
        {
            var data = new float[] { 1f, 2f, 3f, 4f };
            var weight = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
            using var src = Tensor<float>.From(data, 4);
            using var w = Tensor<float>.From(weight, 4);

            var rmsInv = 0.5f;

            var cpuDst = new float[data.Length];
            var gpuDst = new float[data.Length];

            _cpuOps.ApplyRMSNormRow(src.Data, w.Data, cpuDst, rmsInv);
            _gpuOps.ApplyRMSNormRow(src.Data, w.Data, gpuDst, rmsInv);

            for (int i = 0; i < data.Length; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 4);
        }

        [Fact]
        public void MatMul_CPUAndGPU_ProduceSameResults()
        {
            var a = new float[] { 1f, 0f, 0f, 1f };
            var b = new float[] { 1f, 0f, 0f, 1f };
            using var aTensor = Tensor<float>.From(a, 2, 2);
            using var bTensor = Tensor<float>.From(b, 2, 2);

            var cpuOps = TensorOpsFactory.Create(SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });

            var cpuDst = new float[4];
            var cTensor = new Tensor<float>(2, 2);
            cpuOps.MatMulInto(aTensor, bTensor, cTensor);
            cTensor.Data.CopyTo(cpuDst);

            var gpuDst = new float[4];
            var gpuCTensor = new Tensor<float>(2, 2);
            unsafe
            {
                GPUActivationKernels.MatMulInnerGPU(
                    aTensor.DataPtr,
                    bTensor.DataPtr,
                    gpuCTensor.DataPtr,
                    2, 2, 2);
            }
            gpuCTensor.Data.CopyTo(gpuDst);

            for (int i = 0; i < 4; i++)
                Assert.Equal(cpuDst[i], gpuDst[i], precision: 4);
        }
    }
}