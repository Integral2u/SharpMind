using SharpMind.Core.Activations;
using SharpMind.Core.Tensors;

namespace SharpMind.Tests.Core
{
    [Collection("Non-Parallel")]
    public sealed class ActivationTests
    {
        private readonly ActivationOps _gpt;
        private readonly ActivationOps _llama;

        public ActivationTests()
        {
            var scalar = HardwareTier.Scalar;
            _gpt = ActivationFactory.Create(SharpMindConfig.Gpt with { Hardware = scalar });
            _llama = ActivationFactory.Create(SharpMindConfig.Llama with { Hardware = scalar });
        }

        // ── Softmax ───────────────────────────────────────────────────────────

        [Fact]
        public void Softmax_SumsToOne()
        {
            using var x = Tensor<float>.From([1f, 2f, 3f, 4f], 4);
            using var r = _gpt.Softmax(x);
            float sum = r.Data.ToArray().Sum();
            Assert.Equal(1f, sum, precision: 5);
        }

        [Fact]
        public void Softmax_MonotonicallyOrdered()
        {
            using var x = Tensor<float>.From([1f, 2f, 3f], 3);
            using var r = _gpt.Softmax(x);
            var data = r.Data.ToArray();
            Assert.True(data[0] < data[1]);
            Assert.True(data[1] < data[2]);
        }

        [Fact]
        public void Softmax_NumericallyStable_LargeInputs()
        {
            using var x = Tensor<float>.From([1000f, 1001f, 1002f], 3);
            using var r = _gpt.Softmax(x);
            float sum = r.Data.ToArray().Sum();
            Assert.Equal(1f, sum, precision: 4);
            Assert.False(float.IsNaN(r[0]));
        }

        [Fact]
        public void Softmax_2D_AppliedRowWise()
        {
            using var x = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f], 2, 3);
            using var r = _gpt.Softmax(x);
            float row0Sum = r.RowSpan(0).ToArray().Sum();
            float row1Sum = r.RowSpan(1).ToArray().Sum();
            Assert.Equal(1f, row0Sum, precision: 5);
            Assert.Equal(1f, row1Sum, precision: 5);
        }

        // ── RMSNorm ───────────────────────────────────────────────────────────

        [Fact]
        public void RMSNorm_UnitWeights_NormalisesInput()
        {
            using var x = Tensor<float>.From([3f, 4f], 2);  // RMS = sqrt((9+16)/2) = sqrt(12.5)
            using var w = Tensor<float>.Ones(2);
            using var r = _gpt.RMSNorm(x, w, eps: 0f);
            float rms = MathF.Sqrt(r.Data.ToArray().Select(v => v * v).Average());
            Assert.Equal(1f, rms, precision: 4);
        }

        [Fact]
        public void RMSNorm_WeightScales()
        {
            using var x = Tensor<float>.From([1f, 1f, 1f, 1f], 4);
            using var w = Tensor<float>.From([2f, 2f, 2f, 2f], 4);
            using var r = _gpt.RMSNorm(x, w, eps: 0f);
            Assert.All(r.Data.ToArray(), v => Assert.Equal(2f, v, precision: 4));
        }

        // ── GELU ──────────────────────────────────────────────────────────────

        [Fact]
        public void GELU_Zero_ReturnsZero()
        {
            using var x = Tensor<float>.Zeros(4);
            using var r = _gpt.Activate(x);
            Assert.All(r.Data.ToArray(), v => Assert.Equal(0f, v, precision: 5));
        }

        [Fact]
        public void GELU_PositiveLarge_ApproachesIdentity()
        {
            using var x = Tensor<float>.From([10f], 1);
            using var r = _gpt.Activate(x);
            Assert.Equal(10f, r[0], precision: 2);
        }

        [Fact]
        public void GELU_NegativeLarge_ApproachesZero()
        {
            using var x = Tensor<float>.From([-10f], 1);
            using var r = _gpt.Activate(x);
            Assert.Equal(0f, r[0], precision: 2);
        }

        // ── SiLU ──────────────────────────────────────────────────────────────

        [Fact]
        public void SiLU_Zero_ReturnsZero()
        {
            using var x = Tensor<float>.Zeros(4);
            using var r = _llama.Activate(x);
            Assert.All(r.Data.ToArray(), v => Assert.Equal(0f, v, precision: 5));
        }

        [Fact]
        public void SiLU_AlwaysGtNegativeHalf()
        {
            using var x = Tensor<float>.From([-100f, -10f, 0f, 10f, 100f], 5);
            using var r = _llama.Activate(x);
            Assert.All(r.Data.ToArray(), v => Assert.True(v >= -0.4f));
        }

        // ── SwiGLU ────────────────────────────────────────────────────────────

        [Fact]
        public void SwiGLU_ZeroGate_ZeroOutput()
        {
            using var gate = Tensor<float>.Zeros(4);
            using var up = Tensor<float>.Ones(4);
            using var r = _llama.GatedActivate(gate, up);
            Assert.All(r.Data.ToArray(), v => Assert.Equal(0f, v, precision: 5));
        }

        [Fact]
        public void SwiGLU_PositiveGate_ScalesUp()
        {
            using var gate = Tensor<float>.From([100f], 1);
            using var up = Tensor<float>.From([3f], 1);
            using var r = _llama.GatedActivate(gate, up);
            Assert.Equal(300f, r[0], precision: 2);  // silu(100) ≈ 100 → 100 * 3 = 300
        }

        // ── AVX2 parity ───────────────────────────────────────────────────────

        [Fact]
        public void ActivationParity_ScalarMatchesAvx2_RMSNorm()
        {
            if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;

            var avx2Ops = ActivationFactory.Create(SharpMindConfig.Gpt with { Hardware = HardwareTier.AVX2 });
            using var x = Tensor<float>.From([3f, 1f, 4f, 1f, 5f, 9f, 2f, 6f], 8);
            using var w = Tensor<float>.Ones(8);
            using var rS = _gpt.RMSNorm(x, w);
            using var rA = avx2Ops.RMSNorm(x, w);

            for (int i = 0; i < rS.ElementCount; i++)
                Assert.Equal(rS[i], rA[i], precision: 5);
        }
    }
}
