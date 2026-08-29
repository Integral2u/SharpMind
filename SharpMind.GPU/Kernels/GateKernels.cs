using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>SwiGLU / tanh-GELU gating over the fused [M, 2F] layout — no gate/up split copies.
/// Formulas are BackpropEngine.GateValue / GateDerivative verbatim.</summary>
internal static class GateKernels
{
    const float Sqrt2PiInv = 0.7978845608028654f, GeluCoeff = 0.044715f;

    static float Sigmoid(float g) => 1f / (1f + XMath.Exp(-g));
    static float GateValue(float g, int gelu) => gelu != 0 ? 0.5f * g * (1f + XMath.Tanh(Sqrt2PiInv * (g + GeluCoeff * g * g * g))) : g * Sigmoid(g);
    static float GateDeriv(float g, int gelu)
    {
        if (gelu == 0) { float sig = Sigmoid(g); return sig * (1f + g * (1f - sig)); }
        float z = Sqrt2PiInv * (g + GeluCoeff * g * g * g); float t = XMath.Tanh(z); float sech2 = 1f - t * t;
        return 0.5f * (1f + t) + 0.5f * g * sech2 * Sqrt2PiInv * (1f + 3f * GeluCoeff * g * g);
    }

    public static void Fwd(Index1D i, ArrayView<float> act, ArrayView<float> fused, int f, int gelu)
    {
        int r = i / f, d = i % f; long b = (long)r * 2 * f;
        act[i] = GateValue(fused[b + d], gelu) * fused[b + f + d];
    }

    public static void Bwd(Index1D i, ArrayView<float> dFused, ArrayView<float> dAct, ArrayView<float> fused, int f, int gelu)
    {
        int r = i / f, d = i % f; long b = (long)r * 2 * f;
        float g = fused[b + d], u = fused[b + f + d], da = dAct[i];
        dFused[b + d] = da * GateDeriv(g, gelu) * u;
        dFused[b + f + d] = da * GateValue(g, gelu);
    }
}
