using SharpMind.GPU;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class GateKernelTests
{
    const int M = 5, F = 6;
    const float Sqrt2PiInv = 0.7978845608028654f, GeluCoeff = 0.044715f;

    static float Gate(float g, bool gelu) { float sig = 1f / (1f + MathF.Exp(-g)); return gelu ? 0.5f * g * (1f + MathF.Tanh(Sqrt2PiInv * (g + GeluCoeff * g * g * g))) : g * sig; }
    static float GateD(float g, bool gelu)
    {
        float sig = 1f / (1f + MathF.Exp(-g));
        if (!gelu) return sig * (1f + g * (1f - sig));
        float z = Sqrt2PiInv * (g + GeluCoeff * g * g * g); float t = MathF.Tanh(z); float sech2 = 1f - t * t;
        return 0.5f * (1f + t) + 0.5f * g * sech2 * Sqrt2PiInv * (1f + 3f * GeluCoeff * g * g);
    }

    [Theory] [InlineData(false)] [InlineData(true)]
    public void GateFwdBwd_MatchBackpropEngineFormulas(bool gelu)
    {
        var dev = GpuTestDevice.Device;
        var fused = GpuTestDevice.Random(M * 2 * F, 11, 2f); var dAct = GpuTestDevice.Random(M * F, 12, 1f);
        var wantAct = new float[M * F]; var wantDFused = new float[M * 2 * F];
        for (int r = 0; r < M; r++) for (int d = 0; d < F; d++)
        {
            float g = fused[r * 2 * F + d], u = fused[r * 2 * F + F + d], da = dAct[r * F + d];
            wantAct[r * F + d] = Gate(g, gelu) * u;
            wantDFused[r * 2 * F + d] = da * GateD(g, gelu) * u;      // dGate
            wantDFused[r * 2 * F + F + d] = da * Gate(g, gelu);       // dUp
        }
        using var arena = new DeviceArena(dev, 1 << 12);
        var tf = arena.Rent(M, 2 * F); tf.Upload(fused);
        var ta = arena.Rent(M, F); var tda = arena.Rent(M, F); tda.Upload(dAct); var tdf = arena.Rent(M, 2 * F);
        dev.Kernels.GateFwd(ta, tf, gelu);
        dev.Kernels.GateBwd(tdf, tda, tf, gelu);
        dev.Synchronize();
        GpuTestDevice.AssertClose(wantAct, ta.ToArray(), 1e-5, "act");
        GpuTestDevice.AssertClose(wantDFused, tdf.ToArray(), 1e-5, "dFused");
    }

    [Fact]
    public void GateFwd_ThrowsOnMismatchedFusedCols()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var act = arena.Rent(M, F); var badFused = arena.Rent(M, 2 * F - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.GateFwd(act, badFused, gelu: false));
    }

    [Fact]
    public void GateFwd_ThrowsOnMismatchedRows()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var act = arena.Rent(M, F); var badFused = arena.Rent(M - 1, 2 * F);
        Assert.Throws<ArgumentException>(() => dev.Kernels.GateFwd(act, badFused, gelu: false));
    }

    [Fact]
    public void GateBwd_ThrowsOnMismatchedFusedShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var fused = arena.Rent(M, 2 * F); var dAct = arena.Rent(M, F); var badDFused = arena.Rent(M, 2 * F - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.GateBwd(badDFused, dAct, fused, gelu: false));
    }

    [Fact]
    public void GateBwd_ThrowsOnMismatchedDActCols()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var fused = arena.Rent(M, 2 * F); var dFused = arena.Rent(M, 2 * F); var badDAct = arena.Rent(M, F - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.GateBwd(dFused, badDAct, fused, gelu: false));
    }
}
