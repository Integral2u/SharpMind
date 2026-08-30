using SharpMind.GPU;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class NormKernelTests
{
    const int Rows = 7, D = 24; const float Eps = 1e-5f;

    [Fact]
    public void RmsNormFwd_MatchesRowFormula()
    {
        var dev = GpuTestDevice.Device;
        var x = GpuTestDevice.Random(Rows * D, 1, 1f); var w = GpuTestDevice.Random(D, 2, 1f);
        var wantY = new float[Rows * D]; var wantR = new float[Rows];
        for (int r = 0; r < Rows; r++)
        {
            double ss = 0; for (int d = 0; d < D; d++) ss += x[r * D + d] * x[r * D + d];
            float rInv = 1f / MathF.Sqrt((float)(ss / D) + Eps); wantR[r] = rInv;
            for (int d = 0; d < D; d++) wantY[r * D + d] = x[r * D + d] * rInv * w[d];
        }
        using var arena = new DeviceArena(dev, 1 << 12);
        var dx = arena.Rent(Rows, D); dx.Upload(x);
        var dw = arena.Rent(1, D); dw.Upload(w);
        var dy = arena.Rent(Rows, D); var dr = arena.Rent(Rows, 1);
        dev.Kernels.RmsNormFwd(dy, dr, dx, dw, Eps); dev.Synchronize();
        GpuTestDevice.AssertClose(wantY, dy.ToArray(), 1e-5, "y");
        GpuTestDevice.AssertClose(wantR, dr.ToArray(), 1e-5, "rInv");
    }

    [Fact]
    public void RmsNormBwd_MatchesGradientMappingFormula()
    {
        // dx = rInv·(g − xNorm·mean(g·xNorm)), g = dy·w, xNorm = x·rInv   (GradientMapping.RMSNorm, w frozen)
        var dev = GpuTestDevice.Device;
        var x = GpuTestDevice.Random(Rows * D, 3, 1f); var w = GpuTestDevice.Random(D, 4, 1f); var dy = GpuTestDevice.Random(Rows * D, 5, 1f);
        var rInv = new float[Rows]; var want = new float[Rows * D];
        for (int r = 0; r < Rows; r++)
        {
            double ss = 0; for (int d = 0; d < D; d++) ss += x[r * D + d] * x[r * D + d];
            rInv[r] = 1f / MathF.Sqrt((float)(ss / D) + Eps);
            double mean = 0; for (int d = 0; d < D; d++) mean += dy[r * D + d] * w[d] * x[r * D + d] * rInv[r];
            mean /= D;
            for (int d = 0; d < D; d++) want[r * D + d] = rInv[r] * (dy[r * D + d] * w[d] - x[r * D + d] * rInv[r] * (float)mean);
        }
        using var arena = new DeviceArena(dev, 1 << 12);
        var tx = arena.Rent(Rows, D); tx.Upload(x); var tw = arena.Rent(1, D); tw.Upload(w);
        var tdy = arena.Rent(Rows, D); tdy.Upload(dy); var tr = arena.Rent(Rows, 1); tr.Upload(rInv);
        var tdx = arena.Rent(Rows, D);
        dev.Kernels.RmsNormBwd(tdx, tdy, tx, tr, tw); dev.Synchronize();
        GpuTestDevice.AssertClose(want, tdx.ToArray(), 1e-5, "dx");
    }

    [Fact]
    public void RmsNormFwd_ThrowsOnMismatchedWeightLength()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var x = arena.Rent(Rows, D); var y = arena.Rent(Rows, D); var r = arena.Rent(Rows, 1);
        var badW = arena.Rent(1, D - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.RmsNormFwd(y, r, x, badW, Eps));
    }

    [Fact]
    public void RmsNormFwd_ThrowsOnMismatchedRInvShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var x = arena.Rent(Rows, D); var y = arena.Rent(Rows, D); var w = arena.Rent(1, D);
        var badR = arena.Rent(Rows - 1, 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.RmsNormFwd(y, badR, x, w, Eps));
    }

    [Fact]
    public void RmsNormBwd_ThrowsOnMismatchedWeightLength()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var x = arena.Rent(Rows, D); var dy = arena.Rent(Rows, D); var dx = arena.Rent(Rows, D); var r = arena.Rent(Rows, 1);
        var badW = arena.Rent(1, D - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.RmsNormBwd(dx, dy, x, r, badW));
    }

    [Fact]
    public void RmsNormBwd_ThrowsOnMismatchedRInvShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var x = arena.Rent(Rows, D); var dy = arena.Rent(Rows, D); var dx = arena.Rent(Rows, D); var w = arena.Rent(1, D);
        var badR = arena.Rent(Rows, 2);
        Assert.Throws<ArgumentException>(() => dev.Kernels.RmsNormBwd(dx, dy, x, badR, w));
    }
}
