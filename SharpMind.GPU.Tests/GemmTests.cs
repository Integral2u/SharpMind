using SharpMind.GPU;
using Xunit;

namespace SharpMind.GPU.Tests;

[Collection("GPU")]
public sealed class GemmTests
{
    // C = op(A)·op(B) for all four stride layouts, checked against a double reference.
    [Theory]
    [InlineData(false, false)] [InlineData(false, true)] [InlineData(true, false)] [InlineData(true, true)]
    public void Gemm_AllLayouts_MatchDoubleReference(bool aTransposed, bool bTransposed)
    {
        var dev = GpuTestDevice.Device;
        const int m = 37, n = 53, k = 29;   // deliberately not multiples of 16
        var a = GpuTestDevice.Random(m * k, 1); var b = GpuTestDevice.Random(k * n, 2);
        // Logical A[i,k] stored row-major [m,k] (saI=k, saK=1) or as its transpose [k,m] (saI=1, saK=m).
        int saI = aTransposed ? 1 : k, saK = aTransposed ? m : 1;
        int sbK = bTransposed ? 1 : n, sbJ = bTransposed ? k : 1;
        var want = new float[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { double s = 0; for (int t = 0; t < k; t++) s += (double)a[i * saI + t * saK] * b[t * sbK + j * sbJ]; want[i * n + j] = (float)s; }

        using var arena = new DeviceArena(dev, 1 << 16);
        var da = arena.Rent(1, m * k); da.Upload(a);
        var db = arena.Rent(1, k * n); db.Upload(b);
        var dc = arena.Rent(m, n);
        dev.Gemm(dc, da, db, m, n, k, saI, saK, sbK, sbJ);
        dev.Synchronize();
        GpuTestDevice.AssertClose(want, dc.ToArray(), 1e-5, $"gemm aT={aTransposed} bT={bTransposed}");
    }

    [Fact]
    public void Gemm_BetaOne_Accumulates()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var a = arena.Rent(1, 4); a.Upload([1f, 2f, 3f, 4f]);       // [2,2]
        var b = arena.Rent(1, 4); b.Upload([1f, 0f, 0f, 1f]);       // identity
        var c = arena.Rent(2, 2); c.Upload([10f, 10f, 10f, 10f]);
        dev.Gemm(c, a, b, 2, 2, 2, 2, 1, 2, 1, beta: 1f);
        dev.Synchronize();
        Assert.Equal([11f, 12f, 13f, 14f], c.ToArray());
    }

    // cuBLAS needs exactly one stride per operand to be 1; the tiled kernel does not care, so only
    // an explicit check keeps a caller mistake from passing here and computing garbage on CUDA.
    [Theory]
    [InlineData(4, 2, 4, 1)]    // A: neither stride is 1
    [InlineData(1, 1, 4, 1)]    // A: both are 1
    [InlineData(4, 1, 4, 2)]    // B: neither stride is 1
    [InlineData(4, 1, 1, 1)]    // B: both are 1
    [InlineData(-4, 1, 4, 1)]   // A: negative stride
    public void Gemm_AmbiguousStrides_Throws(int saI, int saK, int sbK, int sbJ)
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var a = arena.Rent(1, 64); var b = arena.Rent(1, 64); var c = arena.Rent(4, 4);
        Assert.Throws<ArgumentException>(() => dev.Gemm(c, a, b, 4, 4, 4, saI, saK, sbK, sbJ));
    }

    [Fact]
    public void Gemm_UndersizedOperand_Throws()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var small = arena.Rent(1, 8);       // a [4,4] row-major operand is addressed up to element 16
        var full = arena.Rent(1, 16);
        var c = arena.Rent(4, 4);
        Assert.Throws<ArgumentException>(() => dev.Gemm(c, small, full, 4, 4, 4, 4, 1, 4, 1));
        Assert.Throws<ArgumentException>(() => dev.Gemm(c, full, small, 4, 4, 4, 4, 1, 4, 1));
        dev.Gemm(c, full, full, 4, 4, 4, 4, 1, 4, 1);   // exactly-sized operands are accepted
        dev.Synchronize();
    }

    // Attention writes one head's [S, headDim] result into a [B·S, heads·headDim] tensor, so C is
    // a column block of something wider and ldc carries its row stride. Nothing outside the block
    // may be touched — on the cuBLAS path a wrong ldc would silently smear into the next head.
    [Fact]
    public void Gemm_StridedC_WritesOnlyItsColumnBlock()
    {
        var dev = GpuTestDevice.Device;
        const int m = 12, n = 5, k = 7, wide = 13, colOffset = 6, sentinel = -7;
        var a = GpuTestDevice.Random(m * k, 51); var b = GpuTestDevice.Random(k * n, 52);
        var want = new float[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { double s = 0; for (int t = 0; t < k; t++) s += (double)a[i * k + t] * b[t * n + j]; want[i * n + j] = (float)s; }

        using var arena = new DeviceArena(dev, 1 << 14);
        var da = arena.Rent(1, m * k); da.Upload(a);
        var db = arena.Rent(1, k * n); db.Upload(b);
        var dc = arena.Rent(m, wide);
        var filled = new float[m * wide]; Array.Fill(filled, (float)sentinel); dc.Upload(filled);

        dev.Gemm(dc.Window(colOffset, (long)(m - 1) * wide + n), da, db, m, n, k, k, 1, n, 1, ldc: wide);
        dev.Synchronize();

        var got = dc.ToArray();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < wide; j++)
            {
                bool inBlock = j >= colOffset && j < colOffset + n;
                float expect = inBlock ? want[i * n + (j - colOffset)] : sentinel;
                Assert.True(Math.Abs(got[i * wide + j] - expect) < 1e-5f, $"C[{i},{j}] = {got[i * wide + j]}, want {expect}");
            }
    }

    [Fact]
    public void Gemm_LdcNarrowerThanN_Throws()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var a = arena.Rent(1, 16); var b = arena.Rent(1, 16); var c = arena.Rent(4, 4);
        Assert.Contains("narrower", Assert.Throws<ArgumentException>(() => dev.Gemm(c, a, b, 4, 4, 4, 4, 1, 4, 1, ldc: 3)).Message);
    }

    /// <summary>An ldc that reaches past the destination must be caught here, not by the driver:
    /// on the cuBLAS path an out-of-bounds C write is memory corruption, not an exception.</summary>
    [Fact]
    public void Gemm_StridedC_UndersizedDestination_Throws()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 12);
        var a = arena.Rent(1, 16); var b = arena.Rent(1, 16);
        var c = arena.Rent(1, 27);          // [4,4] at ldc 8 is addressed up to element 28
        Assert.Throws<ArgumentException>(() => dev.Gemm(c, a, b, 4, 4, 4, 4, 1, 4, 1, ldc: 8));
        dev.Gemm(arena.Rent(1, 28), a, b, 4, 4, 4, 4, 1, 4, 1, ldc: 8);   // exactly-sized is accepted
        dev.Synchronize();
    }
}
