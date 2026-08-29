using SharpMind.GPU;
using SharpMind.GPU.Kernels;
using Xunit;

namespace SharpMind.GPU.Tests;

/// <summary>
/// The flash kernels must compute what <see cref="AttentionKernels"/> computes, so they are
/// checked against the same CPU reference <see cref="AttentionKernelTests"/> uses — not against
/// the materialised kernels — which keeps the two GPU paths independently verified.
///
/// Tolerances are the reference test's. The online softmax accumulates the output in a different
/// order (rescaling on every new row maximum) and the backward recomputes p from m and l rather
/// than reading it back, so the two paths agree to float rounding, not bit for bit.
/// </summary>
[Collection("GPU")]
public sealed class FlashAttentionKernelTests
{
    const int B = AttentionKernelTests.B, S = AttentionKernelTests.S, D = AttentionKernelTests.D;

    public static TheoryData<int, int> HeadShapes => new() { { 4, 4 }, { 4, 2 }, { 4, 1 } };

    [Theory]
    [MemberData(nameof(HeadShapes))]
    public void FlashFwd_Bwd_MatchReference(int H, int KV)
    {
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 21, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 22, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 23, 1f); var dO = GpuTestDevice.Random(B * S * H * D, 24, 1f);
        var (wantO, wantP) = AttentionKernelTests.RefForward(q, k, v, H, KV);
        var (wantDq, wantDk, wantDv) = AttentionKernelTests.RefBackward(q, k, v, wantP, dO, H, KV);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D);
        var stats = arena.Rent(B * H * S, FlashAttentionKernels.StatCols);
        dev.Kernels.AttnFwdFlash(to, stats, tq, tk, tv, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantO, to.ToArray(), 1e-5, "out");

        var tdo = arena.Rent(B * S, H * D); tdo.Upload(dO);
        var tdq = arena.Rent(B * S, H * D); var tdk = arena.Rent(B * S, KV * D); var tdv = arena.Rent(B * S, KV * D);
        dev.Kernels.AttnBwdFlash(tdq, tdk, tdv, tdo, to, tq, tk, tv, stats, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantDq, tdq.ToArray(), 1e-4, "dQ");
        GpuTestDevice.AssertClose(wantDk, tdk.ToArray(), 1e-4, "dK");
        GpuTestDevice.AssertClose(wantDv, tdv.ToArray(), 1e-4, "dV");
    }

    /// <summary>
    /// The statistics are the whole contract between the two halves: m must be the row maximum
    /// and l the sum of exp(s − m) over j ≤ i, or the backward reconstructs the wrong p.
    /// </summary>
    [Fact]
    public void FlashFwd_WritesRowMaxAndSumExp()
    {
        const int H = 4, KV = 2;
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 41, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 42, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 43, 1f);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D);
        var stats = arena.Rent(B * H * S, FlashAttentionKernels.StatCols);
        dev.Kernels.AttnFwdFlash(to, stats, tq, tk, tv, B, S, H, KV, D); dev.Synchronize();
        var got = stats.ToArray();

        float scale = 1f / MathF.Sqrt(D); int qDim = H * D, kvDim = KV * D, grp = H / KV;
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) { int kvh = h / grp; for (int i = 0; i < S; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j <= i; j++)
            {
                float s = 0; for (int d = 0; d < D; d++) s += q[(b * S + i) * qDim + h * D + d] * k[(b * S + j) * kvDim + kvh * D + d];
                max = MathF.Max(max, s * scale);
            }
            float sum = 0;
            for (int j = 0; j <= i; j++)
            {
                float s = 0; for (int d = 0; d < D; d++) s += q[(b * S + i) * qDim + h * D + d] * k[(b * S + j) * kvDim + kvh * D + d];
                sum += MathF.Exp(s * scale - max);
            }
            int st = ((b * H + h) * S + i) * FlashAttentionKernels.StatCols;
            Assert.True(MathF.Abs(got[st] - max) < 1e-4f, $"m at (b{b},h{h},i{i}): want {max}, got {got[st]}");
            Assert.True(MathF.Abs(got[st + 1] - sum) / MathF.Max(1f, sum) < 1e-4f, $"l at (b{b},h{h},i{i}): want {sum}, got {got[st + 1]}");
        } }
    }

    /// <summary>Destinations are written, not accumulated: stale state must not survive.</summary>
    [Fact]
    public void FlashBwd_OverwritesDestinations()
    {
        const int H = 4, KV = 2;
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 51, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 52, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 53, 1f); var dO = GpuTestDevice.Random(B * S * H * D, 54, 1f);
        var (_, wantP) = AttentionKernelTests.RefForward(q, k, v, H, KV);
        var (wantDq, wantDk, wantDv) = AttentionKernelTests.RefBackward(q, k, v, wantP, dO, H, KV);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D);
        var stats = arena.Rent(B * H * S, FlashAttentionKernels.StatCols);
        dev.Kernels.AttnFwdFlash(to, stats, tq, tk, tv, B, S, H, KV, D);

        var tdo = arena.Rent(B * S, H * D); tdo.Upload(dO);
        var tdq = arena.Rent(B * S, H * D); var tdk = arena.Rent(B * S, KV * D); var tdv = arena.Rent(B * S, KV * D);
        tdq.Upload(GpuTestDevice.Random(B * S * H * D, 55, 7f)); tdk.Upload(GpuTestDevice.Random(B * S * KV * D, 56, 7f));
        tdv.Upload(GpuTestDevice.Random(B * S * KV * D, 57, 7f));
        dev.Kernels.AttnBwdFlash(tdq, tdk, tdv, tdo, to, tq, tk, tv, stats, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantDq, tdq.ToArray(), 1e-4, "dQ");
        GpuTestDevice.AssertClose(wantDk, tdk.ToArray(), 1e-4, "dK");
        GpuTestDevice.AssertClose(wantDv, tdv.ToArray(), 1e-4, "dV");
    }

    /// <summary>
    /// The backward re-reads k and v inside the loop that writes dK and dV, so unlike the
    /// materialised path those may not alias. A silently wrong gradient is the failure this
    /// prevents.
    /// </summary>
    [Fact]
    public void FlashBwd_RejectsDkOverK()
    {
        const int H = 4, KV = 2;
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); var tk = arena.Rent(B * S, KV * D); var tv = arena.Rent(B * S, KV * D);
        var to = arena.Rent(B * S, H * D); var stats = arena.Rent(B * H * S, FlashAttentionKernels.StatCols);
        var tdo = arena.Rent(B * S, H * D); var tdq = arena.Rent(B * S, H * D); var tdv = arena.Rent(B * S, KV * D);
        var ex = Assert.Throws<ArgumentException>(() =>
            dev.Kernels.AttnBwdFlash(tdq, tk, tdv, tdo, to, tq, tk, tv, stats, B, S, H, KV, D));
        Assert.Contains("overlaps", ex.Message);
    }

    [Fact]
    public void FlashFwd_ThrowsOnWrongStatsCols()
    {
        const int H = 4, KV = 2;
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); var tk = arena.Rent(B * S, KV * D); var tv = arena.Rent(B * S, KV * D);
        var to = arena.Rent(B * S, H * D); var bad = arena.Rent(B * H * S, 2);
        var ex = Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwdFlash(to, bad, tq, tk, tv, B, S, H, KV, D));
        Assert.Contains("stats cols", ex.Message);
    }
}
