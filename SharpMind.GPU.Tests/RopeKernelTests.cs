using SharpMind.Core.Embeddings;
using SharpMind.Core.Tensors;
using SharpMind.GPU;
using Xunit;

namespace SharpMind.GPU.Tests;

[Collection("GPU")]
public sealed class RopeKernelTests
{
    const int B = 2, S = 5, H = 3, D = 8;

    [Theory] [InlineData(false)] [InlineData(true)]
    public void RopeFwd_And_Bwd_MatchCpuRoPE(bool neox)
    {
        var dev = GpuTestDevice.Device;
        var rope = new RoPE(D, maxSeqLen: 16, theta: 10_000f, neoxStyle: neox);
        var data = GpuTestDevice.Random(B * S * H * D, 7, 1f);

        using var cpu = Tensor<float>.From(data, B, S, H, D);
        rope.ApplyBatched(cpu, 0);
        var wantFwd = cpu.Data.ToArray();
        rope.ApplyBatchedBackward(cpu, 0);          // inverse rotation brings the data back
        var wantRoundTrip = cpu.Data.ToArray();

        using var arena = new DeviceArena(dev, 1 << 12);
        var cos = arena.Rent(16, D / 2); cos.Upload(rope.CosTable);
        var sin = arena.Rent(16, D / 2); sin.Upload(rope.SinTable);
        var x = arena.Rent(B * S, H * D); x.Upload(data);
        dev.Kernels.RopeFwd(x, cos, sin, S, H, D, rope.RopeDim, neox); dev.Synchronize();
        GpuTestDevice.AssertClose(wantFwd, x.ToArray(), 1e-5, "fwd");
        dev.Kernels.RopeBwd(x, cos, sin, S, H, D, rope.RopeDim, neox); dev.Synchronize();
        GpuTestDevice.AssertClose(wantRoundTrip, x.ToArray(), 1e-5, "bwd");
        GpuTestDevice.AssertClose(data, x.ToArray(), 1e-4, "round trip");
    }

    // Partial RoPE (ropeDim < headDim): dims [ropeDim, headDim) must pass through
    // untouched. The main test above uses ropeDim == headDim and cannot see this —
    // a kernel that mistakenly sized its launch/indexing off headDim instead of
    // ropeDim would still pass it but corrupt the un-rotated tail here.
    [Theory] [InlineData(false)] [InlineData(true)]
    public void RopeFwd_And_Bwd_LeaveUnrotatedTailUntouched_WhenRopeDimSmallerThanHeadDim(bool neox)
    {
        const int ropeDim = 4; // < D (8)
        var dev = GpuTestDevice.Device;
        var rope = new RoPE(D, maxSeqLen: 16, theta: 10_000f, ropeDim: ropeDim, neoxStyle: neox);
        var data = GpuTestDevice.Random(B * S * H * D, 11, 1f);

        using var cpu = Tensor<float>.From(data, B, S, H, D);
        rope.ApplyBatched(cpu, 0);
        var wantFwd = cpu.Data.ToArray();
        rope.ApplyBatchedBackward(cpu, 0);
        var wantRoundTrip = cpu.Data.ToArray();

        using var arena = new DeviceArena(dev, 1 << 12);
        var cos = arena.Rent(16, ropeDim / 2); cos.Upload(rope.CosTable);
        var sin = arena.Rent(16, ropeDim / 2); sin.Upload(rope.SinTable);
        var x = arena.Rent(B * S, H * D); x.Upload(data);
        dev.Kernels.RopeFwd(x, cos, sin, S, H, D, rope.RopeDim, neox); dev.Synchronize();
        GpuTestDevice.AssertClose(wantFwd, x.ToArray(), 1e-5, "fwd");
        dev.Kernels.RopeBwd(x, cos, sin, S, H, D, rope.RopeDim, neox); dev.Synchronize();
        GpuTestDevice.AssertClose(wantRoundTrip, x.ToArray(), 1e-5, "bwd");
        GpuTestDevice.AssertClose(data, x.ToArray(), 1e-4, "round trip");
    }
}
