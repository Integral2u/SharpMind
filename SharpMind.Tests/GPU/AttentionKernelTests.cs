using SharpMind.GPU;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class AttentionKernelTests
{
    internal const int B = 2, S = 6, H = 4, KV = 2, D = 8;   // guard-test shape; GQA: two query heads per kv head

    /// <summary>(numHeads, numKv): MHA (grp 1), GQA (grp 2), MQA (grp = numHeads).</summary>
    public static TheoryData<int, int> HeadShapes => new() { { 4, 4 }, { 4, 2 }, { 4, 1 } };

    internal static (float[] o, float[] p) RefForward(float[] q, float[] k, float[] v, int H, int KV)
    {
        float scale = 1f / MathF.Sqrt(D); int qDim = H * D, kvDim = KV * D, grp = H / KV;
        var o = new float[B * S * qDim]; var p = new float[B * H * S * S];
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) { int kvh = h / grp; for (int i = 0; i < S; i++)
        {
            int pBase = ((b * H + h) * S + i) * S; float max = float.NegativeInfinity;
            for (int j = 0; j <= i; j++) { float s = 0; for (int d = 0; d < D; d++) s += q[(b * S + i) * qDim + h * D + d] * k[(b * S + j) * kvDim + kvh * D + d]; s *= scale; p[pBase + j] = s; max = MathF.Max(max, s); }
            float sum = 0; for (int j = 0; j <= i; j++) { p[pBase + j] = MathF.Exp(p[pBase + j] - max); sum += p[pBase + j]; }
            for (int j = 0; j <= i; j++) p[pBase + j] /= sum;
            for (int d = 0; d < D; d++) { float acc = 0; for (int j = 0; j <= i; j++) acc += p[pBase + j] * v[(b * S + j) * kvDim + kvh * D + d]; o[(b * S + i) * qDim + h * D + d] = acc; }
        } }
        return (o, p);
    }

    // dP = dO·Vᵀ ; dS = P∘(dP − rowsum(dP∘P)) ; dQ = scale·dS·K ; dK += scale·dSᵀ·Q ; dV += Pᵀ·dO   (GradientMapping.Attention)
    internal static (float[] dq, float[] dk, float[] dv) RefBackward(float[] q, float[] k, float[] v, float[] p, float[] dO, int H, int KV)
    {
        float scale = 1f / MathF.Sqrt(D); int qDim = H * D, kvDim = KV * D, grp = H / KV;
        var dq = new float[B * S * qDim]; var dk = new float[B * S * kvDim]; var dv = new float[B * S * kvDim];
        for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) { int kvh = h / grp; for (int i = 0; i < S; i++)
        {
            int pBase = ((b * H + h) * S + i) * S; var dP = new float[S]; float rowSum = 0;
            for (int j = 0; j <= i; j++) { float s = 0; for (int d = 0; d < D; d++) s += dO[(b * S + i) * qDim + h * D + d] * v[(b * S + j) * kvDim + kvh * D + d]; dP[j] = s; rowSum += s * p[pBase + j]; }
            for (int j = 0; j <= i; j++)
            {
                float dS = p[pBase + j] * (dP[j] - rowSum);
                for (int d = 0; d < D; d++)
                {
                    dq[(b * S + i) * qDim + h * D + d] += scale * dS * k[(b * S + j) * kvDim + kvh * D + d];
                    dk[(b * S + j) * kvDim + kvh * D + d] += scale * dS * q[(b * S + i) * qDim + h * D + d];
                    dv[(b * S + j) * kvDim + kvh * D + d] += p[pBase + j] * dO[(b * S + i) * qDim + h * D + d];
                }
            }
        } }
        return (dq, dk, dv);
    }

    [Theory]
    [MemberData(nameof(HeadShapes))]
    public void AttnFwd_Bwd_MatchReference(int H, int KV)
    {
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 21, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 22, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 23, 1f); var dO = GpuTestDevice.Random(B * S * H * D, 24, 1f);
        var (wantO, wantP) = RefForward(q, k, v, H, KV);
        var (wantDq, wantDk, wantDv) = RefBackward(q, k, v, wantP, dO, H, KV);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D); var tp = arena.Rent(B * H * S, S); tp.Zero();
        dev.Kernels.AttnFwd(to, tp, tq, tk, tv, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantO, to.ToArray(), 1e-5, "out");
        GpuTestDevice.AssertClose(wantP, tp.ToArray(), 1e-5, "probs");

        var tdo = arena.Rent(B * S, H * D); tdo.Upload(dO);
        var tdq = arena.Rent(B * S, H * D); var tdk = arena.Rent(B * S, KV * D); var tdv = arena.Rent(B * S, KV * D); var scratch = arena.Rent(B * H * S, S);
        dev.Kernels.AttnBwd(tdq, tdk, tdv, tdo, tq, tk, tv, tp, scratch, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantDq, tdq.ToArray(), 1e-4, "dQ");
        GpuTestDevice.AssertClose(wantDk, tdk.ToArray(), 1e-4, "dK");
        GpuTestDevice.AssertClose(wantDv, tdv.ToArray(), 1e-4, "dV");
    }

    /// <summary>dK/dV are written, not accumulated: pre-existing garbage in the destination must not survive.</summary>
    [Fact]
    public void AttnBwd_OverwritesDestinations()
    {
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 31, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 32, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 33, 1f); var dO = GpuTestDevice.Random(B * S * H * D, 34, 1f);
        var (_, wantP) = RefForward(q, k, v, H, KV);
        var (wantDq, wantDk, wantDv) = RefBackward(q, k, v, wantP, dO, H, KV);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D); var tp = arena.Rent(B * H * S, S); tp.Zero();
        dev.Kernels.AttnFwd(to, tp, tq, tk, tv, B, S, H, KV, D);

        var tdo = arena.Rent(B * S, H * D); tdo.Upload(dO);
        var tdq = arena.Rent(B * S, H * D); var tdk = arena.Rent(B * S, KV * D); var tdv = arena.Rent(B * S, KV * D); var scratch = arena.Rent(B * H * S, S);
        // Poison every destination and the scratch: correct results prove no read of stale state.
        tdq.Upload(GpuTestDevice.Random(B * S * H * D, 35, 7f)); tdk.Upload(GpuTestDevice.Random(B * S * KV * D, 36, 7f));
        tdv.Upload(GpuTestDevice.Random(B * S * KV * D, 37, 7f)); scratch.Upload(GpuTestDevice.Random(B * H * S * S, 38, 7f));
        dev.Kernels.AttnBwd(tdq, tdk, tdv, tdo, tq, tk, tv, tp, scratch, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantDq, tdq.ToArray(), 1e-4, "dQ");
        GpuTestDevice.AssertClose(wantDk, tdk.ToArray(), 1e-4, "dK");
        GpuTestDevice.AssertClose(wantDv, tdv.ToArray(), 1e-4, "dV");
    }

    [Fact]
    public void AttnFwd_ThrowsOnWrongQCols()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D - 1); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D - 1); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnWrongKvCols()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D + D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnMismatchedVCols()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D - 1);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnWrongRowCount()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S + 1, H * D); var k = arena.Rent(B * S + 1, KV * D); var v = arena.Rent(B * S + 1, KV * D);
        var o = arena.Rent(B * S + 1, H * D); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnWrongProbsShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S, S + 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnWrongProbsRows()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S - S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    [Fact]
    public void AttnFwd_ThrowsWhenHeadsNotDivisibleByKv()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        const int badKv = 3;
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, badKv * D); var v = arena.Rent(B * S, badKv * D);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, badKv, D));
    }

    [Fact]
    public void AttnFwd_ThrowsOnMismatchedOutShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D - D); var p = arena.Rent(B * H * S, S);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, KV, D));
    }

    /// <summary>The positivity guard is ordered first so numKv == 0 cannot reach the host's
    /// numHeads % numKv or the kernel's numHeads / numKv — both divide by zero.</summary>
    [Fact]
    public void AttnFwd_ThrowsOnNonPositiveDimension()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var o = arena.Rent(B * S, H * D); var p = arena.Rent(B * H * S, S);
        // Not just "an ArgumentException": without this branch numKv == 0 would be a DivideByZeroException.
        Assert.Contains("must be positive", Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, S, H, numKv: 0, D)).Message);
        Assert.Contains("must be positive", Assert.Throws<ArgumentException>(() => dev.Kernels.AttnFwd(o, p, q, k, v, B, seqLen: 0, H, KV, D)).Message);
    }

    /// <summary>Launch 3 still reads q after launch 2 has overwritten dQ, so an in-place dQ would
    /// silently corrupt dK. Same(dQ, q) makes the two shape-identical, so only this guard catches it.</summary>
    [Fact]
    public void AttnBwd_ThrowsWhenDQAliasesQ()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var p = arena.Rent(B * H * S, S); var dOut = arena.Rent(B * S, H * D); var scratch = arena.Rent(B * H * S, S);
        var dK = arena.Rent(B * S, KV * D); var dV = arena.Rent(B * S, KV * D);
        Assert.Contains("dQ overlaps q", Assert.Throws<ArgumentException>(() => dev.Kernels.AttnBwd(q, dK, dV, dOut, q, k, v, p, scratch, B, S, H, KV, D)).Message);
    }

    [Fact]
    public void AttnBwd_ThrowsWhenScratchAliasesProbs()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var p = arena.Rent(B * H * S, S); var dOut = arena.Rent(B * S, H * D);
        var dQ = arena.Rent(B * S, H * D); var dK = arena.Rent(B * S, KV * D); var dV = arena.Rent(B * S, KV * D);
        Assert.Contains("dProbsScratch overlaps probs", Assert.Throws<ArgumentException>(() => dev.Kernels.AttnBwd(dQ, dK, dV, dOut, q, k, v, p, p, B, S, H, KV, D)).Message);
    }

    /// <summary>dK/dV may alias k/v — those are read in launches 1-2, before launch 3 writes.
    /// The overlap guard must not reject that.</summary>
    [Fact]
    public void AttnBwd_AllowsDKAliasingK()
    {
        var dev = GpuTestDevice.Device;
        var q = GpuTestDevice.Random(B * S * H * D, 41, 1f); var k = GpuTestDevice.Random(B * S * KV * D, 42, 1f);
        var v = GpuTestDevice.Random(B * S * KV * D, 43, 1f); var dO = GpuTestDevice.Random(B * S * H * D, 44, 1f);
        var (_, wantP) = RefForward(q, k, v, H, KV);
        var (_, wantDk, wantDv) = RefBackward(q, k, v, wantP, dO, H, KV);

        using var arena = new DeviceArena(dev, 1 << 16);
        var tq = arena.Rent(B * S, H * D); tq.Upload(q); var tk = arena.Rent(B * S, KV * D); tk.Upload(k); var tv = arena.Rent(B * S, KV * D); tv.Upload(v);
        var to = arena.Rent(B * S, H * D); var tp = arena.Rent(B * H * S, S); tp.Zero();
        dev.Kernels.AttnFwd(to, tp, tq, tk, tv, B, S, H, KV, D);
        var tdo = arena.Rent(B * S, H * D); tdo.Upload(dO);
        var tdq = arena.Rent(B * S, H * D); var scratch = arena.Rent(B * H * S, S);
        // dK writes over k and dV over v, in place.
        dev.Kernels.AttnBwd(tdq, tk, tv, tdo, tq, tk, tv, tp, scratch, B, S, H, KV, D); dev.Synchronize();
        GpuTestDevice.AssertClose(wantDk, tk.ToArray(), 1e-4, "dK in place over k");
        GpuTestDevice.AssertClose(wantDv, tv.ToArray(), 1e-4, "dV in place over v");
    }

    [Fact]
    public void AttnBwd_ThrowsOnWrongScratchShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var p = arena.Rent(B * H * S, S); var dOut = arena.Rent(B * S, H * D);
        var dQ = arena.Rent(B * S, H * D); var dK = arena.Rent(B * S, KV * D); var dV = arena.Rent(B * S, KV * D);
        var badScratch = arena.Rent(B * H * S, S - 1);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnBwd(dQ, dK, dV, dOut, q, k, v, p, badScratch, B, S, H, KV, D));
    }

    [Fact]
    public void AttnBwd_ThrowsOnMismatchedGradientShapes()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 16);
        var q = arena.Rent(B * S, H * D); var k = arena.Rent(B * S, KV * D); var v = arena.Rent(B * S, KV * D);
        var p = arena.Rent(B * H * S, S); var dOut = arena.Rent(B * S, H * D); var scratch = arena.Rent(B * H * S, S);
        var dQ = arena.Rent(B * S, H * D); var dV = arena.Rent(B * S, KV * D);
        var badDk = arena.Rent(B * S, H * D);   // query-shaped, not kv-shaped
        Assert.Throws<ArgumentException>(() => dev.Kernels.AttnBwd(dQ, badDk, dV, dOut, q, k, v, p, scratch, B, S, H, KV, D));
    }
}
