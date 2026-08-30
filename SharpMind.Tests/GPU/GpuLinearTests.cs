using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.GPU;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class GpuLinearTests
{
    const int M = 9, In = 12, Out = 10;

    /// <summary>
    /// The oracle: y, dx, dA and dB computed straight from the definitions in double,
    /// against the CPU layer's own forward. rank 0 means no adapter.
    /// </summary>
    [Theory]
    [InlineData(true, 3)]    // bias + rank-3 LoRA — the full path
    [InlineData(true, 0)]    // frozen weight + bias, no adapter
    [InlineData(false, 4)]   // no bias
    [InlineData(false, 0)]   // frozen weight only
    public void Forward_Backward_MatchesCpu(bool hasBias, int rank)
    {
        var dev = GpuTestDevice.Device;
        using var w = Tensor<float>.From(GpuTestDevice.Random(In * Out, 41, 1f), In, Out);
        using var biasT = hasBias ? Tensor<float>.From(GpuTestDevice.Random(Out, 42, 1f), Out) : null;
        using var layer = new TrainingLinearLayer("t", In, Out, bias: hasBias, w, biasT);
        List<Parameter> lora = [];
        if (rank > 0)
        {
            layer.EnableLoRA(rank, scale: 2f, new Random(5));
            // B is zero after EnableLoRA; give it values so the LoRA path is exercised.
            GpuTestDevice.Random(rank * Out, 43, 1f).CopyTo(layer.LoRAB!.Data);
            lora = layer.LoRAParameters().ToList();      // [A, B]
        }
        var x = GpuTestDevice.Random(M * In, 44, 1f);
        var dy = GpuTestDevice.Random(M * Out, 45, 1f);

        // CPU forward via the layer itself.
        using var xT = Tensor<float>.From(x, M, In);
        using var yCpu = layer.Forward(xT);

        // CPU backward: dx = dy·Wᵀ + s·(dy·Bᵀ)·Aᵀ ; h = x·A ; dB = s·hᵀ·dy ; dA = xᵀ·(s·dy·Bᵀ)
        int R = rank;
        float s = layer.LoRAScale;
        var a = rank > 0 ? layer.LoRAA!.Data.ToArray() : [];
        var bm = rank > 0 ? layer.LoRAB!.Data.ToArray() : [];
        var wd = w.Data.ToArray();
        var h = new float[M * R]; for (int i = 0; i < M; i++) for (int r = 0; r < R; r++) { double acc = 0; for (int t = 0; t < In; t++) acc += x[i * In + t] * a[t * R + r]; h[i * R + r] = (float)acc; }
        var dH = new float[M * R]; for (int i = 0; i < M; i++) for (int r = 0; r < R; r++) { double acc = 0; for (int o = 0; o < Out; o++) acc += dy[i * Out + o] * bm[r * Out + o]; dH[i * R + r] = s * (float)acc; }
        var wantDx = new float[M * In]; for (int i = 0; i < M; i++) for (int t = 0; t < In; t++) { double acc = 0; for (int o = 0; o < Out; o++) acc += dy[i * Out + o] * wd[t * Out + o]; for (int r = 0; r < R; r++) acc += dH[i * R + r] * a[t * R + r]; wantDx[i * In + t] = (float)acc; }
        var wantDB = new float[R * Out]; for (int r = 0; r < R; r++) for (int o = 0; o < Out; o++) { double acc = 0; for (int i = 0; i < M; i++) acc += h[i * R + r] * dy[i * Out + o]; wantDB[r * Out + o] = s * (float)acc; }
        var wantDA = new float[In * R]; for (int t = 0; t < In; t++) for (int r = 0; r < R; r++) { double acc = 0; for (int i = 0; i < M; i++) acc += x[i * In + t] * dH[i * R + r]; wantDA[t * R + r] = (float)acc; }

        using var gl = new GpuLinear(dev, layer, lora.Count > 0 ? lora[0] : null, lora.Count > 0 ? lora[1] : null);
        Assert.Equal(In, gl.In);
        Assert.Equal(Out, gl.Out);
        Assert.Equal(rank, gl.Rank);
        Assert.Equal(rank > 0, gl.HasLoRA);

        using var arena = new DeviceArena(dev, 1 << 14);
        var tx = arena.Rent(M, In); tx.Upload(x); var ty = arena.Rent(M, Out);
        gl.Forward(ty, tx, arena); dev.Synchronize();
        GpuTestDevice.AssertClose(yCpu.Data, ty.ToArray(), 1e-5, "y");

        var tdy = arena.Rent(M, Out); tdy.Upload(dy); var tdx = arena.Rent(M, In);
        gl.ZeroLoRAGrads();
        gl.Backward(tdx, tdy, tx, arena); dev.Synchronize();
        gl.AccumulateLoRAGradsToHost();
        GpuTestDevice.AssertClose(wantDx, tdx.ToArray(), 1e-5, "dx");
        if (rank > 0)
        {
            GpuTestDevice.AssertClose(wantDA, lora[0].Grad.Data, 1e-5, "dA");
            GpuTestDevice.AssertClose(wantDB, lora[1].Grad.Data, 1e-5, "dB");
        }
        foreach (var p in lora) p.Dispose();
    }

    /// <summary>Grads accumulate across backwards in a step and land on the host additively.</summary>
    [Fact]
    public void LoRAGrads_AccumulateOnDeviceAndOnHost()
    {
        using var fx = new Fixture(rank: 3);
        fx.Gpu.Forward(fx.Y, fx.X, fx.Arena);
        fx.Gpu.ZeroLoRAGrads();
        fx.Gpu.Backward(fx.Dx, fx.Dy, fx.X, fx.Arena);
        fx.Device.Synchronize();
        fx.Gpu.AccumulateLoRAGradsToHost();
        var onceA = fx.LoRA[0].Grad.Data.ToArray();
        var onceB = fx.LoRA[1].Grad.Data.ToArray();

        fx.Gpu.Backward(fx.Dx, fx.Dy, fx.X, fx.Arena);   // no Zero: device grads must accumulate
        fx.Device.Synchronize();
        fx.Gpu.AccumulateLoRAGradsToHost();              // host grads accumulate too: 1 + 2 = 3
        GpuTestDevice.AssertClose(onceA.Select(v => v * 3f).ToArray(), fx.LoRA[0].Grad.Data, 1e-5, "dA x3");
        // dB is the half built from _hs, so this also pins that Backward leaves _hs alone.
        GpuTestDevice.AssertClose(onceB.Select(v => v * 3f).ToArray(), fx.LoRA[1].Grad.Data, 1e-5, "dB x3");
    }

    /// <summary>SyncLoRAToDevice re-uploads what the optimizer wrote into Parameter.Data.</summary>
    [Fact]
    public void SyncLoRAToDevice_PicksUpHostEdits()
    {
        using var fx = new Fixture(rank: 3);
        fx.Gpu.Forward(fx.Y, fx.X, fx.Arena); fx.Device.Synchronize();
        var before = fx.Y.ToArray();

        fx.Layer.LoRAB!.Data.Fill(0f);                   // zero B on the host: the adapter contributes nothing
        fx.Gpu.SyncLoRAToDevice();
        var y2 = fx.Arena.Rent(M, Out);
        fx.Gpu.Forward(y2, fx.X, fx.Arena); fx.Device.Synchronize();
        var after = y2.ToArray();

        Assert.NotEqual(before, after);
        using var frozen = new GpuLinear(fx.Device, fx.Layer, fx.LoRA[0], fx.LoRA[1]);
        var y3 = fx.Arena.Rent(M, Out);
        frozen.Forward(y3, fx.X, fx.Arena); fx.Device.Synchronize();
        GpuTestDevice.AssertClose(y3.ToArray(), after, 1e-6, "y after sync");
    }

    [Fact]
    public void Backward_BetaDxOne_AccumulatesIntoDx()
    {
        using var fx = new Fixture(rank: 3);
        fx.Gpu.Forward(fx.Y, fx.X, fx.Arena);
        fx.Gpu.ZeroLoRAGrads();
        fx.Gpu.Backward(fx.Dx, fx.Dy, fx.X, fx.Arena); fx.Device.Synchronize();
        var once = fx.Dx.ToArray();

        fx.Gpu.Backward(fx.Dx, fx.Dy, fx.X, fx.Arena, betaDx: 1f); fx.Device.Synchronize();
        var twice = fx.Dx.ToArray();

        GpuTestDevice.AssertClose(once.Select(v => v * 2f).ToArray(), twice, 1e-5, "dx accumulated");
    }

    /// <summary>
    /// Rank 1 collapses both strides of five of the eight GEMMs to 1, which
    /// GpuDevice.Gemm rejects as ambiguous — so GpuLinear refuses it up front.
    /// </summary>
    [Fact]
    public void Constructor_RejectsRank1LoRA()
    {
        var dev = GpuTestDevice.Device;
        using var w = Tensor<float>.From(GpuTestDevice.Random(In * Out, 41, 1f), In, Out);
        using var layer = new TrainingLinearLayer("t", In, Out, bias: false, w, null);
        layer.EnableLoRA(1, scale: 2f, new Random(5));
        var lora = layer.LoRAParameters().ToList();
        var ex = Assert.Throws<ArgumentException>(() => new GpuLinear(dev, layer, lora[0], lora[1]));
        Assert.Contains("rank", ex.Message, StringComparison.OrdinalIgnoreCase);
        foreach (var p in lora) p.Dispose();
    }

    [Fact]
    public void Constructor_RejectsForeignOrMissingLoRAParameters()
    {
        var dev = GpuTestDevice.Device;
        using var w = Tensor<float>.From(GpuTestDevice.Random(In * Out, 41, 1f), In, Out);
        using var layer = new TrainingLinearLayer("t", In, Out, bias: false, w, null);
        layer.EnableLoRA(3, scale: 2f, new Random(5));
        var lora = layer.LoRAParameters().ToList();
        using var stranger = new Tensor<float>(In, 3);
        using var pStranger = new Parameter("stranger", stranger);

        Assert.Throws<ArgumentException>(() => new GpuLinear(dev, layer, null, null));
        Assert.Throws<ArgumentException>(() => new GpuLinear(dev, layer, pStranger, lora[1]));
        Assert.Throws<ArgumentException>(() => new GpuLinear(dev, layer, lora[0], pStranger));

        using var plain = new TrainingLinearLayer("p", In, Out, bias: false, w, null);
        Assert.Throws<ArgumentException>(() => new GpuLinear(dev, plain, lora[0], lora[1]));
        foreach (var p in lora) p.Dispose();
    }

    [Fact]
    public void Forward_RejectsBadOperands()
    {
        using var fx = new Fixture(rank: 3);
        Assert.Throws<ArgumentException>(() => fx.Gpu.Forward(fx.Arena.Rent(M, Out + 1), fx.X, fx.Arena));
        Assert.Throws<ArgumentException>(() => fx.Gpu.Forward(fx.Y, fx.Arena.Rent(M, In + 1), fx.Arena));
        Assert.Throws<ArgumentException>(() => fx.Gpu.Forward(fx.Arena.Rent(M + 1, Out), fx.X, fx.Arena));
    }

    [Fact]
    public void Backward_RejectsBadOperands()
    {
        using var fx = new Fixture(rank: 3);
        fx.Gpu.Forward(fx.Y, fx.X, fx.Arena);
        Assert.Throws<ArgumentException>(() => fx.Gpu.Backward(fx.Arena.Rent(M, In + 1), fx.Dy, fx.X, fx.Arena));
        Assert.Throws<ArgumentException>(() => fx.Gpu.Backward(fx.Dx, fx.Arena.Rent(M, Out + 1), fx.X, fx.Arena));
        Assert.Throws<ArgumentException>(() => fx.Gpu.Backward(fx.Dx, fx.Dy, fx.Arena.Rent(M, In + 1), fx.Arena));
        Assert.Throws<ArgumentException>(() => fx.Gpu.Backward(fx.Dx, fx.Dy, fx.Arena.Rent(M + 1, In), fx.Arena));   // x rows vs dy rows — the non-vacuous half
        Assert.Throws<ArgumentException>(() => fx.Gpu.Backward(fx.Dx, fx.Arena.Rent(M + 1, Out), fx.X, fx.Arena));
    }

    /// <summary>hs comes from the matching Forward; without one there is nothing to build dB from.</summary>
    [Fact]
    public void Backward_WithoutForward_Throws()
    {
        using var fx = new Fixture(rank: 3);
        Assert.Throws<InvalidOperationException>(() => fx.Gpu.Backward(fx.Dx, fx.Dy, fx.X, fx.Arena));
    }

    /// <summary>A destination overlapping a source is a read/write race in the GEMM, not an in-place op.</summary>
    [Fact]
    public void RejectsDestinationOverlappingSource()
    {
        var dev = GpuTestDevice.Device;
        using var w = Tensor<float>.From(GpuTestDevice.Random(In * In, 41, 1f), In, In);   // square: dst and src can share a window
        using var layer = new TrainingLinearLayer("sq", In, In, bias: false, w, null);
        using var gl = new GpuLinear(dev, layer, null, null);
        using var arena = new DeviceArena(dev, 1 << 12);
        var t = arena.Rent(M, In); var other = arena.Rent(M, In);
        Assert.Throws<ArgumentException>(() => gl.Forward(t, t, arena));            // y over x
        Assert.Throws<ArgumentException>(() => gl.Backward(t, t, other, arena));    // dx over dy
        Assert.Throws<ArgumentException>(() => gl.Backward(t, other, t, arena));    // dx over x
    }

    /// <summary>A layer with no adapter: the LoRA calls are no-ops, not crashes.</summary>
    [Fact]
    public void NoLoRA_SyncAndGradCallsAreNoOps()
    {
        using var fx = new Fixture(rank: 0);
        fx.Gpu.SyncLoRAToDevice();
        fx.Gpu.ZeroLoRAGrads();
        fx.Gpu.AccumulateLoRAGradsToHost();
        Assert.False(fx.Gpu.HasLoRA);
        Assert.Equal(0, fx.Gpu.Rank);
    }

    /// <summary>Layer + device tensors for the tests that do not need the CPU oracle.</summary>
    private sealed class Fixture : IDisposable
    {
        public GpuDevice Device { get; } = GpuTestDevice.Device;
        public TrainingLinearLayer Layer { get; }
        public List<Parameter> LoRA { get; } = [];
        public GpuLinear Gpu { get; }
        public DeviceArena Arena { get; }
        public DeviceTensor X { get; }
        public DeviceTensor Y { get; }
        public DeviceTensor Dx { get; }
        public DeviceTensor Dy { get; }
        private readonly Tensor<float> _w;

        public Fixture(int rank)
        {
            _w = Tensor<float>.From(GpuTestDevice.Random(In * Out, 41, 1f), In, Out);
            Layer = new TrainingLinearLayer("t", In, Out, bias: true, _w, Tensor<float>.From(GpuTestDevice.Random(Out, 42, 1f), Out));
            if (rank > 0)
            {
                Layer.EnableLoRA(rank, scale: 2f, new Random(5));
                GpuTestDevice.Random(rank * Out, 43, 1f).CopyTo(Layer.LoRAB!.Data);
                LoRA = Layer.LoRAParameters().ToList();
            }
            Gpu = new GpuLinear(Device, Layer, LoRA.Count > 0 ? LoRA[0] : null, LoRA.Count > 0 ? LoRA[1] : null);
            Arena = new DeviceArena(Device, 1 << 14);
            X = Arena.Rent(M, In); X.Upload(GpuTestDevice.Random(M * In, 44, 1f));
            Y = Arena.Rent(M, Out);
            Dy = Arena.Rent(M, Out); Dy.Upload(GpuTestDevice.Random(M * Out, 45, 1f));
            Dx = Arena.Rent(M, In);
        }

        public void Dispose()
        {
            Arena.Dispose();
            Gpu.Dispose();
            foreach (var p in LoRA) p.Dispose();
            Layer.Dispose();
            _w.Dispose();
        }
    }
}
