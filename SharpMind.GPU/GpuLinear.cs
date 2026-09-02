using System.Buffers;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.GPU;

/// <summary>
/// A linear layer resident on the device: frozen W in the layer's [In, Out] layout
/// (so every product is a stride pattern, never a transpose), optional frozen bias,
/// optional LoRA adapter whose A/B mirror the host Parameters and whose grads live
/// on device until <see cref="AccumulateLoRAGradsToHost"/>.
///
/// The device grads accumulate across every <see cref="Backward"/> in a step; the
/// caller zeroes them once per optimizer step with <see cref="ZeroLoRAGrads"/>.
/// <see cref="Forward"/> parks s·(x·A) in arena memory for the matching
/// <see cref="Backward"/>, so the arena must not be reset between the two.
/// </summary>
internal sealed class GpuLinear : IDisposable
{
    private readonly GpuDevice _dev;
    private readonly DeviceBuffer _w;
    private readonly DeviceBuffer? _bias;
    private readonly DeviceByteBuffer? _rawW;
    private readonly QuantDType? _quant;
    private readonly DeviceBuffer? _a, _b, _dA, _dB;
    private readonly Parameter? _pA, _pB;
    private readonly float _scale;
    private readonly float[]? _gradScratch;
    private DeviceTensor _hs;   // s·(x·A) from the last Forward (arena memory, valid until Reset)
    /// <summary>A quantized-resident layer whose quant has no on-device kernel (e.g. Q8_K): its
    /// forward is host-routed — download x, run the layer's own CPU quantized matmul, upload y.</summary>
    private readonly InferenceLinearLayer? _hostLayer;

    public int In { get; }
    public int Out { get; }
    public int Rank { get; }
    public bool HasLoRA => _a is not null;

    public GpuLinear(GpuDevice device, LinearLayer layer, Parameter? loraA, Parameter? loraB)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(layer);
        _dev = device; In = layer.InFeatures; Out = layer.OutFeatures;
        // Same unwind as GpuBlock/GpuBackpropEngine: the caller only takes ownership once this
        // constructor returns, so a mid-sequence failure has to free what it already allocated.
        var owned = new List<IDisposable>();
        try
        {
            // A quantized-resident layer (the standard chat loader's InferenceLinearLayer) holds its
            // real weights as GGUF raw bytes: layer.Weight is only a tiny [In,1] place-holder, so the
            // F32 GEMM below would read far past it — the "B holds N floats, GEMM needs M" crash.
            // Upload the raw bytes and run the on-device dequant matmul for the dtypes with a kernel
            // (F32 raw data — InferenceLinearLayer with QuantDtype.F32 — is a full float weight and
            // takes the ordinary path). A quant with no device kernel (e.g. Q8_K) is host-routed: the
            // layer's own CPU forward runs the matmul, so every quant the CPU transformer runs works,
            // just on the host for that tensor.
            if (layer is InferenceLinearLayer inf && inf.RawQuantizedData is not null)
            {
                if (IsOnDeviceDequant(inf.QuantDtype))
                {
                    _quant = inf.QuantDtype;
                    _rawW = new DeviceByteBuffer(device.Accelerator, inf.RawQuantizedData); owned.Add(_rawW);
                    _w = DeviceBuffer.From(device, layer.Weight); owned.Add(_w);
                }
                else
                {
                    _hostLayer = inf;
                    _w = null!;   // never used: the host-routed forward uploads x/y around the CPU matmul
                }
            }
            else
            {
                _w = DeviceBuffer.From(device, layer.Weight); owned.Add(_w);
            }
            // The host-routed forward bicycles the layer's own bias (InferenceLinearLayer.Forward adds
            // it); uploading a device copy would be dead weight.
            if (_hostLayer is null && layer.Bias is { } bias) { _bias = DeviceBuffer.From(device, bias); owned.Add(_bias); }
            if (layer is TrainingLinearLayer { HasLoRA: true } t)
            {
                if (loraA is null || loraB is null || !ReferenceEquals(loraA.Data, t.LoRAA) || !ReferenceEquals(loraB.Data, t.LoRAB))
                    throw new ArgumentException($"{layer.Name}: LoRA parameters must wrap the layer's own A and B tensors.");
                // ponytail: rank 1 rejected, not special-cased. It collapses both strides of five of the
                // eight GEMMs to (1,1) — an operand that is at once row- and column-major — which
                // GpuDevice.Gemm rejects because cuBLAS reads one stride of each operand as its leading
                // dimension and the right one then depends on which dimension degenerated. Teaching Gemm
                // that would put an untestable (no NVIDIA card here) branch on the hot path for an
                // adapter nobody trains. Raise the rank instead.
                if (t.LoRARank < 2)
                    throw new ArgumentException($"{layer.Name}: GPU LoRA needs rank >= 2, got {t.LoRARank}; at rank 1 the adapter's GEMM strides collapse to an ambiguous (1,1) pair.");
                Rank = t.LoRARank; _scale = t.LoRAScale; _pA = loraA; _pB = loraB;
                _a = DeviceBuffer.From(device, t.LoRAA!); owned.Add(_a);
                _b = DeviceBuffer.From(device, t.LoRAB!); owned.Add(_b);
                _dA = new DeviceBuffer(device, In, Rank); owned.Add(_dA);
                _dB = new DeviceBuffer(device, Rank, Out); owned.Add(_dB);
                ZeroLoRAGrads();   // fresh device memory is undefined; a caller who forgets the first Zero gets zeros, not garbage
                _gradScratch = new float[Math.Max(In * Rank, Rank * Out)];
            }
            else if (loraA is not null || loraB is not null)
                throw new ArgumentException($"{layer.Name}: LoRA parameters passed for a layer with no adapter.");
        }
        catch
        {
            foreach (var d in owned) d.Dispose();
            throw;
        }
    }

    /// <summary>Host A, B (what the optimizer updated) → device. Call at the start of a step.</summary>
    public void SyncLoRAToDevice()
    {
        if (_a is null) return;
        _a.Tensor.Upload(_pA!.Data.Data); _b!.Tensor.Upload(_pB!.Data.Data);
    }

    /// <summary>Zeroes the device grads. Once per optimizer step, before the step's first Backward.</summary>
    public void ZeroLoRAGrads() { _dA?.Tensor.Zero(); _dB?.Tensor.Zero(); }

    /// <summary>Device dA, dB → Parameter.Grad, adding (so micro-batches accumulate on the host too).</summary>
    public void AccumulateLoRAGradsToHost()
    {
        if (_dA is null) return;
        var s = _gradScratch!.AsSpan(0, In * Rank); _dA.Tensor.Download(s); _pA!.AccumulateGrad(s);
        s = _gradScratch.AsSpan(0, Rank * Out); _dB!.Tensor.Download(s); _pB!.AccumulateGrad(s);
    }

    /// <summary>y = x·W (+ bias) (+ s·h·B with h = x·A). Keeps hs = s·h for the backward.</summary>
    public void Forward(DeviceTensor y, DeviceTensor x, DeviceArena arena)
    {
        ArgumentNullException.ThrowIfNull(arena);
        int m = x.Rows;
        Check(x, m, In, "x"); Check(y, m, Out, "y");
        GpuKernels.NoOverlap(y, x, "y", "x");
        if (_hostLayer is not null) { ForwardHost(y, x, m); return; }
        if (_quant is { } q) _dev.Kernels.DequantMatmul(y, x, _rawW!, In, Out, q);       // y = x·W (quantized)
        else _dev.Gemm(y, x, _w.Tensor, m, Out, In, saI: In, saK: 1, sbK: Out, sbJ: 1);   // y = x·W
        if (_bias is not null) _dev.Kernels.AddBiasRows(y, _bias.Tensor);
        if (_a is null) return;
        _hs = arena.Rent(m, Rank);
        _dev.Gemm(_hs, x, _a.Tensor, m, Rank, In, saI: In, saK: 1, sbK: Rank, sbJ: 1);                // h = x·A
        _dev.Kernels.Scale(_hs, _scale);                                                              // hs = s·h
        _dev.Gemm(y, _hs, _b!.Tensor, m, Out, Rank, saI: Rank, saK: 1, sbK: Out, sbJ: 1, beta: 1f);   // y += hs·B
    }

    /// <summary>
    /// Per-tensor CPU fallback for a quant with no on-device kernel: download <paramref name="x"/>,
    /// run the layer's own host <see cref="InferenceLinearLayer.Forward"/> (the exact call the CPU
    /// transformer makes, raw-weighted, bias included), upload the result back into <paramref name="y"/>.
    /// The download forces a device sync, so the rest of the model keeps its GPU kernels.
    /// </summary>
    private unsafe void ForwardHost(DeviceTensor y, DeviceTensor x, int m)
    {
        int xLen = m * In;
        int yLen = m * Out;
        float[] xh = ArrayPool<float>.Shared.Rent(xLen);
        float[] yh = ArrayPool<float>.Shared.Rent(yLen);
        try
        {
            x.Download(xh.AsSpan(0, xLen));
            using (var xt = Tensor<float>.From(xh.AsSpan(0, xLen), m, In))
            using (var yt = _hostLayer!.Forward(xt, workspace: null))
            {
                if (yt.ElementCount != yLen)
                    throw new InvalidOperationException($"{_hostLayer.Name}: host forward produced {yt.ElementCount} elements, expected {yLen}.");
                yt.Data.CopyTo(yh.AsSpan(0, yLen));
            }
            y.Upload(yh.AsSpan(0, yLen));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(xh);
            ArrayPool<float>.Shared.Return(yh);
        }
    }

    /// <summary>True for the quant dtypes with an on-device dequant matmul kernel in this engine.</summary>
    private static bool IsOnDeviceDequant(QuantDType q) => GpuInferenceEngine.IsOnDeviceDequant(q);

    /// <summary>dx = dy·Wᵀ (+ dH·Aᵀ with dH = s·dy·Bᵀ); dB += hsᵀ·dy; dA += xᵀ·dH. betaDx = 1 accumulates into dx.</summary>
    public void Backward(DeviceTensor dx, DeviceTensor dy, DeviceTensor x, DeviceArena arena, float betaDx = 0f)
    {
        ArgumentNullException.ThrowIfNull(arena);
        int m = dy.Rows;
        Check(dy, m, Out, "dy"); Check(dx, m, In, "dx"); Check(x, m, In, "x");
        // Every check before the first GEMM: a throw must not leave a half-written dx behind.
        if (_a is not null && (_hs.Rows != m || _hs.Cols != Rank))
            throw new InvalidOperationException($"Backward needs the matching Forward's hs [{m},{Rank}], got [{_hs.Rows},{_hs.Cols}]. Forward first, and do not reset the arena in between.");
        GpuKernels.NoOverlap(dx, dy, "dx", "dy");
        GpuKernels.NoOverlap(dx, x, "dx", "x");
        _dev.Gemm(dx, dy, _w.Tensor, m, In, Out, saI: Out, saK: 1, sbK: 1, sbJ: Out, beta: betaDx);   // dx = dy·Wᵀ   B[k=o, j=t] = W[t·Out + o]
        if (_a is null) return;
        var dH = arena.Rent(m, Rank);
        _dev.Gemm(dH, dy, _b!.Tensor, m, Rank, Out, saI: Out, saK: 1, sbK: 1, sbJ: Out);              // dH = dy·Bᵀ   B[k=o, j=r] = B[r·Out + o]
        _dev.Kernels.Scale(dH, _scale);                                                               // dH = s·dy·Bᵀ
        _dev.Gemm(dx, dH, _a.Tensor, m, In, Rank, saI: Rank, saK: 1, sbK: 1, sbJ: Rank, beta: 1f);    // dx += dH·Aᵀ  B[k=r, j=t] = A[t·Rank + r]
        _dev.Gemm(_dB!.Tensor, _hs, dy, Rank, Out, m, saI: 1, saK: Rank, sbK: Out, sbJ: 1, beta: 1f); // dB += hsᵀ·dy A[i=r, k=row] = hs[row·Rank + r]
        _dev.Gemm(_dA!.Tensor, x, dH, In, Rank, m, saI: 1, saK: In, sbK: Rank, sbJ: 1, beta: 1f);     // dA += xᵀ·dH  A[i=t, k=row] = x[row·In + t]
    }

    private static void Check(DeviceTensor t, int rows, int cols, string name)
    {
        if (t.Rows != rows || t.Cols != cols)
            throw new ArgumentException($"{name} must be [{rows},{cols}], got [{t.Rows},{t.Cols}].");
    }

    public void Dispose() { _w?.Dispose(); _rawW?.Dispose(); _bias?.Dispose(); _a?.Dispose(); _b?.Dispose(); _dA?.Dispose(); _dB?.Dispose(); }
}
