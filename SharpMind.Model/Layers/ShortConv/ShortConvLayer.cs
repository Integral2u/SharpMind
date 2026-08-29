using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.ShortConv;

/// <summary>
/// LFM2 short-conv (no-attention) block branch — Section 2.2 of the LFM2
/// technical report. A single in-projection fans out to the <c>[b, c, x]</c>
/// gates, a depthwise 1D conv mixes each channel's short history, and an
/// out-projection returns to the hidden dimension:
///
///   bx  = b ⊙ x;   convOut = conv([state | bx], kernel);   y = c ⊙ convOut;   out = WOut y
///
/// The conv is fully local (its history lives in <see cref="ShortConvCache"/>), so
/// it replaces the KV-cache machinery of an attention block entirely — the layer
/// never touches a KV cache, and the block feeds it a <see cref="ShortConvCache"/>
/// only so the generator's uniform reset/trim/truncate/snapshot loops treat every
/// layer alike.
/// </summary>
public sealed class ShortConvLayer : IDisposable
{
    private readonly ModelConfig _config;
    private readonly LinearLayer _wIn;    // [HiddenDim, 3*HiddenDim]
    private readonly LinearLayer _wOut;   // [HiddenDim, HiddenDim]
    private Tensor<float> _kernel;        // [l_cache, HiddenDim] — always F32
    private readonly bool _ownsKernel;
    private readonly Tensor<float> _fallbackState; // [1, l_cache-1, HiddenDim] for training / no-cache paths
    private bool _disposed;

    public ShortConvLayer(ModelConfig config, QuantizationOps qOps,
        TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        int hidden = config.HiddenDim;
        int lCache = config.ShortConvCacheLength;

        var tm = weights?.TensorMeta;
        var inDtype = weights?.QuantDtypeWScIn ?? tm?.GetValueOrDefault("RawWScIn").Dtype ?? QuantDType.F32;
        var outDtype = weights?.QuantDtypeWScOut ?? tm?.GetValueOrDefault("RawWScOut").Dtype ?? QuantDType.F32;

        _wIn = LinearLayerFactory.Create("sc_in_proj", hidden, 3 * hidden, false,
            weights?.WScIn, null, inDtype, mapping);
        _wOut = LinearLayerFactory.Create("sc_out_proj", hidden, hidden, false,
            weights?.WScOut, null, outDtype, mapping);

        if (weights?.WScConv is not null)
        {
            _kernel = weights.WScConv;
            _ownsKernel = false;
        }
        else
        {
            _kernel = new Tensor<float>(lCache, hidden);
            _ownsKernel = true;
        }

        _fallbackState = new Tensor<float>(1, lCache - 1, hidden);
    }

    public LinearLayer WIn => _wIn;
    public LinearLayer WOut => _wOut;
    internal Tensor<float> Kernel => _kernel;
    internal int StateRows => _config.ShortConvCacheLength - 1;
    internal int KernelRows => _config.ShortConvCacheLength;

    public void SetKernel(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _kernel.ElementCount)
            throw new ArgumentException($"Expected {_kernel.ElementCount} kernel values, got {data.Length}.");
        data.CopyTo(_kernel.Data);
    }

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (name.Contains("shortconv.in_proj", StringComparison.OrdinalIgnoreCase))
            _wIn.LoadWeightTransposed(data);
        else if (name.Contains("shortconv.out_proj", StringComparison.OrdinalIgnoreCase))
            _wOut.LoadWeightTransposed(data);
        else if (name.Contains("shortconv.conv.weight", StringComparison.OrdinalIgnoreCase))
            SetKernel(data);
    }

    public bool SetRawWeight(string name, byte[] rawData, QuantDType _)
    {
        ThrowIfDisposed();
        if (name.EndsWith(".bias", StringComparison.OrdinalIgnoreCase)) return false;
        if (name.Contains("shortconv.in_proj", StringComparison.OrdinalIgnoreCase))
        {
            _wIn.SetRawWeight(rawData);
            return true;
        }
        if (name.Contains("shortconv.out_proj", StringComparison.OrdinalIgnoreCase))
        {
            _wOut.SetRawWeight(rawData);
            return true;
        }
        return false;
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        ThrowIfDisposed();
        if (weights.WScIn != null) _wIn.ReplaceWeights(weights.WScIn, null);
        _wIn.SetRawWeight(weights.RawWScIn);
        if (weights.WScOut != null) _wOut.ReplaceWeights(weights.WScOut, null);
        _wOut.SetRawWeight(weights.RawWScOut);
        if (weights.WScConv is not null)
        {
            if (_ownsKernel) _kernel.Dispose();
            _kernel = weights.WScConv;
        }
    }

    /// <summary>
    /// Executes the short-conv branch: gate, conv, gate, project.
    /// <paramref name="cache"/> carries the layer's rolling conv history; when null
    /// (training, unit tests, or any cache-less forward) a private fallback buffer
    /// is used so the layer still progresses step to step.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> x, ShortConvCache? cache, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();

        int batch = 1;
        int seq = x.Shape.Rows;
        if (x.Rank == 3)
        {
            batch = x.Shape.Dims[0];
            seq = x.Shape.Dims[1];
        }
        int hidden = _config.HiddenDim;
        int rows = batch * seq;

        // Fan out to [batch, seq, 3H] (in-projection applies the transposed matmul).
        using var projected = _wIn.Forward(x, workspace);

        using var bx = RentOrNew(batch, seq, hidden, x.Rank, workspace);
        ShortConvKernels.ComputeGatedInput(projected, bx, rows, hidden);

        var state = cache?.State ?? _fallbackState;
        if (state.Shape.Dims[0] < batch)
            throw new InvalidOperationException(
                $"ShortConvCache holds {state.Shape.Dims[0]} sequence(s) but this forward sees {batch}.");

        using var convOut = RentOrNew(batch, seq, hidden, x.Rank, workspace);
        ShortConvKernels.ApplyConv(bx, state, _kernel, convOut, batch, seq, hidden, KernelRows);
        ShortConvKernels.ApplyOutputGate(projected, convOut, rows, hidden);

        // Roll the history forward: the last (l_cache - 1) gated rows are the
        // previous state for the next step.
        ShortConvKernels.UpdateState(bx, state, batch, seq, hidden, StateRows);

        return _wOut.Forward(convOut, workspace);
    }

    private static Tensor<float> RentOrNew(int batch, int seq, int hidden, int rank, IWorkspace? workspace)
    {
        if (workspace is not null)
        {
            return rank == 3
                ? workspace.Rent<float>([batch, seq, hidden])
                : workspace.Rent<float>([seq, hidden]);
        }
        return rank == 3
            ? new Tensor<float>(batch, seq, hidden)
            : new Tensor<float>(seq, hidden);
    }

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in _wIn.Parameters()) yield return p;
        foreach (var p in _wOut.Parameters()) yield return p;
    }

    /// <summary>Quantum-aware training (QAT) applies to the two projections.</summary>
    public void EnableQuantAwareTraining(QuantDType? target)
    {
        _wIn.EnableQuantAwareTraining(target);
        _wOut.EnableQuantAwareTraining(target);
    }

    public void FreeFloatWeights()
    {
        _wIn.FreeFloatWeight();
        _wOut.FreeFloatWeight();
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            _wIn.Dispose();
            _wOut.Dispose();
            if (_ownsKernel) _kernel.Dispose();
            _fallbackState.Dispose();
        }
    }

    ~ShortConvLayer() => Dispose(false);

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(ShortConvLayer));
}