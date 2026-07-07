using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Model.Layers.Ffn;

/// <summary>
/// FFN layer assembled by JigSawDotNet.
/// The "ffn" mapping key selects the implementation:
///   "dense"  → standard 2-layer FFN (GPT-2 style)
///   "gated"  → SwiGLU / GeGLU gated FFN (LLaMA style)
///   "moe"    → Mixture of Experts with top-k routing
/// </summary>
public abstract class FfnLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(Ffn)}.{nameof(FfnKernels)}";

    protected readonly ModelConfig Config;
    protected readonly ActivationOps Acts;
    protected readonly TensorOps Ops;
    private bool _disposed;

    // Dense weights
    protected readonly LinearLayer? W1;   // [FfnDim,    HiddenDim]
    protected readonly LinearLayer? W2;   // [HiddenDim, FfnDim]

    // Gated weights
    public readonly LinearLayer? WGated; // [HiddenDim, 2*FfnDim] — fused gate+up
    public readonly LinearLayer? WDown;  // [HiddenDim, FfnDim]

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        bool isBias = name.Contains("bias", StringComparison.OrdinalIgnoreCase);

        if (name.Contains("gate", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias)
                LoadFusedBias(data, 0);
            else
                LoadFusedWeightTransposed(data, 0);
        }
        else if (name.Contains("up", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias)
                LoadFusedBias(data, Config.FfnDim);
            else
                LoadFusedWeightTransposed(data, Config.FfnDim);
        }
        else if (name.Contains("down", StringComparison.OrdinalIgnoreCase))
        {
            if (WDown is FloatLinearLayer fd)
            {
                if (isBias) fd.LoadBias(data); else fd.LoadWeightTransposed(data);
            }
        }
    }

    private unsafe void LoadFusedWeightTransposed(ReadOnlySpan<float> data, int colOffset)
    {
        // WGated weight: [HiddenDim, 2*FfnDim]
        // GGUF: [FfnDim, HiddenDim] → SharpMind: [HiddenDim, FfnDim] starting at colOffset
        if (WGated is not FloatLinearLayer fg) return;
        var w = fg.Weight;
        int hidden = Config.HiddenDim;
        int ffnDim = Config.FfnDim;
        int outStride = 2 * ffnDim;
        for (int o = 0; o < ffnDim; o++)
            for (int i = 0; i < hidden; i++)
                w.Data[i * outStride + colOffset + o] = data[o * hidden + i];
    }

    private void LoadFusedBias(ReadOnlySpan<float> data, int offset)
    {
        if (WGated is FloatLinearLayer fg && fg.BiasTensor is not null)
            data.CopyTo(fg.BiasTensor.Data.Slice(offset, data.Length));
    }

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return false;

        // MoE expert tensors: blk.{L}.ffn_gate.exps.{E}.weight → ExpertGate[E]
        if (ExpertGate is not null && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
        {
            var expMatch = RegexGenerated.ExpertIndex.Match(name);
            if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
            {
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertGate.Length)
                    return TryUpdateRaw(ExpertGate[expIdx], rawData, dtype);
                if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertUp!.Length)
                    return TryUpdateRaw(ExpertUp[expIdx], rawData, dtype);
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertDown!.Length)
                    return TryUpdateRaw(ExpertDown[expIdx], rawData, dtype);
            }
            return false;
        }

        // MoE router (ffn_gate.weight without .exps.)
        if (Router is not null && name.Contains("gate", StringComparison.OrdinalIgnoreCase))
            return TryUpdateRaw(Router, rawData, dtype);
        if (Router is not null && name.Contains("up", StringComparison.OrdinalIgnoreCase))
            return false;

        // Gate and up are fused into WGated with separate quantized tensors in GGUF.
        if (name.Contains("gate", StringComparison.OrdinalIgnoreCase) || name.Contains("up", StringComparison.OrdinalIgnoreCase))
            return false;
        if (name.Contains("down", StringComparison.OrdinalIgnoreCase))
            return TryUpdateRaw(WDown!, rawData, dtype);
        return false;
    }

    private static bool TryUpdateRaw(LinearLayer layer, byte[] rawData, GgufDtype dtype)
    {
        if (layer is QuantizedLinearLayer ql)
        {
            ql.UpdateRawData(rawData, dtype);
            return true;
        }
        return false;
    }

    // MoE weights
    protected readonly LinearLayer? Router;     // [NumExperts, HiddenDim]
    protected readonly LinearLayer[]? ExpertGate;
    protected readonly LinearLayer[]? ExpertUp;
    protected readonly LinearLayer[]? ExpertDown;

    protected FfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops, FfnKind kind, QuantizationOps qOps, LinearLayerFactory layerFactory)
        : this(config, acts, ops, kind, qOps, layerFactory, null)
    {
    }

    protected FfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops, FfnKind kind, QuantizationOps qOps, LinearLayerFactory layerFactory, TransformerWeights.BlockWeights? weights)
    {
        Config = config;
        Acts = acts;
        Ops = ops;

        switch (kind)
        {
            case FfnKind.Dense:
                W1 = layerFactory.Create("gate_proj", config.HiddenDim, config.FfnDim, bias: true, qOps,
                    weights?.Wf1, weights?.Wf1Bias, weights?.RawWf1, weights?.QuantDtypeWf1);
                W2 = layerFactory.Create("down_proj", config.FfnDim, config.HiddenDim, bias: true, qOps,
                    weights?.Wf2, weights?.Wf2Bias, weights?.RawWf2, weights?.QuantDtypeWf2);
                break;

            case FfnKind.Gated:
                // WGated always uses float path (gate+up are fused from separate GGUF tensors)
                WGated = new FloatLinearLayer("wgated_proj", config.HiddenDim, 2 * config.FfnDim, bias: true,
                    weights?.Wf1, weights?.Wf1Bias);
                WDown = layerFactory.Create("down_proj", config.FfnDim, config.HiddenDim, bias: true, qOps,
                    weights?.Wf2, weights?.Wf2Bias, weights?.RawWf2, weights?.QuantDtypeWf2);
                break;

            case FfnKind.MoE:
                Router = layerFactory.Create("router", config.HiddenDim, config.NumExperts, bias: true, qOps,
                    null, null, weights?.RawRouter, weights?.QuantDtypeRouter);
                ExpertGate = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    layerFactory.Create($"expert_{i}_gate_proj", config.HiddenDim, config.FfnDim, bias: true, qOps,
                        null, null,
                        weights?.RawWgateExp?.GetValueOrDefault(i),
                        weights?.QuantDtypeWgateExp?.GetValueOrDefault(i)))];
                ExpertUp = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    layerFactory.Create($"expert_{i}_up_proj", config.HiddenDim, config.FfnDim, bias: true, qOps,
                        null, null,
                        weights?.RawWupExp?.GetValueOrDefault(i),
                        weights?.QuantDtypeWupExp?.GetValueOrDefault(i)))];
                ExpertDown = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    layerFactory.Create($"expert_{i}_down_proj", config.FfnDim, config.HiddenDim, bias: true, qOps,
                        null, null,
                        weights?.RawWdownExp?.GetValueOrDefault(i),
                        weights?.QuantDtypeWdownExp?.GetValueOrDefault(i)))];
                break;
        }
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        if (W1 is not null && W2 is not null)
        {
            if (W1 is FloatLinearLayer f1) f1.ReplaceWeights(weights.Wf1, weights.Wf1Bias);
            if (W2 is FloatLinearLayer f2) f2.ReplaceWeights(weights.Wf2, weights.Wf2Bias);
        }
        else if (WGated is not null && WDown is not null)
        {
            if (WGated is FloatLinearLayer fg)
                fg.ReplaceWeights(weights.Wf1, weights.Wf1Bias);
            if (weights.RawWgate != null && weights.RawWup != null && WGated is FloatLinearLayer)
            {
                var fusedDtype = weights.QuantDtypeWgate ?? GgufDtype.F32;
                byte[] fused = new byte[weights.RawWgate.Length + weights.RawWup.Length];
                Buffer.BlockCopy(weights.RawWgate, 0, fused, 0, weights.RawWgate.Length);
                Buffer.BlockCopy(weights.RawWup, 0, fused, weights.RawWgate.Length, weights.RawWup.Length);
            }
            if (WDown is FloatLinearLayer fd)
                fd.ReplaceWeights(weights.Wf2, weights.Wf2Bias);
        }
        else if (Router is not null && ExpertGate is not null)
        {
            // MoE: push per-expert raw data updates for QuantizedLinearLayer
            UpdateQuantizedRaw(ExpertGate, weights.RawWgateExp, weights.QuantDtypeWgateExp);
            UpdateQuantizedRaw(ExpertUp, weights.RawWupExp, weights.QuantDtypeWupExp);
            UpdateQuantizedRaw(ExpertDown, weights.RawWdownExp, weights.QuantDtypeWdownExp);
            if (Router is QuantizedLinearLayer qr && weights.RawRouter != null)
                qr.UpdateRawData(weights.RawRouter, weights.QuantDtypeRouter ?? GgufDtype.F32);
        }
    }

    private static void UpdateQuantizedRaw(LinearLayer[]? layers, Dictionary<int, byte[]>? rawData, Dictionary<int, GgufDtype>? dtypes)
    {
        if (layers is null || rawData is null) return;
        foreach (var (expIdx, data) in rawData)
        {
            if (expIdx < layers.Length && layers[expIdx] is QuantizedLinearLayer ql)
                ql.UpdateRawData(data, dtypes?.GetValueOrDefault(expIdx) ?? GgufDtype.F32);
        }
    }

    /// <summary>Pushes newly loaded raw quantized data into layer instances.
    /// Called by <see cref="CachedWeightLoader"/> after on-demand layer loading.
    /// Safe to call after FreeFloatWeights — only touches raw data, not float tensors.</summary>
    public void UpdateRawWeights(TransformerWeights.BlockWeights weights)
    {
        if (W1 is not null && W2 is not null)
        {
            if (W1 is QuantizedLinearLayer q1) q1.UpdateRawData(weights.RawWf1, weights.QuantDtypeWf1 ?? GgufDtype.F32);
            if (W2 is QuantizedLinearLayer q2) q2.UpdateRawData(weights.RawWf2, weights.QuantDtypeWf2 ?? GgufDtype.F32);
        }
        else if (WGated is not null && WDown is not null)
        {
            UpdateQuantizedRaw([WDown], weights.RawWf2 == null ? null : new() { [0] = weights.RawWf2 },
                weights.QuantDtypeWf2 == null ? null : new() { [0] = weights.QuantDtypeWf2.Value });
        }
        else if (Router is not null && ExpertGate is not null)
        {
            UpdateQuantizedRaw(ExpertGate, weights.RawWgateExp, weights.QuantDtypeWgateExp);
            UpdateQuantizedRaw(ExpertUp, weights.RawWupExp, weights.QuantDtypeWupExp);
            UpdateQuantizedRaw(ExpertDown, weights.RawWdownExp, weights.QuantDtypeWdownExp);
            if (Router is QuantizedLinearLayer qr && weights.RawRouter != null)
                qr.UpdateRawData(weights.RawRouter, weights.QuantDtypeRouter ?? GgufDtype.F32);
        }
    }

    // PuzzleCornerPieces

    [PuzzleCornerPiece(SharpMindConfig.KeyFfn,
        SharpMindConfig.ValFfnDense, NS + "." + nameof(FfnKernels.Dense),
        SharpMindConfig.ValFfnGated, NS + "." + nameof(FfnKernels.Gated),
        SharpMindConfig.ValFfnMoE, NS + "." + nameof(FfnKernels.MoE))]
    public abstract Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null);


    // Forward

    public Tensor<float> Forward(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        return ApplyFfn(x, workspace);
    }

    // Forward + State (for training)

    public abstract (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x);

    public abstract Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state);

    // Parameters & Disposal

    public void FreeFloatWeights()
    {
        foreach (var l in AllLayers())
            l.FreeFloatWeight();
    }

    private IEnumerable<LinearLayer> AllLayers()
    {
        if (W1 is not null) yield return W1;
        if (W2 is not null) yield return W2;
        if (WGated is not null) yield return WGated;
        if (WDown is not null) yield return WDown;
        if (Router is not null) yield return Router;
        if (ExpertGate is not null)
            foreach (var l in ExpertGate) yield return l;
        if (ExpertUp is not null)
            foreach (var l in ExpertUp) yield return l;
        if (ExpertDown is not null)
            foreach (var l in ExpertDown) yield return l;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            foreach (var l in AllLayers()) l.Dispose();
        }
        _disposed = true;
    }

    ~FfnLayer() => Dispose(false);

    public IEnumerable<Parameter> Parameters()
    {
        if (W1 is not null) foreach (var p in W1.Parameters()) yield return p;
        if (W2 is not null) foreach (var p in W2.Parameters()) yield return p;
        if (WGated is not null) foreach (var p in WGated.Parameters()) yield return p;
        if (WDown is not null) foreach (var p in WDown.Parameters()) yield return p;
        if (Router is not null) foreach (var p in Router.Parameters()) yield return p;
        if (ExpertGate is not null) foreach (var l in ExpertGate) foreach (var p in l.Parameters()) yield return p;
        if (ExpertUp is not null) foreach (var l in ExpertUp) foreach (var p in l.Parameters()) yield return p;
        if (ExpertDown is not null) foreach (var l in ExpertDown) foreach (var p in l.Parameters()) yield return p;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(FfnLayer));
}