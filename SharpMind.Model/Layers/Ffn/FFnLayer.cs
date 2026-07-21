using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

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
    protected readonly QuantizationOps _qOps;
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
            if (isBias) WDown!.LoadBias(data); else WDown!.LoadWeightTransposed(data);
        }
    }

    private void LoadFusedWeightTransposed(ReadOnlySpan<float> data, int colOffset)
    {
        // WGated weight: [HiddenDim, 2*FfnDim]
        // GGUF: [FfnDim, HiddenDim] → SharpMind: [HiddenDim, FfnDim] starting at colOffset
        var w = WGated!.Weight;
        int hidden = Config.HiddenDim;
        int ffnDim = Config.FfnDim;
        int outStride = 2 * ffnDim;
        for (int o = 0; o < ffnDim; o++)
            for (int i = 0; i < hidden; i++)
                w.Data[i * outStride + colOffset + o] = data[o * hidden + i];
    }

    private void LoadFusedBias(ReadOnlySpan<float> data, int offset)
        => data.CopyTo(WGated!.Bias!.Data.Slice(offset, data.Length));

    public bool SetRawWeight(string name, byte[] rawData, QuantDType dtype)
    {
        if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return false;

        // MoE expert tensors: blk.{L}.ffn_gate.exps.{E}.weight → ExpertGate[E]
        if (ExpertGate is not null && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
        {
            var expMatch = RegexGenerated.ExpertIndex.Match(name);
            if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
            {
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertGate.Length)
                    { ExpertGate[expIdx].SetRawWeight(rawData); return true; }
                if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertUp!.Length)
                    { ExpertUp[expIdx].SetRawWeight(rawData); return true; }
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && expIdx < ExpertDown!.Length)
                    { ExpertDown[expIdx].SetRawWeight(rawData); return true; }
            }
            return false;
        }

        // MoE router (ffn_gate.weight without .exps.)
        if (Router is not null && name.Contains("gate", StringComparison.OrdinalIgnoreCase))
            { Router.SetRawWeight(rawData); return true; }
        if (Router is not null && name.Contains("up", StringComparison.OrdinalIgnoreCase))
            return false; // Up without .exps. shouldn't appear in MoE models; skip float load

        // Gate and up are fused into WGated with separate quantized tensors in GGUF.
        // Force dequantization so LoadWeights can load into the fused float weight.
        if (name.Contains("gate", StringComparison.OrdinalIgnoreCase) || name.Contains("up", StringComparison.OrdinalIgnoreCase))
            return false;
        if (name.Contains("down", StringComparison.OrdinalIgnoreCase))
            { WDown!.SetRawWeight(rawData); return true; }
        return false;
    }

    // MoE weights
    protected readonly LinearLayer? Router;     // [NumExperts, HiddenDim]
    protected readonly LinearLayer[]? ExpertGate;
    protected readonly LinearLayer[]? ExpertUp;
    protected readonly LinearLayer[]? ExpertDown;

    protected FfnLayer(ModelConfig config, ActivationOps acts, FfnKind kind, QuantizationOps qOps)
        : this(config, acts, kind, qOps, null, null)
    {
    }

    protected FfnLayer(ModelConfig config, ActivationOps acts, FfnKind kind, QuantizationOps qOps, TransformerWeights.BlockWeights? weights, Dictionary<string, string>? mapping = null)
    {
        Config = config;
        Acts = acts;
        _qOps = qOps;

        var tm = weights?.TensorMeta;

        switch (kind)
        {
            case FfnKind.Dense:
                W1 = LinearLayerFactory.Create("gate_proj", config.HiddenDim, config.FfnDim, true,
                    weights?.Wf1, weights?.Wf1Bias, tm?.GetValueOrDefault("RawWf1").Dtype ?? QuantDType.F32, mapping);
                W2 = LinearLayerFactory.Create("down_proj", config.FfnDim, config.HiddenDim, true,
                    weights?.Wf2, weights?.Wf2Bias, tm?.GetValueOrDefault("RawWf2").Dtype ?? QuantDType.F32, mapping);
                break;

            case FfnKind.Gated:
                WGated = LinearLayerFactory.Create("wgated_proj", config.HiddenDim, 2 * config.FfnDim, true,
                    weights?.Wf1, weights?.Wf1Bias, tm?.GetValueOrDefault("RawWgate").Dtype ?? QuantDType.F32, mapping);
                WDown = LinearLayerFactory.Create("down_proj", config.FfnDim, config.HiddenDim, true,
                    weights?.Wf2, weights?.Wf2Bias, tm?.GetValueOrDefault("RawWf2").Dtype ?? QuantDType.F32, mapping);
                break;

            case FfnKind.MoE:
                Router = LinearLayerFactory.Create("router", config.HiddenDim, config.NumExperts, true,
                    null, null, tm?.GetValueOrDefault("RawRouter").Dtype ?? QuantDType.F32, mapping);
                ExpertGate = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    LinearLayerFactory.Create($"expert_{i}_gate_proj", config.HiddenDim, config.FfnDim, true,
                        null, null, tm?.GetValueOrDefault($"RawWgateExp_{i}").Dtype ?? QuantDType.F32, mapping))];
                ExpertUp = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    LinearLayerFactory.Create($"expert_{i}_up_proj", config.HiddenDim, config.FfnDim, true,
                        null, null, tm?.GetValueOrDefault($"RawWupExp_{i}").Dtype ?? QuantDType.F32, mapping))];
                ExpertDown = [.. Enumerable.Range(0, config.NumExperts).Select(i =>
                    LinearLayerFactory.Create($"expert_{i}_down_proj", config.FfnDim, config.HiddenDim, true,
                        null, null, tm?.GetValueOrDefault($"RawWdownExp_{i}").Dtype ?? QuantDType.F32, mapping))];
                break;
        }
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        if (W1 is not null && W2 is not null)
        {
            if (weights.Wf1 != null) W1.ReplaceWeights(weights.Wf1, weights.Wf1Bias);
            if (weights.Wf2 != null) W2.ReplaceWeights(weights.Wf2, weights.Wf2Bias);
            W1.SetRawWeight(weights.RawWf1);
            W2.SetRawWeight(weights.RawWf2);
        }
        else if (WGated is not null && WDown is not null)
        {
            if (weights.Wf1 != null) WGated.ReplaceWeights(weights.Wf1, weights.Wf1Bias);
            if (weights.RawWgate != null && weights.RawWup != null)
            {
                byte[] fused = new byte[weights.RawWgate.Length + weights.RawWup.Length];
                Buffer.BlockCopy(weights.RawWgate, 0, fused, 0, weights.RawWgate.Length);
                Buffer.BlockCopy(weights.RawWup, 0, fused, weights.RawWgate.Length, weights.RawWup.Length);
                WGated.SetRawWeight(fused);
            }
            if (weights.Wf2 != null) WDown.ReplaceWeights(weights.Wf2, weights.Wf2Bias);
            WDown.SetRawWeight(weights.RawWf2);
        }
        else if (Router is not null && ExpertGate is not null)
        {
            // MoE: push per-expert raw data and router raw data
            if (weights.RawWgateExp is not null)
                foreach (var (expIdx, rawData) in weights.RawWgateExp)
                    if (expIdx < ExpertGate.Length)
                        ExpertGate[expIdx].SetRawWeight(rawData);
            if (weights.RawWupExp is not null)
                foreach (var (expIdx, rawData) in weights.RawWupExp)
                    if (expIdx < ExpertUp!.Length)
                        ExpertUp[expIdx].SetRawWeight(rawData);
            if (weights.RawWdownExp is not null)
                foreach (var (expIdx, rawData) in weights.RawWdownExp)
                    if (expIdx < ExpertDown!.Length)
                        ExpertDown[expIdx].SetRawWeight(rawData);
            Router.SetRawWeight(weights.RawRouter);
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
        if (W1 is not null && W2 is not null)
        {
            W1.FreeFloatWeight();
            W2.FreeFloatWeight();
        }
        else if (WGated is not null && WDown is not null)
        {
            WGated.FreeFloatWeight();
            WDown.FreeFloatWeight();
        }
        Router?.FreeFloatWeight();
        if (ExpertGate is not null)
            foreach (var l in ExpertGate) l.FreeFloatWeight();
        if (ExpertUp is not null)
            foreach (var l in ExpertUp) l.FreeFloatWeight();
        if (ExpertDown is not null)
            foreach (var l in ExpertDown) l.FreeFloatWeight();
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
            W1?.Dispose(); W2?.Dispose();
            WGated?.Dispose(); WDown?.Dispose();
            Router?.Dispose();
            if (ExpertGate is not null) foreach (var l in ExpertGate) l.Dispose();
            if (ExpertUp is not null) foreach (var l in ExpertUp) l.Dispose();
            if (ExpertDown is not null) foreach (var l in ExpertDown) l.Dispose();
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