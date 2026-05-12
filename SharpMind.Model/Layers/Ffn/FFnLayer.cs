using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
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
    protected readonly TensorOps Ops;
    private bool _disposed;

    // ── Dense weights ─────────────────────────────────────────────────────
    protected readonly LinearLayer? W1;   // [FfnDim,    HiddenDim]
    protected readonly LinearLayer? W2;   // [HiddenDim, FfnDim]

    // ── Gated weights ─────────────────────────────────────────────────────
    protected readonly LinearLayer? WGate;  // [FfnDim,    HiddenDim]
    protected readonly LinearLayer? WUp;    // [FfnDim,    HiddenDim]
    protected readonly LinearLayer? WDown;  // [HiddenDim, FfnDim]

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        var lower = name.ToLower();
        bool isBias = lower.Contains("bias");

        if (lower.Contains("gate"))
        {
            if (isBias) WGate!.LoadBias(data); else WGate!.LoadWeightTransposed(data);
        }
        else if (lower.Contains("up"))
        {
            if (isBias) WUp!.LoadBias(data); else WUp!.LoadWeightTransposed(data);
        }
        else if (lower.Contains("down"))
        {
            if (isBias) WDown!.LoadBias(data); else WDown!.LoadWeightTransposed(data);
        }
    }

    // ── MoE weights ───────────────────────────────────────────────────────
    protected readonly LinearLayer? Router;     // [NumExperts, HiddenDim]
    protected readonly LinearLayer[]? ExpertGate;
    protected readonly LinearLayer[]? ExpertUp;
    protected readonly LinearLayer[]? ExpertDown;

    protected FfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops, FfnKind kind)
    {
        Config = config;
        Acts = acts;
        Ops = ops;

        switch (kind)
        {
            case FfnKind.Dense:
                W1 = new LinearLayer("gate_proj", config.HiddenDim, config.FfnDim, bias: true);
                W2 = new LinearLayer("down_proj", config.FfnDim, config.HiddenDim, bias: true);
                break;

            case FfnKind.Gated:
                WGate = new LinearLayer("gate_proj", config.HiddenDim, config.FfnDim, bias: true);
                WUp = new LinearLayer("up_proj", config.HiddenDim, config.FfnDim, bias: true);
                WDown = new LinearLayer("down_proj", config.FfnDim, config.HiddenDim, bias: true);
                break;

            case FfnKind.MoE:
                Router = new LinearLayer("router", config.HiddenDim, config.NumExperts, bias: true);
                ExpertGate = [.. Enumerable.Range(0, config.NumExperts).Select(i => new LinearLayer($"expert_{i}_gate_proj", config.HiddenDim, config.FfnDim, bias: true))];
                ExpertUp = [.. Enumerable.Range(0, config.NumExperts).Select(i => new LinearLayer($"expert_{i}_up_proj", config.HiddenDim, config.FfnDim, bias: true))];
                ExpertDown = [.. Enumerable.Range(0, config.NumExperts).Select(i => new LinearLayer($"expert_{i}_down_proj", config.FfnDim, config.HiddenDim, bias: true))];
                break;
        }
    }

    // ── PuzzleCornerPieces ────────────────────────────────────────────────

    [PuzzleCornerPiece(SharpMindConfig.KeyFfn,
        SharpMindConfig.ValFfnDense, NS + "." + nameof(FfnKernels.Dense),
        SharpMindConfig.ValFfnGated, NS + "." + nameof(FfnKernels.Gated),
        SharpMindConfig.ValFfnMoE, NS + "." + nameof(FfnKernels.MoE))]
    public abstract Tensor<float> ApplyFfn(Tensor<float> x);

    // ── Forward ───────────────────────────────────────────────────────

    public Tensor<float> Forward(Tensor<float> x)
    {
        ThrowIfDisposed();
        return ApplyFfn(x);
    }

    // ── Forward + State (for training) ─────────────────────────────────

    public abstract (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x);

    public abstract Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state);

    // ── Parameters & Disposal ───────────────────────────────────────────

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
            WGate?.Dispose(); WUp?.Dispose(); WDown?.Dispose();
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
        if (WGate is not null) foreach (var p in WGate.Parameters()) yield return p;
        if (WUp is not null) foreach (var p in WUp.Parameters()) yield return p;
        if (WDown is not null) foreach (var p in WDown.Parameters()) yield return p;
        if (Router is not null) foreach (var p in Router.Parameters()) yield return p;
        if (ExpertGate is not null) foreach (var l in ExpertGate) foreach (var p in l.Parameters()) yield return p;
        if (ExpertUp is not null) foreach (var l in ExpertUp) foreach (var p in l.Parameters()) yield return p;
        if (ExpertDown is not null) foreach (var l in ExpertDown) foreach (var p in l.Parameters()) yield return p;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(FfnLayer));
}

/// <summary>Training state saved during FFN forward pass.</summary>
public class FfnLayerState
{
    public Tensor<float> Input { get; init; } = null!;
    public Tensor<float> Output { get; init; } = null!;
    public FfnKind Kind { get; init; }
}

/// <summary>Training state for Dense FFN.</summary>
public sealed class DenseFfnState : FfnLayerState
{
    public LinearLayerState? W1State { get; init; }
    public LinearLayerState? W2State { get; init; }
}

/// <summary>Training state for Gated FFN.</summary>
public sealed class GatedFfnState : FfnLayerState
{
    public LinearLayerState? WGateState { get; init; }
    public LinearLayerState? WUpState { get; init; }
    public LinearLayerState? WDownState { get; init; }
    public Tensor<float>? Activated { get; init; }
}

