using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
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
                W1 = new LinearLayer(config.HiddenDim, config.FfnDim);
                W2 = new LinearLayer(config.FfnDim, config.HiddenDim);
                break;

            case FfnKind.Gated:
                WGate = new LinearLayer(config.HiddenDim, config.FfnDim);
                WUp = new LinearLayer(config.HiddenDim, config.FfnDim);
                WDown = new LinearLayer(config.FfnDim, config.HiddenDim);
                break;

            case FfnKind.MoE:
                Router = new LinearLayer(config.HiddenDim, config.NumExperts);
                ExpertGate = [.. Enumerable.Range(0, config.NumExperts).Select(_ => new LinearLayer(config.HiddenDim, config.FfnDim))];
                ExpertUp = [.. Enumerable.Range(0, config.NumExperts).Select(_ => new LinearLayer(config.HiddenDim, config.FfnDim))];
                ExpertDown = [.. Enumerable.Range(0, config.NumExperts).Select(_ => new LinearLayer(config.FfnDim, config.HiddenDim))];
                break;
        }
    }

    // ── PuzzleCornerPieces ────────────────────────────────────────────────

    [PuzzleCornerPiece(SharpMindConfig.KeyFfn,
        SharpMindConfig.ValFfnDense, NS + "." + nameof(FfnKernels.Dense),
        SharpMindConfig.ValFfnGated, NS + "." + nameof(FfnKernels.Gated),
        SharpMindConfig.ValFfnMoE, NS + "." + nameof(FfnKernels.MoE))]
    public abstract Tensor<float> ApplyFfn(Tensor<float> x);

    // ── Public API ────────────────────────────────────────────────────────

    public Tensor<float> Forward(Tensor<float> x)
    {
        ThrowIfDisposed();
        return ApplyFfn(x);
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
            WGate?.Dispose(); WUp?.Dispose(); WDown?.Dispose();
            Router?.Dispose();
            if (ExpertGate is not null) foreach (var l in ExpertGate) l.Dispose();
            if (ExpertUp is not null) foreach (var l in ExpertUp) l.Dispose();
            if (ExpertDown is not null) foreach (var l in ExpertDown) l.Dispose();
        }
        _disposed = true;
    }

    ~FfnLayer() => Dispose(false);

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(FfnLayer));
}

