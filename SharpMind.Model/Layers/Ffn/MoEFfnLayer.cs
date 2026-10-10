using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class MoEFfnLayer(ModelConfig config, ActivationOps acts, QuantizationOps qOps, TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null) : FfnLayer(config, acts, FfnKind.MoE, qOps, weights, mapping)
{
    /// <summary>
    /// Streaming expert-residency hook, wired by the streaming weights when
    /// <c>DesiredResidentExperts &gt; 0</c>. The forward precomputes the batch's top-k union,
    /// loads any of those experts that are not resident, and evaluates them; cold experts are
    /// streamed per token instead of living in the resident window.
    /// </summary>
    internal IExpertResidencyHost? ExpertResidency { get; set; }

    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        var routed = FfnKernels.MoE(x, Router!, ExpertGate!, ExpertUp!, ExpertDown!, Config.TopKExperts, Acts, workspace, Config.NormTopKProb, ExpertResidency);

        if (SharedGate is null || SharedUp is null || SharedDown is null)
            return routed;

        // Qwen1.5-MoE: the always-on shared expert runs on every token, and its output
        // is scaled by sigmoid(SharedGateInp · x) per token before being added to the
        // top-k routed result. SharedGateInp holds the gate's single [1, HiddenDim]
        // row (GGUF rank-1 [HiddenDim]); when the file omits it the gate is 1.0.
        using var shared = FfnKernels.Gated(x, SharedGate, SharedUp, SharedDown, Acts, workspace);

        if (shared.ElementCount != routed.ElementCount)
            throw new InvalidOperationException(
                $"Shared expert produced {shared.ElementCount} elements but the routed MoE produced " +
                $"{routed.ElementCount}; the shared and routed FFN widths disagree.");

        int hidden = x.Shape[^1];
        int batch = routed.ElementCount / hidden;

        if (SharedGateInp is null)
        {
            AddScaledInPlace(routed, shared.Data, null);
            return routed;
        }

        // The gate projects to a single value per token, so a rank-3 input has to be
        // flattened to [tokens, hidden] first — one gate logit per row. The reshape is
        // a view over x's memory, so it is the *view* that gets disposed: assigning x
        // itself and letting `using` own it would free the caller's tensor.
        using var gateInput = x.Rank > 2 ? x.Reshape(batch, hidden) : null;
        using var gateLogit = SharedGateInp.Forward(gateInput ?? x, workspace);

        var scales = new float[batch];
        for (int t = 0; t < batch; t++)
            scales[t] = MathEx.Sigmoid(gateLogit.Data[t]);

        AddScaledInPlace(routed, shared.Data, scales);
        return routed;
    }

    /// <summary>
    /// Adds the shared branch into the routed result in place, scaling each token's
    /// row by its own gate (or by 1.0 when <paramref name="scales"/> is null).
    /// </summary>
    private static void AddScaledInPlace(Tensor<float> target, ReadOnlySpan<float> shared, float[]? scales)
    {
        var dst = target.Data;

        if (scales is null)
        {
            for (int i = 0; i < dst.Length && i < shared.Length; i++)
                dst[i] += shared[i];
            return;
        }

        int hidden = dst.Length / scales.Length;
        for (int t = 0; t < scales.Length; t++)
        {
            float g = scales[t];
            int rowStart = t * hidden;
            int rowEnd = Math.Min(rowStart + hidden, dst.Length);
            for (int i = rowStart; i < rowEnd && i < shared.Length; i++)
                dst[i] += shared[i] * g;
        }
    }

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var output = ApplyFfn(x);
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.MoE };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state) => gradOutput; // Stub - MoE backward is complex
}