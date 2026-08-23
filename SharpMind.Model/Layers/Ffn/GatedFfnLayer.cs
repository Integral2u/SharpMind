using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class GatedFfnLayer(ModelConfig config, ActivationOps acts, QuantizationOps qOps, TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null) : FfnLayer(config, acts, FfnKind.Gated, qOps, weights, mapping)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.IWorkspace? workspace = null)
        => FfnKernels.Gated(x, WGated!, WDown!, Acts, workspace);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var (Output, _) = WGated!.ForwardWithState(x);
        using var gateUp = Output;
        int ffnDim = Config.FfnDim;
        int total = gateUp.ElementCount / (2 * ffnDim);
        var flat = gateUp.Reshape(total, 2 * ffnDim);
        bool hasBatch = gateUp.Rank > 2;
        using var activated = hasBatch
            ? Tensor<float>.Zeros(gateUp.Shape.Dims[0], gateUp.Shape.Dims[1], ffnDim)
            : Tensor<float>.Zeros(total, ffnDim);
        for (int i = 0; i < total; i++)
        {
            var row = flat.RowSpan(i);
            Acts.ApplyGate(row[..ffnDim], row[ffnDim..], activated.RowSpan(i));
        }
        var output = WDown!.Forward(activated);
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.Gated };
        return (output, state);
    }

    public override unsafe Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);

        using var wDownT = WDown!.Weight.Transpose();
        var dHidden = new Tensor<float>(gradOutput.Shape.Rows, WDown.InFeatures);
        fn(gradOutput.DataPtr, (byte*)wDownT.DataPtr, dHidden.DataPtr, gradOutput.Shape.Rows, gradOutput.Shape.Cols, WDown.InFeatures);

        int ffnDim = Config.FfnDim;
        int batchSize = dHidden.ElementCount / ffnDim;

        // Fused backward: reconstruct [B, 2*fD] gradient from [B, fD] gradient.
        // Gate and up have the same gradient (dHidden) in this simplified version.
        int fusedDim = 2 * ffnDim;
        var dFused = Tensor<float>.Zeros(batchSize, fusedDim);
        for (int i = 0; i < batchSize; i++)
        {
            var row = dFused.RowSpan(i);
            dHidden.RowSpan(i).CopyTo(row[..ffnDim]);
            dHidden.RowSpan(i).CopyTo(row[ffnDim..]);
        }
        dHidden.Dispose();

        using var wFusedT = WGated!.Weight.Transpose();
        var gradInput = new Tensor<float>(batchSize, WGated.InFeatures);
        fn(dFused.DataPtr, (byte*)wFusedT.DataPtr, gradInput.DataPtr, batchSize, fusedDim, WGated.InFeatures);
        dFused.Dispose();
        return gradInput;
    }
}
