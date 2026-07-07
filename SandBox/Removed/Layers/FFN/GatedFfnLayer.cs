using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class GatedFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops, QuantizationOps qOps, LinearLayerFactory layerFactory, TransformerWeights.BlockWeights? weights = null) : FfnLayer(config, acts, ops, FfnKind.Gated, qOps, layerFactory, weights)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
        => FfnKernels.Gated(x, WGated!, WDown!, Acts, Ops, workspace);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        if (WGated is not FloatLinearLayer fg)
            throw new InvalidOperationException("Gated FFN requires float linear layer for training.");
        var (Output, _) = fg.ForwardWithState(x, Ops);
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
        var output = WDown!.Forward(activated, Ops);
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.Gated };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        if (WDown is not FloatLinearLayer fd || WGated is not FloatLinearLayer fg)
            throw new InvalidOperationException("Backward requires float linear layers.");
        using var wDownT = TensorOps.Transpose(fd.Weight);
        var dHidden = Ops.MatMul(gradOutput, wDownT);
        int ffnDim = Config.FfnDim;
        int batchSize = dHidden.ElementCount / ffnDim;
                int hiddenDim = fg.InFeatures;

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

        using var wFusedT = TensorOps.Transpose(fg.Weight);
        var gradInput = Ops.MatMul(dFused, wFusedT);
        dFused.Dispose();
        return gradInput;
    }
}
