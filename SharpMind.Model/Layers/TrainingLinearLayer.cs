//Can be upgraded to JigSawDotNet Pattern.  Would remove QuantizationOps and calls to fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public sealed class TrainingLinearLayer : LinearLayer
{
    private static readonly QuantizationOps _staticOps = QuantizationFactory.Create();
    private Tensor<float>? _weightBT;

    public TrainingLinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor)
        : base(name, inFeatures, outFeatures, bias, weight, biasTensor)
    {
    }

    private unsafe Tensor<float> MatMulForward(Tensor<float> input, int batchSize, Workspace? workspace = null)
    {
        Tensor<float> output;
        _weightBT ??= _weight.Transpose();
        if (workspace != null)
            output = workspace.Rent<float>([batchSize, OutFeatures]);
        else
            output = new Tensor<float>(batchSize, OutFeatures);
        var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
        fn(input.DataPtr, (byte*)_weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        return output;
    }

    public override Tensor<float> Forward(Tensor<float> input, Workspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        var output = MatMulForward(flat, batchSize, workspace);

        if (_bias is not null)
        {
            if (workspace != null)
            {
                var biasB = workspace.Rent<float>([batchSize, OutFeatures]);
                for (int i = 0; i < batchSize; i++)
                    _bias!.Data.CopyTo(biasB.RowSpan(i));
                output.AddInPlace(biasB);
            }
            else
            {
                output.AddInPlace(BroadcastBias(batchSize));
            }
        }
        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return reshaped;
        }
        return output;
    }

    public override unsafe (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        using var weightBT = _weight.Transpose();
        var output = new Tensor<float>(batchSize, OutFeatures);
        var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
        fn(flat.DataPtr, (byte*)weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        if (_bias is not null)
            output.AddInPlace(BroadcastBias(batchSize));
        var state = new LinearLayerState(input, flat, needReshape, _weight);
        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return (reshaped, state);
        }
        return (output, state);
    }

    public override unsafe Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state)
    {
        int batchSize = state.NeedReshape
            ? gradOutput.ElementCount / OutFeatures
            : gradOutput.Shape[^2];
        var flatGradOut = state.NeedReshape
            ? gradOutput.Reshape(batchSize, OutFeatures)
            : gradOutput;

        var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
        var gradInputFlat = new Tensor<float>(batchSize, InFeatures);
        fn(flatGradOut.DataPtr, (byte*)_weight.DataPtr, gradInputFlat.DataPtr, batchSize, OutFeatures, InFeatures);

        using var inputT = state.Input.Transpose();
        using var flatGradOutBT = flatGradOut.Transpose();
        var dw = new Tensor<float>(InFeatures, OutFeatures);
        fn(inputT.DataPtr, (byte*)flatGradOutBT.DataPtr, dw.DataPtr, InFeatures, batchSize, OutFeatures);
        var wg = state.WeightGrad;
        for (int i = 0; i < dw.ElementCount; i++)
            wg.Data[i] += dw.Data[i];
        dw.Dispose();
        inputT.Dispose();

        if (_bias is not null)
        {
            state.BiasGrad ??= Tensor<float>.Zeros(OutFeatures);
            for (int i = 0; i < batchSize; i++)
            {
                ReadOnlySpan<float> row = flatGradOut.RowSpan(i);
                for (int j = 0; j < OutFeatures; j++)
                    state.BiasGrad.Data[j] += row[j];
            }
        }

        if (state.NeedReshape)
        {
            flatGradOut.Dispose();
            int[] inDims = [.. state.InputDims[..^1], InFeatures];
            var reshaped = gradInputFlat.Reshape(inDims);
            gradInputFlat.Dispose();
            return reshaped;
        }
        return gradInputFlat;
    }

    protected override void InvalidateCache()
    {
        _weightBT?.Dispose();
        _weightBT = null;
    }
}
