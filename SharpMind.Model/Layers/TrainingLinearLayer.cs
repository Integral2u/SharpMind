//Can be upgraded to JigSawDotNet Pattern.  Would remove QuantizationOps and calls to fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public sealed class TrainingLinearLayer : LinearLayer
{
    private static readonly QuantizationOps _staticOps = QuantizationFactory.Create();
    private QuantDType? _qatTarget;

    public TrainingLinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor)
        : base(name, inFeatures, outFeatures, bias, weight, biasTensor)
    {
    }

    /// <summary>
    /// Enables quantization-aware training for this layer. The master weight
    /// stays F32; each forward pass quantizes a transposed copy of the weight to
    /// <paramref name="target"/> and runs the matching quantized matmul, so the
    /// forward sees quantized weights while backward gradients flow straight
    /// through to the master weight. Null or <see cref="QuantDType.F32"/>
    /// restores the pure-float forward. Block formats (Q8_0/Q4_0) require both
    /// InFeatures and OutFeatures to be multiples of 32; K-quant formats
    /// (Q2_K..Q8_K) require the flattened weight length to be a multiple of 256
    /// and InFeatures (the column width seen by the K-quant VecDot kernels,
    /// which address sub-scales per 128-element half-block) to be a multiple of 128.
    /// </summary>
    public override void EnableQuantAwareTraining(QuantDType? target)
    {
        if (target is QuantDType.Q8_0 or QuantDType.Q4_0 &&
            (InFeatures % 32 != 0 || OutFeatures % 32 != 0))
            throw new InvalidOperationException(
                $"{Name}: QAT with {target} requires every dimension to be a multiple of 32 " +
                $"(got {InFeatures}x{OutFeatures}). Use F16 or disable QAT for this layer.");
        if (IsKQuant(target) &&
            (InFeatures % 128 != 0 || ((long)InFeatures * OutFeatures) % 256 != 0))
            throw new InvalidOperationException(
                $"{Name}: QAT with {target} requires InFeatures to be a multiple of 128 and the " +
                $"flattened weight length ({InFeatures}x{OutFeatures} = {InFeatures * OutFeatures}) " +
                "to be a multiple of 256. Use F16 or disable QAT for this layer.");
        _qatTarget = target;
    }

    /// <summary>True when <paramref name="target"/> is a K-quant block format (Q2_K..Q8_K).</summary>
    private static bool IsKQuant(QuantDType? target) => target is
        QuantDType.Q2_K or QuantDType.Q2_K_S
        or QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L
        or QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M
        or QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M
        or QuantDType.Q6_K or QuantDType.Q6_K_S
        or QuantDType.Q8_K;

    public bool QuantAwareEnabled => _qatTarget is not null and not QuantDType.F32;
    public QuantDType? QuantAwareTarget => _qatTarget;

private unsafe Tensor<float> MatMulForward(Tensor<float> input, int batchSize, Workspace? workspace = null)
    {
        Tensor<float> output;
        // Local, not a field: MoE calls Forward on one shared expert layer from
        // several Parallel.For threads at once, so a per-instance transpose gets
        // disposed under another thread's matmul.
        using var weightBT = _weight.Transpose();
        if (workspace != null)
            output = workspace.Rent<float>([batchSize, OutFeatures]);
        else
            output = new Tensor<float>(batchSize, OutFeatures);
        if (_qatTarget is null or QuantDType.F32)
        {
            var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
            fn(input.DataPtr, (byte*)weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        }
        else
        {
            var fn = _staticOps.QuantizedMatMulOpFor(_qatTarget.Value);
            var raw = TensorQuantizer.Quantize(weightBT.Data, [weightBT.Shape.Rows, weightBT.Shape.Cols], _qatTarget.Value);
            fixed (byte* rawPtr = raw)
                fn(input.DataPtr, rawPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        }
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
                var biasBroadcast = BroadcastBias(batchSize);
                output.AddInPlace(biasBroadcast);
                biasBroadcast.Dispose();
            }
        }
        if (needReshape)
        {
            Span<int> outDims = stackalloc int[input.Rank];
            input.Shape.Dims[..^1].CopyTo(outDims);
            outDims[^1] = OutFeatures;
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
        if (_qatTarget is null or QuantDType.F32)
        {
            var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
            fn(flat.DataPtr, (byte*)weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        }
        else
        {
            var fn = _staticOps.QuantizedMatMulOpFor(_qatTarget.Value);
            var raw = TensorQuantizer.Quantize(weightBT.Data, [weightBT.Shape.Rows, weightBT.Shape.Cols], _qatTarget.Value);
            fixed (byte* rawPtr = raw)
                fn(flat.DataPtr, rawPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        }
        if (_bias is not null)
            output.AddInPlace(BroadcastBias(batchSize));
        var state = new LinearLayerState(input, flat, needReshape, _weight);
        if (needReshape)
        {
            Span<int> outDims = stackalloc int[input.Rank];
            input.Shape.Dims[..^1].CopyTo(outDims);
            outDims[^1] = OutFeatures;
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
            int rank = state.InputDims.Length;
            Span<int> inDims = stackalloc int[rank];
            state.InputDims.AsSpan(0, rank - 1).CopyTo(inDims);
            inDims[^1] = InFeatures;
            var reshaped = gradInputFlat.Reshape(inDims);
            gradInputFlat.Dispose();
            return reshaped;
        }
        return gradInputFlat;
    }
}
