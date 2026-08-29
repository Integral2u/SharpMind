using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public sealed class TrainingLinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor) : LinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor)
{
    private static readonly QuantizationOps _staticOps = QuantizationFactory.Create();
    private QuantDType? _qatTarget;

    // LoRA: y = x·W + scale · (x·A)·B with W frozen. A [In, r], B [r, Out].
    private Tensor<float>? _loraA;
    private Tensor<float>? _loraB;
    private float _loraScale;

    public bool HasLoRA => _loraA is not null;
    public int LoRARank => _loraA?.Shape.Cols ?? 0;
    public float LoRAScale => _loraScale;
    /// <summary>LoRA down-projection A [InFeatures, rank], or null.</summary>
    public Tensor<float>? LoRAA => _loraA;
    /// <summary>LoRA up-projection B [rank, OutFeatures], or null.</summary>
    public Tensor<float>? LoRAB => _loraB;

    /// <summary>
    /// Attaches a rank-<paramref name="rank"/> LoRA adapter. A is initialised
    /// uniformly in ±1/√InFeatures and B to zero, so the layer's output is
    /// unchanged until B is trained. The base weight is not touched and is
    /// frozen by convention: train <see cref="LoRAParameters"/>, not
    /// <see cref="LinearLayer.Parameters"/>. Replaces any existing adapter.
    /// </summary>
    public void EnableLoRA(int rank, float scale, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (rank <= 0 || rank > Math.Min(InFeatures, OutFeatures))
            throw new ArgumentOutOfRangeException(nameof(rank), $"{Name}: LoRA rank must be in 1..{Math.Min(InFeatures, OutFeatures)}, got {rank}.");
        DisableLoRA();
        _loraA = new Tensor<float>(InFeatures, rank);
        _loraB = new Tensor<float>(rank, OutFeatures);   // zero
        _loraScale = scale;
        float bound = 1f / MathF.Sqrt(InFeatures);
        var a = _loraA.Data;
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(rng.NextDouble() * 2 - 1) * bound;
    }

    /// <summary>Drops the adapter without merging it.</summary>
    public void DisableLoRA()
    {
        _loraA?.Dispose();
        _loraB?.Dispose();
        _loraA = null;
        _loraB = null;
        _loraScale = 0f;
    }

    /// <summary>Folds the adapter into the base weight (W += scale·A·B) and drops it.</summary>
    public void MergeLoRA()
    {
        if (_loraA is null || _loraB is null) return;
        int r = _loraA.Shape.Cols;
        var a = _loraA.Data; var b = _loraB.Data; var w = _weight.Data;
        for (int i = 0; i < InFeatures; i++)
        {
            var wRow = w.Slice(i * OutFeatures, OutFeatures);
            for (int k = 0; k < r; k++)
            {
                float aik = a[i * r + k] * _loraScale;
                if (aik == 0f) continue;
                var bRow = b.Slice(k * OutFeatures, OutFeatures);
                for (int o = 0; o < OutFeatures; o++)
                    wRow[o] += aik * bRow[o];
            }
        }
        DisableLoRA();
        InvalidateCache();
    }

    /// <summary>The adapter's A and B as trainable parameters (fresh wrappers each call, like <see cref="LinearLayer.Parameters"/>).</summary>
    public IEnumerable<Parameter> LoRAParameters()
    {
        if (_loraA is null || _loraB is null) yield break;
        yield return new Parameter($"{Name}.lora_A", _loraA);
        yield return new Parameter($"{Name}.lora_B", _loraB);
    }

    /// <summary>Adds scale·(x·A)·B into <paramref name="output"/> [batch, Out] for flat input [batch, In].</summary>
    private unsafe void AddLoRAInPlace(Tensor<float> flatInput, Tensor<float> output, int batchSize)
    {
        if (_loraA is null || _loraB is null) return;
        int r = _loraA.Shape.Cols;
        var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
        // The F32 matmul wants its weight as [N, K]; A and B are tiny, so transposing per call is nothing.
        using var aT = _loraA.Transpose();               // [r, In]
        using var bT = _loraB.Transpose();               // [Out, r]
        using var h = new Tensor<float>(batchSize, r);
        fn(flatInput.DataPtr, (byte*)aT.DataPtr, h.DataPtr, batchSize, InFeatures, r);
        using var delta = new Tensor<float>(batchSize, OutFeatures);
        fn(h.DataPtr, (byte*)bT.DataPtr, delta.DataPtr, batchSize, r, OutFeatures);
        var o = output.Data; var d = delta.Data; float s = _loraScale;
        for (int i = 0; i < o.Length; i++)
            o[i] += s * d[i];
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

    private unsafe Tensor<float> MatMulForward(Tensor<float> input, int batchSize, IWorkspace? workspace = null)
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

    public override Tensor<float> Forward(Tensor<float> input, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        var output = MatMulForward(flat, batchSize, workspace);
        AddLoRAInPlace(flat, output, batchSize);

        if (_bias is not null)
            AddBiasInPlace(output, batchSize);
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

    protected override void OnDispose() => DisableLoRA();

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
            AddBiasInPlace(output, batchSize);
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
