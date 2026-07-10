using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public sealed class LinearLayer : IDisposable
{
    private Tensor<float> _weight;
    private Tensor<float>? _weightBT;
    private Tensor<float>? _bias;
    private readonly QuantizationOps _qOps;
    private bool _ownsWeight;
    private bool _ownsBias;
    private bool _disposed;

    private QuantizedMatMulFn? _matMulFn;

    public byte[]? RawQuantizedData { get; set; }
    public QuantDType? QuantDtype { get; set; }
    public bool UseQuantizedForward => RawQuantizedData != null && QuantDtype != null;

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias, QuantizationOps? qOps, Tensor<float>? weight, Tensor<float>? biasTensor)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight ?? new Tensor<float>(inFeatures, outFeatures);
        _bias = biasTensor ?? (bias ? new Tensor<float>(outFeatures) : null);
        _qOps = qOps ?? QuantizationFactory.Create();
        _ownsWeight = weight == null;
        _ownsBias = biasTensor == null && _bias != null;
    }

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public bool HasBias => _bias is not null;
    public string Name { get; }
    public Tensor<float> Weight => _weight;
    public Tensor<float>? Bias => _bias;

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter($"{Name}.weight", _weight);
        if (_bias is not null)
            yield return new Parameter($"{Name}.bias", _bias);
    }

    public Tensor<float> Forward(Tensor<float> input, Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        Tensor<float>? output = QuantizedForward(flat, workspace) ?? ScalarForward(flat, batchSize, workspace);
        
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
    private unsafe Tensor<float> ScalarForward(Tensor<float> input, int batchSize, Core.Memory.Workspace? workspace = null)
    {
        Tensor<float> output;
        _weightBT ??= _weight.Transpose();
        if (workspace != null)
            output = workspace.Rent<float>([batchSize, OutFeatures]);
        else
            output = new Tensor<float>(batchSize, OutFeatures);
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);
        fn(input.DataPtr, (byte*)_weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        return output;
    }

    private Tensor<float>? QuantizedForward(Tensor<float> input, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        if (_matMulFn == null) return null;
        var dtype = QuantDtype!.Value;
        var rawData = RawQuantizedData!;
        int m = input.ElementCount / InFeatures;
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);
        unsafe
        {
            fixed (byte* pRaw = rawData)
            {
                _matMulFn(input.DataPtr, pRaw, result.DataPtr, m, InFeatures, OutFeatures);
            }
        }
        return result;
    }

    public unsafe bool SetRawWeight(byte[]? rawData, QuantDType dtype)
    {
        RawQuantizedData = rawData;
        QuantDtype = dtype;
        _matMulFn = RawQuantizedData != null ? _qOps.QuantizedMatMulOpFor(dtype) : null;
        return _matMulFn != null;
    }

    public unsafe (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        using var weightBT = _weight.Transpose();
        var output = new Tensor<float>(batchSize, OutFeatures);
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);
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

    public unsafe Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state)
    {
        int batchSize = state.NeedReshape
            ? gradOutput.ElementCount / OutFeatures
            : gradOutput.Shape[^2];
        var flatGradOut = state.NeedReshape
            ? gradOutput.Reshape(batchSize, OutFeatures)
            : gradOutput;

        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);
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

    public void ReplaceWeights(Tensor<float> weight, Tensor<float>? biasTensor)
    {
        ThrowIfDisposed();

        if (_ownsWeight) _weight.Dispose();
        if (_ownsBias) _bias?.Dispose();

        _weight = weight;
        _bias = biasTensor;
        _ownsWeight = false;
        _ownsBias = false;
        InvalidateCache();
    }

    public void LoadWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(_weight.Data);
        InvalidateCache();
    }

    public void LoadWeightTransposed(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");

        int inF = InFeatures;
        int outF = OutFeatures;
        for (int o = 0; o < outF; o++)
            for (int i = 0; i < inF; i++)
                _weight.Data[i * outF + o] = data[o * inF + i];
        InvalidateCache();
    }

    private void InvalidateCache()
    {
        _weightBT?.Dispose();
        _weightBT = null;
    }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (_bias is null) throw new InvalidOperationException("No bias.");
        if (data.Length != _bias.ElementCount)
            throw new ArgumentException($"Expected {_bias.ElementCount} bias values, got {data.Length}.");
        data.CopyTo(_bias.Data);
    }

    public void FreeFloatWeight()
    {
        if (!UseQuantizedForward || RawQuantizedData == null) return;
        if (_ownsWeight)
            _weight.Dispose();
        _weight = new Tensor<float>(InFeatures, 1);
        _ownsWeight = true;
        _weightBT?.Dispose();
        _weightBT = null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsWeight) _weight.Dispose();
        _weightBT?.Dispose();
        if (_ownsBias) _bias?.Dispose();
    }

    private Tensor<float> BroadcastBias(int batchSize)
    {
        var broadcast = new Tensor<float>(batchSize, OutFeatures);
        for (int i = 0; i < batchSize; i++)
            _bias!.Data.CopyTo(broadcast.RowSpan(i));
        return broadcast;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LinearLayer));
}
