using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public sealed class LinearLayer : IDisposable
{
    private readonly Tensor<float> _weight;
    private readonly Tensor<float>? _bias;
    private bool _disposed;

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = new Tensor<float>(inFeatures, outFeatures);
        _bias = bias ? new Tensor<float>(outFeatures) : null;
    }

    public LinearLayer(int inFeatures, int outFeatures, bool bias = false)
        : this($"Linear.{Guid.NewGuid():N}", inFeatures, outFeatures, bias)
    {
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

    public Tensor<float> Forward(Tensor<float> input, TensorOps ops)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        var output = ops.MatMul(flat, _weight);
        if (_bias is not null)
            TensorOps.AddInPlace(output, BroadcastBias(batchSize));
        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return reshaped;
        }
        return output;
    }

    public (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input, TensorOps ops)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        var output = ops.MatMul(flat, _weight);
        if (_bias is not null)
            TensorOps.AddInPlace(output, BroadcastBias(batchSize));
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

    public Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state, TensorOps ops)
    {
        int batchSize = state.NeedReshape
            ? gradOutput.ElementCount / OutFeatures
            : gradOutput.Shape[^2];
        var flatGradOut = state.NeedReshape
            ? gradOutput.Reshape(batchSize, OutFeatures)
            : gradOutput;
        
        // gradInput = gradOutput @ weight
        var gradInputFlat = ops.MatMul(flatGradOut, TensorOps.Transpose(_weight));
        
        // gradWeight += input^T @ gradOutput
        using var inputT = TensorOps.Transpose(state.Input);
        using var dw = ops.MatMul(inputT, flatGradOut);
        var wg = state.WeightGrad;
        for (int i = 0; i < dw.ElementCount; i++)
            wg.Data[i] += dw.Data[i];
        dw.Dispose();
        inputT.Dispose();

        // gradBias
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
        
        flatGradOut.Dispose();
        
        if (state.NeedReshape)
        {
            int[] inDims = [.. state.InputDims[..^1], InFeatures];
            var reshaped = gradInputFlat.Reshape(inDims);
            gradInputFlat.Dispose();
            return reshaped;
        }
        return gradInputFlat;
    }

    public void LoadWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(_weight.Data);
    }

    public void LoadWeightTransposed(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");

        // GGUF: [Out, In] -> SharpMind: [In, Out]
        int inF = InFeatures;
        int outF = OutFeatures;
        for (int o = 0; o < outF; o++)
        {
            for (int i = 0; i < inF; i++)
            {
                float val = data[o * inF + i];
                if (float.IsInfinity(val) || float.IsNaN(val))
                    val = 0f;
                _weight.Data[i * outF + o] = val;
            }
        }
    }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (_bias is null) throw new InvalidOperationException("No bias.");
        if (data.Length != _bias.ElementCount)
            throw new ArgumentException($"Expected {_bias.ElementCount} bias values, got {data.Length}.");
        
        // Check for corrupted bias values (e.g., F16 stored as F32 gives extreme values)
        bool bad = false;
        int checkLen = Math.Min(data.Length, 16);
        float maxVal = 0f;
        for (int i = 0; i < checkLen; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { bad = true; break; }
            float av = Math.Abs(v);
            if (av > maxVal) maxVal = av;
        }
        if (!bad && maxVal > 100f)
            bad = true;
        if (bad)
        {
            // Zero out rather than letting extreme values cause NaN
            _bias.Data.Clear();
            return;
        }

        for (int i = 0; i < data.Length; i++)
        {
            float val = data[i];
            if (float.IsInfinity(val) || float.IsNaN(val))
                val = 0f;
            _bias.Data[i] = val;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weight.Dispose();
        _bias?.Dispose();
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

public sealed class LinearLayerState
{
    public Tensor<float> Input { get; }
    public int[] InputDims { get; }
    public bool NeedReshape { get; }
    public Tensor<float> WeightGrad { get; }
    public Tensor<float>? BiasGrad { get; set; }

    public LinearLayerState(Tensor<float> originalInput, Tensor<float> flatInput, bool needReshape, Tensor<float> weight)
    {
        Input = flatInput;
        InputDims = originalInput.Shape.Dims.ToArray();
        NeedReshape = needReshape;
        var dims = weight.Shape.Dims.ToArray();
        WeightGrad = Tensor<float>.Zeros(dims);
    }
}