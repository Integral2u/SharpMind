using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Format;

namespace SharpMind.Model.Layers;

public sealed class LinearLayer : IDisposable
{
    private readonly Tensor<float> _weight;
    private Tensor<float>? _weightBT;
    private readonly Tensor<float>? _bias;
    private QuantizationOps _qOps;
    private bool _disposed;

    // Raw GGUF quantized data for quantized matmul (null means use float32 path).
    public byte[]? RawQuantizedData { get; set; }
    public GgufDtype? QuantDtype { get; set; }
    public bool UseQuantizedForward { get; set; }

    public QuantizationOps QuantizationOps
    {
        get => _qOps;
        set => _qOps = value ?? throw new ArgumentNullException(nameof(value));
    }

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias = false)
        : this(name, inFeatures, outFeatures, bias, null)
    {
    }

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias, QuantizationOps? qOps)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = new Tensor<float>(inFeatures, outFeatures);
        _bias = bias ? new Tensor<float>(outFeatures) : null;
        _qOps = qOps ?? QuantizationFactory.CreateForSystem();
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

        Tensor<float> output;
        if (UseQuantizedForward && RawQuantizedData != null && QuantDtype.HasValue)
        {
            output = QuantizedForward(flat, ops);
        }
        else
        {
            _weightBT ??= TensorOps.Transpose(_weight);
            output = ops.MatMulWithBT(flat, _weightBT);
        }

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

    private Tensor<float> QuantizedForward(Tensor<float> input, TensorOps ops)
    {
        var dtype = QuantDtype!.Value;
        var rawData = RawQuantizedData!;
        int m = input.ElementCount / InFeatures;
        var result = new Tensor<float>(m, OutFeatures);
        int inF = InFeatures, outF = OutFeatures;

        for (int row = 0; row < m; row++)
        {
            unsafe
            {
                float* pIn = input.DataPtr + (long)row * inF;
                float* pOut = result.DataPtr + (long)row * outF;
                fixed (byte* pRaw = rawData)
                {
                    IntPtr pInPtr = (IntPtr)pIn;
                    IntPtr pOutPtr = (IntPtr)pOut;
                    IntPtr pRawPtr = (IntPtr)pRaw;
                    Parallel.For(0, outF, col =>
                    {
                        float* pInL = (float*)pInPtr;
                        float* pOutL = (float*)pOutPtr;
                        byte* pRawL = (byte*)pRawPtr;
                        pOutL[col] = VecDotQxK(pInL, pRawL, col, inF, dtype);
                    });
                }
            }
        }
        return result;
    }

    private unsafe float VecDotQxK(float* input, byte* rawWeights, int col, int inFeatures, GgufDtype dtype)
    {
        return dtype switch
        {
            GgufDtype.Q3_K => _qOps.VecDotQ3K(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_K => _qOps.VecDotQ4K(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_K => _qOps.VecDotQ5K(input, rawWeights, col, inFeatures),
            GgufDtype.Q6_K => _qOps.VecDotQ6K(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_0 => _qOps.VecDotQ4_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_1 => _qOps.VecDotQ4_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_0 => _qOps.VecDotQ5_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_1 => _qOps.VecDotQ5_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_0 => _qOps.VecDotQ8_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_1 => _qOps.VecDotQ8_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q2_K => _qOps.VecDotQ2K(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_K => _qOps.VecDotQ8K(input, rawWeights, col, inFeatures),
            _ => 0f
        };
    }

    private static bool IsSupportedQuantDtype(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q8_0 => true,
        GgufDtype.Q4_0 => true,
        GgufDtype.Q4_1 => true,
        GgufDtype.Q5_0 => true,
        GgufDtype.Q5_1 => true,
        GgufDtype.Q8_1 => true,
        GgufDtype.Q2_K => true,
        GgufDtype.Q3_K => true,
        GgufDtype.Q4_K => true,
        GgufDtype.Q5_K => true,
        GgufDtype.Q6_K => true,
        GgufDtype.Q8_K => true,
        _ => false
    };

    public bool SetRawWeight(byte[] rawData, GgufDtype dtype)
    {
        RawQuantizedData = rawData;
        QuantDtype = dtype;
        UseQuantizedForward = IsSupportedQuantDtype(dtype);
        return UseQuantizedForward;
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
        InvalidateCache();
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weight.Dispose();
        _weightBT?.Dispose();
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