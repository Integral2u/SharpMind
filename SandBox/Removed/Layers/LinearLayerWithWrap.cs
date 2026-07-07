using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Format;

namespace SharpMind.Model.Layers;

public sealed class LinearLayer : IDisposable
{
    private Tensor<float> _weight;
    private Tensor<float>? _weightBT;
    private Tensor<float>? _bias;
    private QuantizationOps _qOps;
    private bool _disposed;
    private unsafe delegate float VecDotFn(float* input, byte* rawWeights, int col, int inFeatures);
    private VecDotFn? _vecDotFn;
    private unsafe delegate void QuantizedMatMulFn(float* input, byte* rawWeights, float* output, int M, int K, int N);
    private QuantizedMatMulFn? _matMulFn;

    // Raw GGUF quantized data for quantized matmul (null means use float32 path).
    // DEBUG: set to true before CreateSession to force float forward path on all layers
    public static bool ForceFloatForward { get; set; }

    public byte[]? RawQuantizedData { get; set; }
    public GgufDtype? QuantDtype { get; set; }
    public bool UseQuantizedForward => !ForceFloatForward && RawQuantizedData != null && QuantDtype != null && IsSupportedQuantDtype(QuantDtype.Value);

    public QuantizationOps QuantizationOps
    {
        get => _qOps;
        set => _qOps = value ?? throw new ArgumentNullException(nameof(value));
    }

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias = false)
        : this(name, inFeatures, outFeatures, bias, null, null, null)
    {
    }

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias, QuantizationOps? qOps)
        : this(name, inFeatures, outFeatures, bias, qOps, null, null)
    {
    }

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

    private bool _ownsWeight;
    private bool _ownsBias;

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

    public Tensor<float> Forward(Tensor<float> input, TensorOps ops, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        Tensor<float> output;
        if (UseQuantizedForward && RawQuantizedData != null && QuantDtype.HasValue)
        {
            output = QuantizedForward(flat, ops, workspace);
        }
        else
        {
            _weightBT ??= TensorOps.Transpose(_weight);
            if (workspace != null)
            {
                output = workspace.Rent<float>([batchSize, OutFeatures]);
                ops.MatMulWithBTInto(flat, _weightBT, output);
            }
            else
            {
                output = ops.MatMulWithBT(flat, _weightBT);
            }
        }

        if (_bias is not null)
        {
            if (workspace != null)
            {
                var biasB = workspace.Rent<float>([batchSize, OutFeatures]);
                for (int i = 0; i < batchSize; i++)
                    _bias!.Data.CopyTo(biasB.RowSpan(i));
                TensorOps.AddInPlace<float>(output, biasB);
                // Rent returns a tensor that doesn't own memory, so we don't dispose it.
                // But the workspace will be reset anyway.
            }
            else
            {
                TensorOps.AddInPlace(output, BroadcastBias(batchSize));
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

    private Tensor<float> QuantizedForward(Tensor<float> input, TensorOps ops, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        var dtype = QuantDtype!.Value;
        var rawData = RawQuantizedData!;
        int m = input.ElementCount / InFeatures;
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);
        int inF = InFeatures, outF = OutFeatures;

        unsafe
        {
            fixed (byte* pRaw = rawData)
            {
                if (_matMulFn != null)
                {
                    _matMulFn(input.DataPtr, pRaw, result.DataPtr, m, inF, outF);
                }

                else if (m <= 1)
                {
                    float* pInRow = input.DataPtr;
                    float* pOutRow = result.DataPtr;
                    for (int col = 0; col < outF; col++)
                        pOutRow[col] = VecDotQxK(pInRow, pRaw, col, inF);
                }
                else
                {
                    IntPtr pRawPtr = (IntPtr)pRaw;
                    Parallel.For(0, m, row =>
                    {
                        byte* pRawL = (byte*)pRawPtr;
                        float* pInRow = input.DataPtr + (long)row * inF;
                        float* pOutRow = result.DataPtr + (long)row * outF;
                        for (int col = 0; col < outF; col++)
                            pOutRow[col] = VecDotQxK(pInRow, pRawL, col, inF);
                    });
                }
            }
        }
        return result;
    }


    private unsafe float VecDotQxK(float* input, byte* rawWeights, int col, int inFeatures) => _vecDotFn!(input, rawWeights, col, inFeatures);

    private static bool IsSupportedQuantDtype(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q8_0 => true,
        GgufDtype.IQ4_NL => true,
        GgufDtype.Q4_0 => true,
        GgufDtype.Q4_1 => true,
        GgufDtype.Q5_0 => true,
        GgufDtype.Q5_1 => true,
        GgufDtype.Q8_1 => true,
        GgufDtype.Q2_K or GgufDtype.Q2_K_S => true,
        GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => true,
        GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => true,
        GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => true,
        GgufDtype.Q6_K or GgufDtype.Q6_K_S => true,
        GgufDtype.Q8_K => true,
        _ => false
    };

    public unsafe bool SetRawWeight(byte[]? rawData, GgufDtype dtype)
    {
        RawQuantizedData = rawData;
        QuantDtype = dtype;
        _vecDotFn = dtype switch
        {
            GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => _qOps.VecDotQ3K,
            GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => _qOps.VecDotQ4K,
            GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => _qOps.VecDotQ5K,
            GgufDtype.Q6_K or GgufDtype.Q6_K_S => _qOps.VecDotQ6K,
            GgufDtype.IQ4_NL => _qOps.VecDotQ4_NL,
            GgufDtype.Q4_0 => _qOps.VecDotQ4_0,
            GgufDtype.Q4_1 => _qOps.VecDotQ4_1,
            GgufDtype.Q5_0 => _qOps.VecDotQ5_0,
            GgufDtype.Q5_1 => _qOps.VecDotQ5_1,
            GgufDtype.Q8_0 => _qOps.VecDotQ8_0,
            GgufDtype.Q8_1 => _qOps.VecDotQ8_1,
            GgufDtype.Q2_K or GgufDtype.Q2_K_S => _qOps.VecDotQ2K,
            GgufDtype.Q8_K => _qOps.VecDotQ8K,
            _ => null
        };
        _matMulFn = dtype switch
        {
            GgufDtype.Q8_0 => _qOps.QuantizedMatMulQ8_0,
            GgufDtype.Q5_0 => _qOps.QuantizedMatMulQ5_0,
            GgufDtype.Q6_K or GgufDtype.Q6_K_S => _qOps.QuantizedMatMulQ6K,
            GgufDtype.IQ4_NL when _qOps.VecDotQ4_NL != null => WrapVecDotAsMatMul(_qOps.VecDotQ4_NL),
            GgufDtype.Q4_0 when _qOps.VecDotQ4_0 != null => WrapVecDotAsMatMul(_qOps.VecDotQ4_0),
            GgufDtype.Q4_1 when _qOps.VecDotQ4_1 != null => WrapVecDotAsMatMul(_qOps.VecDotQ4_1),
            GgufDtype.Q5_1 when _qOps.VecDotQ5_1 != null => WrapVecDotAsMatMul(_qOps.VecDotQ5_1),
            GgufDtype.Q2_K or GgufDtype.Q2_K_S when _qOps.VecDotQ2K != null => WrapVecDotAsMatMul(_qOps.VecDotQ2K),
            GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L when _qOps.VecDotQ3K != null => WrapVecDotAsMatMul(_qOps.VecDotQ3K),
            GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M when _qOps.VecDotQ4K != null => WrapVecDotAsMatMul(_qOps.VecDotQ4K),
            GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M when _qOps.VecDotQ5K != null => WrapVecDotAsMatMul(_qOps.VecDotQ5K),
            GgufDtype.Q8_K when _qOps.VecDotQ8K != null => WrapVecDotAsMatMul(_qOps.VecDotQ8K),
            _ => null
        };

        return UseQuantizedForward;
    }

    private static unsafe QuantizedMatMulFn WrapVecDotAsMatMul(VecDotFn vecDot)
    {
        return (float* input, byte* rawWeights, float* output, int M, int K, int N) =>
        {
            for (int row = 0; row < M; row++)
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = vecDot(pInRow, rawWeights, col, K);
            }
        };
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

        // Dispose old weights only if we owned them
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

    /// <summary>
    /// When quantized forward is active and raw data exists, the float weight
    /// tensor is never read. Replace with a minimal placeholder to free memory.
    /// The original float tensor is owned by BlockWeights (when _ownsWeight is
    /// false) and stays alive there.
    /// </summary>
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