using JigSawDotNet;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public abstract class LinearLayer : IDisposable
{
    private const string QKernels = "SharpMind.Core.Quantization.QuantizationKernels";
    private static readonly QuantizationOps _staticOps = QuantizationFactory.Create();

    private Tensor<float> _weight;
    private Tensor<float>? _weightBT;
    private Tensor<float>? _bias;
    private bool _ownsWeight;
    private bool _ownsBias;
    private bool _disposed;

    public byte[]? RawQuantizedData { get; set; }
    public readonly QuantDType QuantDtype;
    public bool UseQuantizedForward => RawQuantizedData != null;

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType quantDType)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight ?? new Tensor<float>(inFeatures, outFeatures);
        _bias = biasTensor ?? (bias ? new Tensor<float>(outFeatures) : null);
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

    [PuzzleCornerPiece(SharpMindConfig.KeyLinear, true, null,
        "q8_0_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA)}",
        "q8_0_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA)}",
        "q8_0_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_AVX2)}",
        "q8_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_AVX2)}",
        "q8_0_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "q8_0_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "q8_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "q8_0_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "q5_0_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA)}",
        "q5_0_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA)}",
        "q5_0_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_AVX2)}",
        "q5_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_AVX2)}",
        "q5_0_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "q5_0_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "q5_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "q5_0_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "q6k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA)}",
        "q6k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA)}",
        "q6k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_AVX2)}",
        "q6k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_AVX2)}",
        "q6k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "q6k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "q6k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "q6k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "q4_0_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "q4_0_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}",
        "q4_0_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "q4_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "q4_0_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "q4_0_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}",
        "q4_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "q4_0_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}",
        "q4_1_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "q4_1_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "q4_1_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "q4_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "q4_1_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "q4_1_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "q4_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "q4_1_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "q2k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_FMA)}",
        "q2k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_FMA)}",
        "q2k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_AVX2)}",
        "q2k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_AVX2)}",
        "q2k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "q2k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "q2k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "q2k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "q3k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_FMA)}",
        "q3k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_FMA)}",
        "q3k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_AVX2)}",
        "q3k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_AVX2)}",
        "q3k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "q3k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "q3k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "q3k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "q4k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA)}",
        "q4k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA)}",
        "q4k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_AVX2)}",
        "q4k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_AVX2)}",
        "q4k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "q4k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "q4k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "q4k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "q5k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_FMA)}",
        "q5k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_FMA)}",
        "q5k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_AVX2)}",
        "q5k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_AVX2)}",
        "q5k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "q5k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "q5k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "q5k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "q8k_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_FMA)}",
        "q8k_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_FMA)}",
        "q8k_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_AVX2)}",
        "q8k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_AVX2)}",
        "q8k_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "q8k_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}",
        "q8k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "q8k_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}",
        "q8_1_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_FMA)}",
        "q8_1_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_FMA)}",
        "q8_1_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "q8_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}",
        "q8_1_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "q8_1_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}",
        "q8_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "q8_1_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}",
        "q5_1_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_FMA)}",
        "q5_1_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_FMA)}",
        "q5_1_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "q5_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "q5_1_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "q5_1_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "q5_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "q5_1_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "q4_nl_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "q4_nl_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "q4_nl_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "q4_nl_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "q4_nl_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "q4_nl_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "q4_nl_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "q4_nl_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "f32_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f32_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f32_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f32_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f16_serial_fma",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_fma",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "f16_serial_avx2",   $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "f16_serial_sse",    $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_sse",  $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "f16_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_scalar",$"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}")]
    public unsafe abstract void QuantizedMatMulFn(float* input, byte* rawWeights, float* output, int M, int K, int N);

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
        var fn = _staticOps.QuantizedMatMulOpFor(QuantDType.F32);
        fn(input.DataPtr, (byte*)_weightBT.DataPtr, output.DataPtr, batchSize, InFeatures, OutFeatures);
        return output;
    }

    private Tensor<float>? QuantizedForward(Tensor<float> input, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        if (RawQuantizedData == null) return null;
        var dtype = QuantDtype;
        var rawData = RawQuantizedData!;
        int m = input.ElementCount / InFeatures;
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);
        unsafe
        {
            fixed (byte* pRaw = rawData)
            {
                QuantizedMatMulFn(input.DataPtr, pRaw, result.DataPtr, m, InFeatures, OutFeatures);
            }
        }
        return result;
    }

    public void SetRawWeight(byte[]? rawData)
    {
        RawQuantizedData = rawData;
    }

    public unsafe (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input)
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

    public unsafe Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state)
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
        if (!UseQuantizedForward) return;
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
