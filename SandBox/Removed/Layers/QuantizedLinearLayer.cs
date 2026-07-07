using JigSawDotNet;
using SharpMind.Core.Memory;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Format;

namespace SharpMind.Model.Layers;

public abstract class QuantizedLinearLayer : LinearLayer
{
    public const string KeyQMatMul = "qmatmul";
    private const string NS_K = "SharpMind.Core.Quantization.QuantizationKernels";

    private byte[] _rawData;
    private GgufDtype _dtype;
    private bool _ownsBias;

    internal byte[] RawData => _rawData;
    internal GgufDtype Dtype => _dtype;

    protected QuantizedLinearLayer(string name, int inFeatures, int outFeatures,
        byte[] rawData, GgufDtype dtype, Tensor<float>? biasTensor)
        : base(name, inFeatures, outFeatures, biasTensor)
    {
        _rawData = rawData ?? throw new ArgumentNullException(nameof(rawData));
        _dtype = dtype;
        _ownsBias = biasTensor == null;
    }

    [PuzzleCornerPiece(KeyQMatMul,
        "q8_0_fma",    NS_K + ".QuantizedMatMulQ8_0_FMA",
        "q8_0_avx2",   NS_K + ".QuantizedMatMulQ8_0_AVX2",
        "q8_0_sse",    NS_K + ".QuantizedMatMulQ8_0_Scalar",
        "q8_0_scalar", NS_K + ".QuantizedMatMulQ8_0_Scalar",
        "q4_0_fma",    NS_K + ".QuantizedMatMulQ4_0_Scalar",
        "q4_0_avx2",   NS_K + ".QuantizedMatMulQ4_0_Scalar",
        "q4_0_sse",    NS_K + ".QuantizedMatMulQ4_0_Scalar",
        "q4_0_scalar", NS_K + ".QuantizedMatMulQ4_0_Scalar",
        "q4_1_fma",    NS_K + ".QuantizedMatMulQ4_1_Scalar",
        "q4_1_avx2",   NS_K + ".QuantizedMatMulQ4_1_Scalar",
        "q4_1_sse",    NS_K + ".QuantizedMatMulQ4_1_Scalar",
        "q4_1_scalar", NS_K + ".QuantizedMatMulQ4_1_Scalar",
        "q5_0_fma",    NS_K + ".QuantizedMatMulQ5_0_FMA",
        "q5_0_avx2",   NS_K + ".QuantizedMatMulQ5_0_AVX2",
        "q5_0_sse",    NS_K + ".QuantizedMatMulQ5_0_Scalar",
        "q5_0_scalar", NS_K + ".QuantizedMatMulQ5_0_Scalar",
        "q5_1_fma",    NS_K + ".QuantizedMatMulQ5_1_Scalar",
        "q5_1_avx2",   NS_K + ".QuantizedMatMulQ5_1_Scalar",
        "q5_1_sse",    NS_K + ".QuantizedMatMulQ5_1_Scalar",
        "q5_1_scalar", NS_K + ".QuantizedMatMulQ5_1_Scalar",
        "q8_1_fma",    NS_K + ".QuantizedMatMulQ8_1_Scalar",
        "q8_1_avx2",   NS_K + ".QuantizedMatMulQ8_1_Scalar",
        "q8_1_sse",    NS_K + ".QuantizedMatMulQ8_1_Scalar",
        "q8_1_scalar", NS_K + ".QuantizedMatMulQ8_1_Scalar",
        "q4_nl_fma",   NS_K + ".QuantizedMatMulQ4_NL_Scalar",
        "q4_nl_avx2",  NS_K + ".QuantizedMatMulQ4_NL_Scalar",
        "q4_nl_sse",   NS_K + ".QuantizedMatMulQ4_NL_Scalar",
        "q4_nl_scalar",NS_K + ".QuantizedMatMulQ4_NL_Scalar",
        "q2k_fma",     NS_K + ".QuantizedMatMulQ2K_Scalar",
        "q2k_avx2",    NS_K + ".QuantizedMatMulQ2K_Scalar",
        "q2k_sse",     NS_K + ".QuantizedMatMulQ2K_Scalar",
        "q2k_scalar",  NS_K + ".QuantizedMatMulQ2K_Scalar",
        "q3k_fma",     NS_K + ".QuantizedMatMulQ3K_Scalar",
        "q3k_avx2",    NS_K + ".QuantizedMatMulQ3K_Scalar",
        "q3k_sse",     NS_K + ".QuantizedMatMulQ3K_Scalar",
        "q3k_scalar",  NS_K + ".QuantizedMatMulQ3K_Scalar",
        "q4k_fma",     NS_K + ".QuantizedMatMulQ4K_Scalar",
        "q4k_avx2",    NS_K + ".QuantizedMatMulQ4K_Scalar",
        "q4k_sse",     NS_K + ".QuantizedMatMulQ4K_Scalar",
        "q4k_scalar",  NS_K + ".QuantizedMatMulQ4K_Scalar",
        "q5k_fma",     NS_K + ".QuantizedMatMulQ5K_Scalar",
        "q5k_avx2",    NS_K + ".QuantizedMatMulQ5K_Scalar",
        "q5k_sse",     NS_K + ".QuantizedMatMulQ5K_Scalar",
        "q5k_scalar",  NS_K + ".QuantizedMatMulQ5K_Scalar",
        "q6k_fma",     NS_K + ".QuantizedMatMulQ6K_FMA",
        "q6k_avx2",    NS_K + ".QuantizedMatMulQ6K_AVX2",
        "q6k_sse",     NS_K + ".QuantizedMatMulQ6K_Scalar",
        "q6k_scalar",  NS_K + ".QuantizedMatMulQ6K_Scalar",
        "q8k_fma",     NS_K + ".QuantizedMatMulQ8K_Scalar",
        "q8k_avx2",    NS_K + ".QuantizedMatMulQ8K_Scalar",
        "q8k_sse",     NS_K + ".QuantizedMatMulQ8K_Scalar",
        "q8k_scalar",  NS_K + ".QuantizedMatMulQ8K_Scalar",
        "f32_fma",     NS_K + ".QuantizedMatMulF32_Scalar",
        "f32_avx2",    NS_K + ".QuantizedMatMulF32_Scalar",
        "f32_sse",     NS_K + ".QuantizedMatMulF32_Scalar",
        "f32_scalar",  NS_K + ".QuantizedMatMulF32_Scalar",
        "f16_fma",     NS_K + ".QuantizedMatMulF16_Scalar",
        "f16_avx2",    NS_K + ".QuantizedMatMulF16_Scalar",
        "f16_sse",     NS_K + ".QuantizedMatMulF16_Scalar",
        "f16_scalar",  NS_K + ".QuantizedMatMulF16_Scalar")]
    public abstract unsafe void QuantizedMatMul(float* input, byte* rawWeights, float* output, int M, int K, int N);

    public override Tensor<float> Forward(Tensor<float> input, TensorOps ops, Workspace? workspace = null)
    {
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        int m = flat.ElementCount / InFeatures;

        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);

        unsafe
        {
            fixed (byte* pRaw = _rawData)
            {
                QuantizedMatMul(flat.DataPtr, pRaw, result.DataPtr, m, InFeatures, OutFeatures);
            }
        }

        AddBias(result, m, workspace);

        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = result.Reshape(outDims);
            result.Dispose();
            return reshaped;
        }
        return result;
    }

    internal void UpdateRawData(byte[] rawData, GgufDtype dtype)
    {
        _rawData = rawData ?? throw new ArgumentNullException(nameof(rawData));
        _dtype = dtype;
    }

    public override IEnumerable<Parameter> Parameters()
    {
        yield break;
    }

    protected override void DisposeCore()
    {
        if (_ownsBias) _bias?.Dispose();
    }
}
