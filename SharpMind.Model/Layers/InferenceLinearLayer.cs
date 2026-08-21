using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public abstract class InferenceLinearLayer : LinearLayer
{
    private const string QKernels = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(SharpMind.Core.Quantization)}.{nameof(QuantizationKernels)}";

    public byte[]? RawQuantizedData { get; set; }
    public readonly QuantDType QuantDtype;

    protected InferenceLinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType quantDType)
        // Forward reads RawQuantizedData, never the float weight, so a null weight
        // (quantized-resident loading) must not materialise a full F32 copy —
        // that second copy is what put a 7B out of reach and OOM'd a 14B at load.
        : base(name, inFeatures, outFeatures, bias, weight, biasTensor, allocateFullWeight: false)
    {
        QuantDtype = quantDType;
    }

    [PuzzleCornerPiece(SharpMindConfig.KeyLinear, true, null,
        "q8_0_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA)}",
        "q8_0_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA)}",
        "q8_0_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_AVX2)}",
        "q8_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_AVX2)}",
        "q8_0_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "q8_0_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "q8_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "q8_0_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "q5_0_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA)}",
        "q5_0_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA)}",
        "q5_0_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_AVX2)}",
        "q5_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_AVX2)}",
        "q5_0_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "q5_0_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "q5_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "q5_0_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "q6k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA)}",
        "q6k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA)}",
        "q6k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_AVX2)}",
        "q6k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_AVX2)}",
        "q6k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "q6k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "q6k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "q6k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "q4_0_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "q4_0_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "q4_0_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "q4_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "q4_0_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_SSE)}",
        "q4_0_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_SSE)}",
        "q4_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "q4_0_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}",
        "q4_1_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2)}",
        "q4_1_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2)}",
        "q4_1_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2)}",
        "q4_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2)}",
        "q4_1_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_SSE)}",
        "q4_1_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_SSE)}",
        "q4_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "q4_1_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "q2k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_FMA)}",
        "q2k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_FMA)}",
        "q2k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_AVX2)}",
        "q2k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_AVX2)}",
        "q2k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "q2k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "q2k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "q2k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "q3k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_FMA)}",
        "q3k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_FMA)}",
        "q3k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_AVX2)}",
        "q3k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_AVX2)}",
        "q3k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "q3k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "q3k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "q3k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "q4k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA)}",
        "q4k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA)}",
        "q4k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_AVX2)}",
        "q4k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_AVX2)}",
        "q4k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "q4k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "q4k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "q4k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "q5k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_FMA)}",
        "q5k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_FMA)}",
        "q5k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_AVX2)}",
        "q5k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_AVX2)}",
        "q5k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "q5k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "q5k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "q5k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "q8k_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_FMA)}",
        "q8k_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_FMA)}",
        "q8k_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_AVX2)}",
        "q8k_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_AVX2)}",
        "q8k_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "q8k_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}",
        "q8k_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "q8k_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}",
        "q8_1_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_FMA)}",
        "q8_1_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_FMA)}",
        "q8_1_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_AVX2)}",
        "q8_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_AVX2)}",
        "q8_1_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_SSE)}",
        "q8_1_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_SSE)}",
        "q8_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "q8_1_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}",
        "q5_1_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_FMA)}",
        "q5_1_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_FMA)}",
        "q5_1_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_AVX2)}",
        "q5_1_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_AVX2)}",
        "q5_1_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "q5_1_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "q5_1_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "q5_1_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "q4_nl_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "q4_nl_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "q4_nl_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "q4_nl_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "q4_nl_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "q4_nl_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "q4_nl_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "q4_nl_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "f32_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_FMA)}",
        "f32_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_FMA)}",
        "f32_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_FMA)}",
        "f32_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_FMA)}",
        "f32_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f32_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "f32_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "f16_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_FMA)}",
        "f16_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_FMA)}",
        "f16_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_FMA)}",
        "f16_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_FMA)}",
        "f16_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "f16_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "f16_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}")]
    public unsafe abstract void QuantizedMatMulFn(float* input, byte* rawWeights, float* output, int M, int K, int N);

    public override unsafe Tensor<float> Forward(Tensor<float> input, Workspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        Tensor<float>? flatView = needReshape ? input.Reshape(batchSize, InFeatures) : null;
        var flat = flatView ?? input;

        int m = flat.ElementCount / InFeatures;
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);

        // --- Heap-corruption guard: verify raw data size matches dtype/K/N before kernel call ---
        if (RawQuantizedData is not null)
        {
            long expectedBytes = QuantizationOps.GetRawTensorByteCount([OutFeatures, InFeatures], QuantDtype);
            if (RawQuantizedData.Length != expectedBytes)
                throw new InvalidOperationException(
                    $"[{Name}] RawQuantizedData size mismatch: dtype={QuantDtype}, " +
                    $"K={InFeatures}, N={OutFeatures}, " +
                    $"expected={expectedBytes}, actual={RawQuantizedData.Length}. " +
                    $"A kernel would write beyond this buffer.");
        }

        fixed (byte* pRaw = RawQuantizedData)
        {
            QuantizedMatMulFn(flat.DataPtr, pRaw, result.DataPtr, m, InFeatures, OutFeatures);
        }

        if (_bias is not null)
            AddBiasInPlace(result, batchSize);
        if (needReshape)
        {
            Span<int> outDims = stackalloc int[input.Rank];
            input.Shape.Dims[..^1].CopyTo(outDims);
            outDims[^1] = OutFeatures;
            var reshaped = result.Reshape(outDims);
            result.Dispose();
            return reshaped;
        }
        return result;
    }

    public override void FreeFloatWeight()
    {
        if (_ownsWeight)
            _weight.Dispose();
        _weight = new Tensor<float>(InFeatures, 1);
        _ownsWeight = true;
    }

    public override void SetRawWeight(byte[]? rawData)
    {
        // Check on arrival, not at the first matmul. The same mismatch used to
        // surface deep inside a forward pass after the model had "loaded"
        // successfully, which reads as a runtime failure rather than what it is:
        // a model whose tensor shapes do not match the architecture we derived
        // from its config. Forward keeps its own guard as defence in depth.
        if (rawData is not null)
        {
            long expectedBytes = QuantizationOps.GetRawTensorByteCount([OutFeatures, InFeatures], QuantDtype);
            if (rawData.Length != expectedBytes)
                throw new NotSupportedException(
                    $"[{Name}] weight shape does not match this architecture: dtype={QuantDtype}, " +
                    $"K={InFeatures}, N={OutFeatures} expects {expectedBytes} bytes but the model " +
                    $"provides {rawData.Length}. The file's tensor is " +
                    $"{(expectedBytes % rawData.Length == 0 ? $"1/{expectedBytes / rawData.Length} of" : "not")} " +
                    "the expected size, so the layer dimensions derived from the model config are wrong " +
                    "for this architecture. Loading it would produce garbage or read past the buffer.");
        }

        RawQuantizedData = rawData;
    }
}
