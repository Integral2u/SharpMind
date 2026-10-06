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
    private readonly Q8_0WideWeights.Cache _wide = new();

    /// <summary>
    /// Set by <see cref="LinearLayerFactory"/> when the selected kernel is the parallel FMA Q8_0 one,
    /// which <see cref="Q8_0WideWeights"/> stands in for; serial or lower-tier selections keep their kernel.
    /// </summary>
    public bool WideAllowed { get; internal set; }

    /// <summary>Whether the repacked wide copy (and its raw source) is currently cached.</summary>
    internal bool HasCachedWide => _wide.HasSlot;

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
        "q1_0_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Serial_FMA)}",
        "q1_0_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Parallel_FMA)}",
        "q1_0_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Serial_FMA)}",
        "q1_0_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Parallel_FMA)}",
        "q1_0_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Serial_Scalar)}",
        "q1_0_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Parallel_Scalar)}",
        "q1_0_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Serial_Scalar)}",
        "q1_0_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulQ1_0_Parallel_Scalar)}",
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
        "f16_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "bf16_serial_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Serial_FMA)}",
        "bf16_parallel_fma", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Parallel_FMA)}",
        "bf16_serial_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Serial_FMA)}",
        "bf16_parallel_avx2", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Parallel_FMA)}",
        "bf16_serial_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Serial_Scalar)}",
        "bf16_parallel_sse", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Parallel_Scalar)}",
        "bf16_serial_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Serial_Scalar)}",
        "bf16_parallel_scalar", $"{QKernels}.{nameof(QuantizationKernels.QuantizedMatMulBF16_Parallel_Scalar)}")]
    public unsafe abstract void QuantizedMatMulFn(float* input, byte* rawWeights, float* output, int M, int K, int N);

    /// <summary>
    /// Float fallback for layers whose weights stay F32 (MoE routers, and any other
    /// unquantized tensor). <paramref name="w"/> is [K, N] row-major, so element (i, o)
    /// lives at w[i * N + o] — the layout <see cref="LoadWeightTransposed"/> produces.
    /// </summary>
    private static unsafe void FloatMatMul(float* input, float* w, float* output, int M, int K, int N)
    {
        for (int r = 0; r < M; r++)
        {
            float* x = input + (long)r * K;
            float* y = output + (long)r * N;
            new Span<float>(y, N).Clear();
            for (int i = 0; i < K; i++)
            {
                float xi = x[i];
                if (xi == 0f) continue;
                float* wRow = w + (long)i * N;
                for (int o = 0; o < N; o++) y[o] += xi * wRow[o];
            }
        }
    }

    public override unsafe Tensor<float> Forward(Tensor<float> input, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        using var flatView = needReshape ? input.Reshape(batchSize, InFeatures) : null;
        var flat = flatView ?? input;

        int m = flat.ElementCount / InFeatures;
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>([m, OutFeatures])
            : new Tensor<float>(m, OutFeatures);

        if (RawQuantizedData is not null)
        {
            if (WideAllowed && Q8_0WideWeights.Enabled &&
                _wide.Get(RawQuantizedData, InFeatures, OutFeatures) is { } wide)
            {
                wide.MatMul(flat.DataPtr, result.DataPtr, m);
            }
            else
            {
                fixed (byte* pRaw = RawQuantizedData)
                {
                    QuantizedMatMulFn(flat.DataPtr, pRaw, result.DataPtr, m, InFeatures, OutFeatures);
                }
            }
        }
        else if (QuantDtype == QuantDType.F32 && _weight.ElementCount == (long)InFeatures * OutFeatures)
        {
            // An F32 weight must not be handed to a quantized kernel: every
            // QuantizedMatMulFn reinterprets its input as (fp16 scale, int8 x 32)
            // blocks, so an F32 router like Qwen2-MoE's ffn_gate_inp came out as
            // garbage — a near-uniform softmax that routed every token to the wrong
            // experts. Do the multiply in float instead, using the same
            // [inFeatures, outFeatures] layout the loader writes (row i, column o).
            fixed (float* pWeight = _weight.Data)
            {
                FloatMatMul(flat.DataPtr, pWeight, result.DataPtr, m, InFeatures, OutFeatures);
            }
        }
        else
        {
            throw new InvalidOperationException(
                $"[{Name}] RawQuantizedData is null and no float fallback available " +
                $"(dtype={QuantDtype}, weightElements={_weight.ElementCount}, " +
                $"expected={(long)InFeatures * OutFeatures}). " +
                $"The forward pass cannot proceed without weight data.");
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
        // F32/F16 tensors (e.g. MoE router `ffn_gate_inp`, Qwen shared-expert
        // `ffn_gate_inp_shexp`) have no raw quantized payload — the float tensor IS the
        // only copy. Freeing it left the layer holding an InFeatures-sized placeholder,
        // so the forward pass died with "weightElements=2048, expected=122880".
        if (RawQuantizedData is null)
            return;

        // Drop the repack and the raw source it roots. Streaming frees and reloads a layer
        // every forward; without this the freed layer's whole Q8_0 payload (and the ~6%
        // wider repack) stayed rooted in the cache, so a streaming load ended up holding
        // every layer's weights at once - more than a full load. The next reload installs
        // a fresh raw array and rebuilds. See Q8_0WideWeights.Cache.Clear.
        _wide.Clear();
        if (_ownsWeight)
            _weight.Dispose();
        _weight = new Tensor<float>(InFeatures, 1);
        _ownsWeight = true;
    }

    protected override void OnDispose()
    {
        _wide.Clear();
        base.OnDispose();
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
            // GGUF stores ne[0] (the quantised row, K) first, so the shape is [InFeatures,
            // OutFeatures]. Passing [OutFeatures, InFeatures] happened to agree while the
            // byte-count formula ignored which dim was the row, and diverged as soon as
            // the row length was not block-aligned.
            long expectedBytes = QuantizationOps.GetRawTensorByteCount([InFeatures, OutFeatures], QuantDtype);
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
