using SharpMind.Core.Quantization;
using SharpMind.Model.Format;

namespace SandBox;

/// <summary>
/// Tests whether VecDotQ8_0's stride formula correctly indexes into
/// GGUF row-major [OutDim, InDim] quantized data for various shapes.
/// 
/// VecDotQ8_0 stride:  byte offset = col * ceil(InFeatures/32) * 34
/// GGUF row stride:    byte offset = col * InDim * sizeof(Q8_0_block)
/// 
/// For square matrices (InDim == OutDim in GGUF?): strides match.
/// For non-square fused weights: strides may differ — this test detects it.
/// </summary>
public static class VecDotStrideTest
{
    private const int QK = 32;
    private const int BLOCK_BYTES = 34;

    /// <summary>
    /// Quantize a float row to Q8_0 blocks.  
    /// data[i] = scale * qvalues[i]  (qvalues are sbytes, scale is float16)
    /// </summary>
    private static unsafe byte[] QuantizeQ8_0(ReadOnlySpan<float> values, int count)
    {
        int nBlocks = (count + QK - 1) / QK;
        byte[] result = new byte[nBlocks * BLOCK_BYTES];

        fixed (byte* pResult = result)
        {
            for (int b = 0; b < nBlocks; b++)
            {
                int start = b * QK;
                int end = Math.Min(start + QK, count);
                int valid = end - start;

                // Find scale
                float maxAbs = 0;
                for (int i = start; i < end; i++)
                {
                    float abs = Math.Abs(values[i]);
                    if (abs > maxAbs) maxAbs = abs;
                }
                float d = maxAbs / 127f;
                if (d == 0) d = 1f; // avoid div by zero

                // Write scale as half
                ushort dHalf = FloatToHalf(d);
                *(ushort*)(pResult + b * BLOCK_BYTES) = dHalf;

                // Write quantized values
                sbyte* pVal = (sbyte*)(pResult + b * BLOCK_BYTES + 2);
                for (int i = 0; i < valid; i++)
                {
                    float q = values[start + i] / d;
                    pVal[i] = (sbyte)Math.Clamp((int)MathF.Round(q), -128, 127);
                }
                // Remaining (if partial block) already zero from alloc
            }
        }
        return result;
    }

    /// <summary>
    /// Float32 → IEEE half (truncated, no round)
    /// </summary>
    private static unsafe ushort FloatToHalf(float f)
    {
        uint bits = *(uint*)&f;
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = (bits >> 13) & 0x3FF;

        if (exp <= 0)
        {
            // Denormal in half (or zero)
            return (ushort)(sign | mant >> 1);
        }
        if (exp > 31)
        {
            // Infinity or NaN
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 1u : 0));
        }

        return (ushort)(sign | (uint)exp << 10 | mant);
    }

    /// <summary>
    /// Build synthetic Q8_0 raw data in GGUF [OutDim, InDim] row-major layout.
    /// weight[i, j] = i * 0.01f + j * 0.001f  (predictable, non-symmetric)
    /// </summary>
    private static (byte[] rawData, float[] dequant) BuildQ8_0Weight(int outDim, int inDim)
    {
        int count = outDim * inDim;
        float[] floatData = new float[count];
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
                floatData[o * inDim + i] = o * 0.01f + i * 0.001f;

        // Quantize in GGUF row-major order: each row = one output dim
        int rowElements = inDim;
        int totalBytes = 0;
        for (int o = 0; o < outDim; o++)
            totalBytes += QuantizeQ8_0(floatData.AsSpan(o * inDim, inDim), inDim).Length;

        byte[] raw = new byte[totalBytes];
        int offset = 0;
        for (int o = 0; o < outDim; o++)
        {
            byte[] rowBlock = QuantizeQ8_0(floatData.AsSpan(o * inDim, inDim), inDim);
            Buffer.BlockCopy(rowBlock, 0, raw, offset, rowBlock.Length);
            offset += rowBlock.Length;
        }

        // Dequantize for comparison
        float[] dequant = new float[count];
        DequantizeQ8_0(raw, dequant, outDim, inDim);

        return (raw, dequant);
    }

    private static unsafe void DequantizeQ8_0(byte[] raw, float[] dest, int outDim, int inDim)
    {
        int nBlocks = (inDim + QK - 1) / QK;
        fixed (byte* pRaw = raw)
        fixed (float* pDest = dest)
        {
            for (int o = 0; o < outDim; o++)
            {
                float* pRow = pDest + (long)o * inDim;
                for (int b = 0; b < nBlocks; b++)
                {
                    byte* block = pRaw + (long)o * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
                    float d = GgufLoaderFactory.Default.HalfToFloat(*(ushort*)block);
                    sbyte* values = (sbyte*)(block + 2);
                    int start = b * QK;
                    int end = Math.Min(start + QK, inDim);
                    for (int i = start; i < end; i++)
                        pRow[i] = values[i - start] * d;
                }
            }
        }
    }

    /// <summary>
    /// Compute the CORRECT float dot product for output column 'col':
    /// output[col] = sum_{i=0}^{inDim-1} input[i] * weight[i][col]
    /// In GGUF [Out, In] row-major: weight[i][col] = dequant[col * inDim + i]
    /// </summary>
    private static unsafe float CorrectFloatDot(float* input, float* dequant, int col, int inDim, int outDim)
    {
        double sum = 0;
        for (int i = 0; i < inDim; i++)
            sum += input[i] * dequant[col * inDim + i];
        return (float)sum;
    }

    /// <summary>
    /// Simulates what VecDotQ8_0 would compute for column 'col' if the raw data
    /// were in the WRONG [InDim, OutDim] layout (i.e., treating shape[0]=In, shape[1]=Out):
    /// output[col] = sum_{i=0}^{inDim-1} input[i] * weight[col][i]
    /// In GGUF [In, Out] row-major: weight[col][i] = dequant[col * outDim + i]
    /// </summary>
    private static unsafe float WrongFloatDot(float* input, float* dequant, int col, int inDim, int outDim)
    {
        double sum = 0;
        for (int i = 0; i < inDim; i++)
            sum += input[i] * dequant[col * outDim + i];
        return (float)sum;
    }

    public static unsafe void Run()
    {
        Console.Error.WriteLine("=== VecDotQ8_0 Stride Test ===\n");

        // Test configurations: (outDim, inDim, label)
        var configs = new (int Out, int In, string Label)[]
        {
            (1024, 1024, "Square (1024x1024)"),
            (1024, 3072, "Gate  (1024x3072)"),
            (1024, 6144, "Fused (1024x6144)"),
            (2048, 1024, "Q proj (1024x2048)"),
            (3072, 1024, "Down  (3072x1024)"),
            (576,  1536, "SmolLM2 gate (576x1536)"),
            (4096, 14336, "LLaMA-7B gate (4096x14336)"),
        };

        foreach (var (outDim, inDim, label) in configs)
        {
            Console.Error.WriteLine($"\n--- {label} ---");
            Console.Error.WriteLine($"  GGUF stores as [Out={outDim}, In={inDim}]");
            Console.Error.WriteLine($"  VecDot InFeatures={inDim}, nBlocks=ceil({inDim}/32)={ (inDim+31)/32 }");
            Console.Error.WriteLine($"  GGUF row blocks = ceil({outDim}/32)={ (outDim+31)/32 }");

            int nBlocks = (inDim + QK - 1) / QK;
            int ggufRowBlocks = (outDim + QK - 1) / QK;
            bool stridesMatch = nBlocks == ggufRowBlocks;
            Console.Error.WriteLine($"  VecDot col stride = {nBlocks} blocks/col, GGUF row stride = {ggufRowBlocks} blocks/row");
            Console.Error.WriteLine($"  Strides {(stridesMatch ? "MATCH" : "DIFFER")} — indexing is {(stridesMatch ? "CORRECT" : "LIKELY WRONG")}");

            if (outDim > 20000 || inDim > 20000)
            {
                Console.Error.WriteLine("  (skipping numeric test — too large for synthetic)");
                continue;
            }

            // Build synthetic data
            var (rawData, dequant) = BuildQ8_0Weight(outDim, inDim);

            // Random input
            Random rng = new(42);
            float[] inputArr = new float[inDim];
            for (int i = 0; i < inDim; i++)
                inputArr[i] = (float)(rng.NextDouble() * 2 - 1);

            int mismatchCount = 0;
            double maxRelError = 0;
            int worstCol = -1;

            fixed (float* pInput = inputArr)
            fixed (byte* pRaw = rawData)
            fixed (float* pDequant = dequant)
            {
                for (int col = 0; col < outDim; col++)
                {
                    float vecDot = QuantizationKernels.VecDotQ8_0_Scalar(pInput, pRaw, col, inDim);
                    float correct = CorrectFloatDot(pInput, pDequant, col, inDim, outDim);
                    float wrong = WrongFloatDot(pInput, pDequant, col, inDim, outDim);

                    float diff = Math.Abs(vecDot - correct);
                    float relError = correct != 0 ? diff / Math.Abs(correct) : diff;

                    if (relError > maxRelError) { maxRelError = relError; worstCol = col; }

                    // Tolerance: Q8_0 quantization error is ~0.5%, plus float rounding
                    if (diff > 0.1f && relError > 0.02f)
                        mismatchCount++;
                }
            }

            Console.Error.WriteLine($"  VecDot {'/'} Correct MATCH count: {outDim - mismatchCount}/{outDim}");
            Console.Error.WriteLine($"  VecDot {'/'} Correct MISMATCH count: {mismatchCount}/{outDim}");
            Console.Error.WriteLine($"  Max relative error: {maxRelError:F6} (col {worstCol})");
            Console.Error.WriteLine($"  Result: {(mismatchCount == 0 ? "PASS (indexing is correct)" : "FAIL (indexing is WRONG)")}");
        }

        Console.Error.WriteLine("\n=== Test Complete ===");
    }
}
