using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Quantization;
using SharpMind.Model.Format;

namespace SharpMind.Tests;

public static class QuantizationDiagnostic
{
    public static unsafe void RunDiagnostics()
    {
        Console.WriteLine($"--- Running Quantization Consistency Diagnostic ---");
        
        Console.WriteLine("\n=== HardwareTier.Scalar ===");
        RunForTier(HardwareTier.Scalar);

        Console.WriteLine("\n=== HardwareTier.AVX2 ===");
        RunForTier(HardwareTier.AVX2);

        Console.WriteLine("\n=== HardwareTier.FMA ===");
        RunForTier(HardwareTier.FMA);
    }

    private static unsafe void RunForTier(HardwareTier tier)
    {
        TestType(tier, GgufDtype.Q2_K, "Q2_K", 84);
        TestType(tier, GgufDtype.Q3_K, "Q3_K", 110);
        TestType(tier, GgufDtype.Q4_K, "Q4_K", 144);
        TestType(tier, GgufDtype.Q5_K, "Q5_K", 176);
        TestType(tier, GgufDtype.Q5_1, "Q5_1", 24);
    }

    private static unsafe void TestType(HardwareTier tier, GgufDtype dtype, string name, int blockBytes)
    {
        Console.WriteLine($"Testing {name}...");
        int qk = (dtype == GgufDtype.Q5_1) ? 32 : 256;
        
        // 1. Create a synthetic block with random data
        byte[] block = new byte[blockBytes];
        Random rng = new Random(42);
        for (int i = 0; i < blockBytes; i++) block[i] = (byte)rng.Next(256);
        
        // 2. Compute dot product via Read (dequantize-then-dot)
        float[] dequantized = new float[qk];
        fixed (byte* pBlock = block)
        {
            DequantizeHelper(dtype, block, dequantized, qk);
        }
        
        // Dummy input vector
        float[] input = new float[qk];
        for(int i=0; i<qk; i++) input[i] = 1.0f;
        
        float dotRead = 0;
        for(int i=0; i<qk; i++) dotRead += input[i] * dequantized[i];
        
        // 3. Compute dot product via VecDot
        var qOps = QuantizationFactory.Create(tier);
        float dotVec = 0;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            dotVec = VecDotDispatch(qOps, dtype, pIn, pBlock, 0, qk);
        }
        
        Console.WriteLine($"  Read: {dotRead:F4}, VecDot: {dotVec:F4}, Diff: {Math.Abs(dotRead - dotVec):F4}");
    }

    private static unsafe float VecDotDispatch(QuantizationOps qOps, GgufDtype dtype, float* input, byte* rawWeights, int col, int inFeatures) => dtype switch
    {
        GgufDtype.Q2_K => qOps.VecDotQ2K(input, rawWeights, col, inFeatures),
        GgufDtype.Q3_K => qOps.VecDotQ3K(input, rawWeights, col, inFeatures),
        GgufDtype.Q4_K => qOps.VecDotQ4K(input, rawWeights, col, inFeatures),
        GgufDtype.Q5_K => qOps.VecDotQ5K(input, rawWeights, col, inFeatures),
        GgufDtype.Q5_1 => qOps.VecDotQ5_1(input, rawWeights, col, inFeatures),
        _ => 0
    };

    private static unsafe void DequantizeHelper(GgufDtype dtype, byte[] block, float[] dest, int qk)
    {
        fixed (byte* pBlock = block)
        {
            // For Q5_1, we must manually read it to match the ReadQ5_1 function exactly
            if (dtype == GgufDtype.Q5_1)
            {
                float d = GgufLoader.HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref block[0]));
                float m = GgufLoader.HalfToFloat(Unsafe.ReadUnaligned<ushort>(ref block[2]));
                uint qh = Unsafe.ReadUnaligned<uint>(ref block[4]);
                
                for (int i = 0; i < qk; i++)
                {
                    int xh = (int)((qh >> i) & 1) << 4;
                    int q = ((block[8 + i / 2] >> (4 * (i % 2))) & 0x0F) | xh;
                    dest[i] = q * d + m;
                }
            }
            else
            {
                using var ms = new MemoryStream(block);
                using var reader = new BinaryReader(ms);
                
                switch (dtype)
                {
                    case GgufDtype.Q2_K:
                        GgufLoader.ReadQ2K(reader, dest, qk);
                        break;
                    case GgufDtype.Q3_K:
                        GgufLoader.ReadQ3_K(reader, dest, qk);
                        break;
                    case GgufDtype.Q4_K:
                        GgufLoader.ReadQ4K(reader, dest, qk);
                        break;
                    case GgufDtype.Q5_K:
                        GgufLoader.ReadQ5_K(reader, dest, qk);
                        break;
                    default:
                        throw new NotSupportedException($"Dequantization helper for {dtype} not implemented.");
                }
            }
        }
    }
}
