using SharpMind.Model.Format;
using SharpMind.Model;
using System;
using System.IO;
using System.Linq;

namespace SandBox;

public static class QuickDiagnostic
{
    public static void Run()
    {
        string modelPath = Path.Combine(
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets",
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");

        Console.WriteLine("═══ Quick Q6_K Dequant Diagnostic ═══");

        // Load just output.weight directly
        var meta = GgufLoader.LoadMeta(modelPath);
        var weights = GgufLoader.LoadWeights(modelPath);
        var tensorMap = meta.Tensors.ToDictionary(t => t.Name);

        if (!weights.TryGetValue("output.weight", out var outputWeight))
        {
            Console.WriteLine("output.weight not found!");
            return;
        }

        Console.WriteLine($"output.weight shape: [{outputWeight.Shape[0]}, {outputWeight.Shape[1]}]");
        var data = outputWeight.Data;

        // Token 71486 ("Alright")
        int checkToken = 71486;
        int hiddenDim = 1536;

        var row = data.Slice(checkToken * hiddenDim, hiddenDim);
        float min = float.MaxValue, max = float.MinValue;
        double mean = 0;
        for (int i = 0; i < row.Length; i++)
        {
            float v = row[i];
            if (v < min) min = v;
            if (v > max) max = v;
            mean += v;
        }
        mean /= row.Length;

        Console.WriteLine($"\nToken {checkToken} LM head weight stats:");
        Console.WriteLine($"  min={min:G4} max={max:G4} mean={mean:G6}");
        Console.Write("  First 10: ");
        for (int k = 0; k < 10; k++)
            Console.Write($"{row[k]:F10} ");
        Console.WriteLine();

        // Expected values from Python gguf.dequantize
        float[] expected = new float[] {
            0.022630691528320312f, -0.009429454803466797f, -0.0603485107421875f, -0.024516582489013672f,
            0.024516582489013672f, 0.02640247344970703f,  0.011315345764160156f, -0.0075435638427734375f,
            0.015087127685546875f, 0.0018858909606933594f
        };

        Console.WriteLine("\nComparison with Python gguf.dequantize:");
        int matches = 0;
        for (int i = 0; i < 10; i++)
        {
            bool match = Math.Abs(row[i] - expected[i]) < 0.0001f;
            if (match) matches++;
            Console.WriteLine($"  [{i}] SharpMind={row[i]:F10}  Python={expected[i]:F10}  {(match ? "PASS" : "FAIL")}");
        }
        Console.WriteLine($"\n{matches}/10 match with Python gguf.dequantize");
        Console.WriteLine(matches == 10 ? "✓ Q6_K FIX IS CORRECT!" : "✗ Q6_K FIX STILL WRONG!");
    }
}
