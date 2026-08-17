namespace SharpMind.Inference;

public static class GeneratorDiagnostics
{
    /// <summary>When set, prints top-5 logits with decoded text per step.</summary>
    private static bool dumpTopLogits;

    public static bool DumpTopLogits { get => dumpTopLogits; set => dumpTopLogits = value; }

public static void PrintTopLogits(Tokenization.Tokenizer tokenizer, int step, ReadOnlySpan<float> logitsSlice, TextWriter? writer = null)
    {
        if (!dumpTopLogits) return;        
        writer ??= Console.Error;
        var top5 = new (float Value, int Id)[5];
        for (int i = 0; i < top5.Length; i++) top5[i] = (float.NegativeInfinity, -1);
        double mean = 0, varAcc = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        for (int i = 0; i < logitsSlice.Length; i++)
        {
            float v = logitsSlice[i];
            mean += v;
            if (v < min) min = v;
            if (v > max) max = v;
            if (v > top5[^1].Value)
            {
                top5[^1] = (v, i);
                for (int j = top5.Length - 1; j > 0 && top5[j].Value > top5[j - 1].Value; j--)
                    (top5[j], top5[j - 1]) = (top5[j - 1], top5[j]);
            }
        }
        mean /= logitsSlice.Length;
        for (int i = 0; i < logitsSlice.Length; i++)
        {
            double d = logitsSlice[i] - mean;
            varAcc += d * d;
        }
        double std = Math.Sqrt(varAcc / Math.Max(1, logitsSlice.Length));
        writer.Write($"  [step {step}] n={logitsSlice.Length} mean={mean:G4} std={std:G4} min={min:G4} max={max:G4} margin1={top5[0].Value - top5[1].Value:G4} top5: ");
        foreach (var (val, id) in top5)
        {
            var text = tokenizer.Decode([id], skipSpecials: true);
            writer.Write($"{id}:'{text.Replace("\n", "\\n").Replace("\r", "\\r")}'({val:G4}) ");
        }
        writer.WriteLine();
    }
}
