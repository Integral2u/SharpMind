using SharpMind.Core.Quantization;
using SharpMind.Model.Format;

namespace SharpMind.Samples.Quantization;

/// <summary>
/// Quantizing an existing .SMM model example: re-encodes every weight tensor
/// to a smaller dtype (K-quants / F16) via <see cref="SmmQuantizer"/>, leaving
/// the source untouched, writing a fresh <c>-Q4_K.smm</c> file alongside it.
///
/// Demonstrates:
/// <list type="bullet">
///   <item>A single fixed target (<see cref="SmmQuantizer.Quantize(string,string,QuantDType,CancellationToken,IProgress{float})"/>)
///   — one line to shrink a model.</item>
///   <item>A per-role plan (<see cref="SmmQuantOptions"/> + <see cref="SmmQuantizer.Quantize(string,string,SmmQuantOptions,CancellationToken,IProgress{float})"/>)
///   — keep the context-sensitive embedding / output at Q8_K while packing the
///   bulky attention + FFN projections down to Q4_K.</item>
///   <item>A byte budget (<see cref="SmmQuantOptions.TargetBytes"/>) — the
///   planner picks the coarsest dtypes that keep the whole file under budget.</item>
///   <item>A size comparison and a reload sanity check on the quantized file.</item>
/// </list>
///
/// Norms, biases, router and unknown tensors are always kept at F16 by the
/// engine (small, sensitive, rarely shape-aligned); K-quant targets give up
/// silently-missing shape errors by falling back to F16 per tensor when a
/// target cannot encode its shape.
/// </summary>
public static class SmmQuantizeModelExample
{
    public const string Name = "smm-quantize";

    /// <summary>Source .SMM to shrink. Point this at any trained/downloaded model.</summary>
    private const string SourcePath = @"c:\temp\my-model.smm";

    public static async Task RunAsync()
    {
        if (!File.Exists(SourcePath))
        {
            await Console.Out.WriteLineAsync($"Quantize example — source not found: {SourcePath}");
            await Console.Out.WriteLineAsync("Point SourcePath at any .SMM (e.g. one produced by the training examples).");
            return;
        }

        string outputRoot = Path.Combine(Path.GetDirectoryName(SourcePath) ?? @".", "quantized");
        Directory.CreateDirectory(outputRoot);
        long sourceBytes = new FileInfo(SourcePath).Length;

        await Console.Out.WriteLineAsync($"== SharpMind .SMM quantization example ==");
        await Console.Out.WriteLineAsync($"Source : {SourcePath}");
        await Console.Out.WriteLineAsync($"Size   : {FormatBytes(sourceBytes)}");
        await Console.Out.WriteLineAsync();

        // 1. Single fixed target — the whole model to Q4_K.
        string q4kPath = Path.Combine(outputRoot, Path.GetFileNameWithoutExtension(SourcePath) + "-Q4_K.smm");
        SmmQuantizer.Quantize(SourcePath, q4kPath, QuantDType.Q4_K, progress: new Progress<float>(
            p => Console.Write($"\r  Q4_K {p * 100:F0}%")));
        Console.WriteLine();

        // 2. Per-role plan — embedding/LM head stay accurate, the big projections pack tight.
        string rolePath = Path.Combine(outputRoot, Path.GetFileNameWithoutExtension(SourcePath) + "-blend.smm");
        SmmQuantizer.Quantize(SourcePath, rolePath, new SmmQuantOptions
        {
            DefaultLevel = QuantDType.Q4_K,
            RoleLevels = new Dictionary<SmmTensorRole, QuantDType>
            {
                [SmmTensorRole.Embedding] = QuantDType.Q8_K,
                [SmmTensorRole.LmHead] = QuantDType.Q8_K,
                [SmmTensorRole.Attention] = QuantDType.Q4_K,
                [SmmTensorRole.Ffn] = QuantDType.Q4_K,
                [SmmTensorRole.Expert] = QuantDType.Q4_K,
            },
        });

        // 3. Byte budget — aim for ~40% of the current size, floor Q4_K.
        long budget = sourceBytes * 4 / 10;
        string budgetPath = Path.Combine(outputRoot, Path.GetFileNameWithoutExtension(SourcePath) + "-budget.smm");
        SmmQuantizer.Quantize(SourcePath, budgetPath, new SmmQuantOptions
        {
            TargetBytes = budget,
            Floor = QuantDType.Q4_K,
        });

        // 4. Sanity check: did the quantized file reload, and what dtypes are written?
        var index = SmmLoader.ReadTensorIndex(rolePath);
        var dtypeCount = index.GroupBy(t => t.Dtype).OrderBy(g => g.Key)
            .Select(g => $"{g.Key}: {g.Count()}");
        await Console.Out.WriteLineAsync($"Tensors on the blend model   : {index.Count} ({string.Join(", ", dtypeCount)})");

        // 5. Size comparison.
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync("Size comparison:");
        OutputLine("Source", sourceBytes, sourceBytes);
        OutputLine("Q4_K (whole model)", new FileInfo(q4kPath).Length, sourceBytes);
        OutputLine("Blend (emb/head Q8_K, rest Q4_K)", new FileInfo(rolePath).Length, sourceBytes);
        OutputLine($"Budget (~{(double)budget / 1_048_576:F1} MB)", new FileInfo(budgetPath).Length, sourceBytes);
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync($"Quantized files: {outputRoot}");

        static void OutputLine(string label, long size, long baseline)
            => Console.WriteLine($"  {label,-36} {FormatBytes(size),10}  {100.0 * size / Math.Max(1, baseline),7:F1}% of source");
    }

    private static string FormatBytes(long bytes)
        => bytes >= 1 << 20 ? $"{bytes / (double)(1 << 20):F2} MiB"
           : bytes >= 1 << 10 ? $"{bytes / (double)(1 << 10):F1} KiB"
           : $"{bytes} B";
}