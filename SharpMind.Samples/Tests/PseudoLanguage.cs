using SharpMind.Data.Sources.PseudoLanguage;
namespace SharpMind.Samples.Tests;

public static class PseudoLanguage
{
    public static async void Run()
    {
        Console.WriteLine("=== Pseudo-Language Data Generation ===");
        var gen = new PseudoLanguageGenerator(VocabConfig.Medium);
        var rec = gen.GetModelSizeRecommendation();
        Console.WriteLine($"Vocab: {rec.VocabSize}, Params: {rec.EstimatedParams:N0}");

        Console.WriteLine($"\n=== Pseudo-Language Pipeline ===");
        var pipeline = new PseudoLanguagePipeline(VocabConfig.Medium, ComplexityLevel.Syntactic, 500);
        var loader = pipeline.ToDataLoader(batchSize: 8, maxSeqLen: 64);

        Console.WriteLine("\n=== Loading Batches ===");
        var batchCount = 0;
        await foreach (var batch in loader.LoadAsync())
        {
            batchCount++;
            if (batchCount <= 3)
            {
                var shape = batch.TokenIds.Shape;
                Console.WriteLine($"Batch {batchCount}: [{shape.Rows}, {shape.Cols}]");
            }
        }
        Console.WriteLine($"Total batches: {batchCount}");

        Console.WriteLine("\n=== Data Pipeline Test PASSED ===");
    }
}

