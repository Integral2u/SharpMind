using SharpMind.Data.Parquet.Sources;
using SharpMind.Data.Sources;
using SharpMind.Training;

namespace SharpMind.Samples.Tests;

public static class AutoSizingTest
{
    public static async Task Run()
    {
        Console.WriteLine("=== Testing Auto-Sizer with Parquet Data ===");
        
        IDataSource source = new ParquetSource(
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\open-perfectblend\train-*.parquet", 
            "source");

        var constraints = new SizingConstraints(
            MinHiddenDim: 16, 
            MaxHiddenDim: 64, 
            MinLayers: 1, 
            MaxLayers: 3);
            
        var budget = new SizingBudget(
            SampleSize: 500, 
            StepsPerConfig: 20);

        var optimalConfig = await ModelSizer.DetermineOptimalConfigAsync(source, constraints, budget);

        Console.WriteLine("\nRecommended Model Configuration:");
        Console.WriteLine($"HiddenDim: {optimalConfig.HiddenDim}");
        Console.WriteLine($"NumLayers: {optimalConfig.NumLayers}");
        Console.WriteLine($"VocabSize: {optimalConfig.VocabSize}");
    }
}
