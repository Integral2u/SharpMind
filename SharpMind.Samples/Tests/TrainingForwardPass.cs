using SharpMind.Core.Tensors;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Samples.Tests;

public static class TrainingForwardPass
{
    public static async void Run()
    {
        await Task.CompletedTask;
        Console.WriteLine("=== Training Test ===");

        var modelConfig = ModelConfig.Tiny;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model params: {model.ParameterCount:N0}");

        var vocabConfig = VocabConfig.Tiny;
        var generator = new PseudoLanguageGenerator(vocabConfig);
        Console.WriteLine($"Vocab size: {generator.VocabSize}");

        var seq = generator.GenerateSyntacticSequences(1).First();
        Console.WriteLine($"Generated: {seq.RawText}");

        var tokens = seq.RawText.Split(' ').Select(w => generator.TextToId(w)).ToArray();
        var padded = new int[8];
        tokens.CopyTo(padded.AsSpan());
        Console.WriteLine($"Tokens: [{string.Join(", ", padded)}]");

        var inputTensor = Tensor<int>.From(padded, 1, 8);
        Console.WriteLine($"Input: {inputTensor.Shape}");

        var logits = model.Forward(inputTensor);
        Console.WriteLine($"Output: {logits.Shape}");

        logits.Dispose();
        Console.WriteLine("Training forward pass successful!");
    }
}
