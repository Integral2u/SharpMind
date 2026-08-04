using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Optimizers;

namespace SharpMind.Samples.Training;

/// <summary>
/// End-to-end .SMM training example: trains a tiny Learnable (PseudoLanguage)
/// model, exports it to <c>c:\temp\pseudo.smm</c> via
/// <see cref="SmmTrainingExporter"/>, reloads it via
/// <see cref="SmmTrainingPipeline.LoadForInference"/>, and runs greedy
/// generation on the reloaded weights.
///
/// Everything here is thin orchestration over the SharpMind.Training library:
/// <see cref="WeightInitializer"/>, <see cref="FiniteDifferenceTrainer"/>,
/// <see cref="Evaluator"/>, <see cref="TrainingTokenizerBuilder"/> and
/// <see cref="SmmTrainingPipeline"/>.
/// </summary>
public static class SmmPseudoLanguageExample
{
    public const string Name = "pseudo";
    private const string SaveDir = @"c:\temp";
    private const string SavePath = @$"c:\temp\{Name}.smm";
    private const int Steps = 150;
    private const int GenTokens = 24;

    // ChatML-style Jinja template, embedded in the .SMM on export and read back
    // to build the chat formatter on reload (see step 5b).
    private const string ChatTemplate =
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}";

    public static async Task RunAsync()
    {
        await Console.Out.WriteLineAsync("== SharpMind SMM training example ==");
        await Console.Out.WriteLineAsync();

        // 1. Create the model (tiny Learnable-style config, SIMD kernels) and
        //    randomise the zeroed weights so gradients are meaningful.
        var modelConfig = new ModelConfig
        {
            VocabSize = 64,
            HiddenDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            NumKvHeads = 2,
            FfnDim = 32,
            MaxSeqLen = 512,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, seed: 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        var learnConfig = new LearnableConfig
        {
            BatchSize = 4,
            SeqLen = 3,
            TrainSamples = 400,
            TestSamples = 100,
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
        };
        var generator = new LearnableGenerator(learnConfig, new Random(1234));

        // 2. Train with AdamW over finite-difference gradients. The generator
        //    is adapted into the same TrainingBatch stream a DataLoader yields.
        var batches = generator.ToTrainingBatches(batchSize: 4, seqLen: 3);
        var parameters = model.Parameters().ToList();
        using var optimizer = new AdamW(parameters, lr: 0.02f, weightDecay: 0f);
        var trainer = new FiniteDifferenceTrainer(model, batches, optimizer, parameters: parameters, config: new FiniteDifferenceConfig
        {
            TotalSteps = Steps,
            LogInterval = 20,
        });

        await trainer.TrainAsync(
            progress: new Progress<float>(p => Console.Write($"\rTraining: {p,6:P0}")),
            onStep: r => Console.WriteLine($"step {r.Step,3}/{Steps}: loss = {r.Loss:F4}"));
        Console.WriteLine();

        var evaluator = new Evaluator(model, new SharpMind.Training.Loss.CrossEntropyLoss());
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync($"Next-token accuracy: {evaluator.NextTokenAccuracy(generator, modelConfig.VocabSize):P1}");

        // 3. Build the tokenizer and export the model + chat template to .SMM.
        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(generator, modelConfig.VocabSize);
        Directory.CreateDirectory(SaveDir);
        SmmTrainingExporter.Export(weights, tokenizer, SavePath, new SmmWriteOptions
        {
            Compression = CompressionMode.Auto,
            Source = "training",
        }, chatTemplate: ChatTemplate);
        await Console.Out.WriteLineAsync($"Saved: {SavePath} ({new FileInfo(SavePath).Length:N0} bytes)");

        // 4. Reload the .SMM from disk and rebuild the inference transformer.
        using var reloaded = SmmTrainingPipeline.LoadForInference(SavePath, out var reloadedTokenizer, out var reloadedConfig);
        string? reloadedChatTemplate = SmmTrainingPipeline.LoadChatTemplate(SavePath);

        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync($"Reloaded config: {reloadedConfig}");
        await Console.Out.WriteLineAsync($"Reloaded tokenizer vocab size: {reloadedTokenizer.VocabSize}");
        await Console.Out.WriteLineAsync($"Reloaded chat template: {reloadedChatTemplate}");
        await Console.Out.WriteLineAsync();

        // 5a. Greedy generation on a fresh pseudo-language prompt.
        var probe = new LearnableGenerator(
            new LearnableConfig { IncludeNouns = true, IncludeVerbs = true, IncludeObjects = true },
            new Random(7));
        var prompt = probe.GenerateTrainingSample().TokenIds;
        var ids = SmmTrainingPipeline.GenerateGreedy(reloaded, prompt, reloadedConfig.VocabSize, steps: 3);

        string decoded = reloadedTokenizer.Decode([.. ids], skipSpecials: false);
        await Console.Out.WriteLineAsync($"Prompt ids : [{string.Join(", ", prompt)}]");
        await Console.Out.WriteLineAsync($"Generated  : [{string.Join(", ", ids)}]");
        await Console.Out.WriteLineAsync($"Decoded    : {decoded}");
        await Console.Out.WriteLineAsync();

        // 5b. Chat formatter — built from the chat template stored in the .SMM,
        //     so a real English prompt exercises the stored-template path too.
        var chatFormatter = ChatPromptFormatterFactory.Create(reloadedChatTemplate);
        await Console.Out.WriteLineAsync($"[chat] formatter: {chatFormatter.GetType().Name}");
        await Console.Out.WriteLineAsync("[chat] Q: What happens at the end of the play?");
        await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            reloaded, reloadedTokenizer, null, null, null, null, null, null, null, chatFormatter)
        {
            MaxNewTokens = GenTokens,
            Temperature = 0.7f,
            TopK = 40,
        };
        session.InitializeChat();
        await Console.Out.WriteAsync("  A:");
        await foreach (var entry in session.GetResponseStreamAsync("What happens at the end of the play?"))
        {
            if (entry.Token is not null)
                await Console.Out.WriteAsync(entry.Token);
        }
        await Console.Out.WriteLineAsync();
    }
}
