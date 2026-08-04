using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;

namespace SharpMind.Samples.Training;

/// <summary>
/// End-to-end .SMM training example on <em>real text</em>: trains a tiny
/// transformer on <c>c:\temp\tinyshakespeare.txt</c> using the library's
/// data pipeline (BPE tokenizer + <see cref="TextFileSource"/> +
/// <see cref="DataLoader"/>), exports it to
/// <c>c:\temp\tinyshakespeare.smm</c>, reloads it via
/// <see cref="SmmTrainingPipeline.LoadForInference"/>, then demonstrates the
/// full "user prompt → response" story: greedy continuation, temperature
/// sampling, and a <see cref="ChatSession"/> whose formatter is built from
/// the chat template that lives in the .SMM file itself.
///
/// The model is pinned to a deliberately tiny config because gradients are
/// computed by <see cref="FiniteDifferenceTrainer"/> (O(parameters) forwards
/// per step); <see cref="ModelSizer.DetermineOptimalConfigAsync"/> can size it
/// automatically once backprop replaces the finite-difference loop.
/// Output is therefore crude but recognisably language-like — a pipeline demo,
/// not a capable model. Full backprop through the transformer is not yet wired
/// in the training loops, which is the next step toward real-scale training.
/// </summary>
public static class SmmRealTextExample
{
    public const string Name = "tinyshakespeare";
    private const string CorpusPath = @$"c:\temp\{Name}.txt";
    private const string TokenizerPath = @$"c:\temp\{Name}.tokenizer.json";
    private const string SavePath = @$"c:\temp\{Name}.smm";

    private const int TargetVocabSize = 1024;
    private const int BatchSize = 1;
    private const int SeqLen = 16;
    private const int MaxContextLen = 256;
    private const int GenTokens = 48;
    private const int Steps = 120;
    private const int LogInterval = 20;
    private const int Seed = 1234;

    // ChatML-style Jinja template, embedded in the .SMM on export and read back
    // to build the chat formatter on reload (see step 8 / 9c).
    private const string ChatTemplate =
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}";

    public static async Task RunAsync()
    {
        await Console.Out.WriteLineAsync("== SharpMind real-text SMM training example ==");
        await Console.Out.WriteLineAsync();

        var source = new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc);

        // 1. Tokenizer — BPE trained on the corpus, cached on disk.
        Tokenizer tokenizer = File.Exists(TokenizerPath)
            ? TokenizationPipeline.Load(TokenizerPath)
            : await TokenizationPipeline.TrainAndSaveAsync(source, TokenizerPath, TargetVocabSize);
        await Console.Out.WriteLineAsync($"Tokenizer: vocab={tokenizer.VocabSize} " +
                                         $"unk={tokenizer.UnkId} bos={tokenizer.BosId} eos={tokenizer.EosId} pad={tokenizer.PadId}");
        await Console.Out.WriteLineAsync();

        // 2. Model size — pinned to a tiny config so the finite-difference run
        //    is a bounded baseline. FD costs O(elements × forward) per step, so
        //    each parameter element re-runs the full (vocab-1024) forward; at
        //    H8/L1 this measures ~8.5s/step on this machine. The next step after
        //    backprop is wired is to let ModelSizer.DetermineOptimalConfigAsync
        //    (already fixed to the training-forward cost model) pick this for us.
        var modelConfig = new ModelConfig
        {
            VocabSize = tokenizer.VocabSize,
            HiddenDim = 8,
            NumLayers = 1,
            NumHeads = 2,
            NumKvHeads = 2,
            FfnDim = 16,
            MaxSeqLen = MaxContextLen,
        };
        await Console.Out.WriteLineAsync($"Training config: {modelConfig}");
        await Console.Out.WriteLineAsync();

        // 3. Data pipeline — lines → clean → tokenise → packed TrainingBatches.
        var pipeline = PipelineNode.From(source)
            .Pipe(new NormaliseWhitespace())
            .Pipe(new MinLengthFilter(8));
        var tokenise = (string text) => tokenizer.Encode(text);
        var batcher = new PackingBatcher(
            batchSize: BatchSize,
            maxSeqLen: SeqLen,
            eosTokenId: tokenizer.EosId,
            padTokenId: tokenizer.PadId);
        var loader = new DataLoader(pipeline, tokenise, batcher, prefetchBuffer: 4);
        await Console.Out.WriteLineAsync($"Data pipeline: {loader.Describe()}");
        await Console.Out.WriteLineAsync();

        // 4. Model — empty float weights, randomised so gradients are meaningful.
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, Seed);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        // 5. Train with AdamW over finite-difference gradients.
        var parameters = model.Parameters().ToList();
        using var optimizer = new AdamW(parameters, lr: 0.01f, weightDecay: 0f);
        var trainer = new FiniteDifferenceTrainer(model, loader.LoadAsync(), optimizer, parameters: parameters, config: new FiniteDifferenceConfig
        {
            TotalSteps = Steps,
            LogInterval = LogInterval,
        });

        await Console.Out.WriteLineAsync($"Training {Steps} steps (random-init loss ≈ ln({modelConfig.VocabSize}) = {MathF.Log(modelConfig.VocabSize):F2})...");
        await trainer.TrainAsync(
            progress: new Progress<float>(p => Console.Write($"\rTraining: {p,6:P0}")),
            onStep: r => Console.WriteLine($"step {r.Step,3}/{Steps}: loss = {r.Loss:F4}"));
        Console.WriteLine();

        // 6. Held-out sanity eval (fresh pass over the corpus).
        var evalLoader = new DataLoader(pipeline, tokenise, batcher, prefetchBuffer: 4);
        var evalBatches = new List<TrainingBatch>();
        await foreach (var batch in evalLoader.LoadAsync())
        {
            evalBatches.Add(batch);
            if (evalBatches.Count >= 5) break;
        }
        try
        {
            var (evalLoss, perplexity) = new Evaluator(model, new CrossEntropyLoss()).Evaluate(evalBatches);
            await Console.Out.WriteLineAsync($"Eval (fresh pass): loss = {evalLoss:F4}  perplexity = {perplexity:F1}");
        }
        finally
        {
            foreach (var batch in evalBatches) batch.Dispose();
        }
        await Console.Out.WriteLineAsync();

        // 7. Export the trained weights + tokenizer + chat template to .SMM.
        Directory.CreateDirectory(Path.GetDirectoryName(SavePath)!);
        SmmTrainingExporter.Export(weights, tokenizer, SavePath, new SmmWriteOptions
        {
            Compression = CompressionMode.Auto,
            Source = "training",
        }, chatTemplate: ChatTemplate);
        await Console.Out.WriteLineAsync($"Saved: {SavePath} ({new FileInfo(SavePath).Length:N0} bytes)");
        await Console.Out.WriteLineAsync();

        // 8. Reload the .SMM from disk and rebuild the inference transformer.
        using var reloaded = SmmTrainingPipeline.LoadForInference(SavePath, out var reloadedTokenizer, out var reloadedConfig);
        string? reloadedChatTemplate = SmmTrainingPipeline.LoadChatTemplate(SavePath);
        await Console.Out.WriteLineAsync($"Reloaded config: {reloadedConfig}");
        await Console.Out.WriteLineAsync($"Reloaded tokenizer vocab size: {reloadedTokenizer.VocabSize}");
        await Console.Out.WriteLineAsync($"Reloaded chat template: {reloadedChatTemplate}");
        await Console.Out.WriteLineAsync();

        // 9a. Greedy continuation (windowed to the model's MaxSeqLen, so the
        //     prompt + generated tokens stay in the RoPE window).
        await Console.Out.WriteLineAsync("[greedy] 'To be, or not to be' →");
        var prompt = tokenizer.Encode("To be, or not to be");
        int greedySteps = Math.Max(1, Math.Min(GenTokens, reloadedConfig.MaxSeqLen - 1 - prompt.Length));
        var greedyIds = SmmTrainingPipeline.GenerateGreedy(reloaded, prompt, reloadedConfig.VocabSize, steps: greedySteps);
        await Console.Out.WriteLineAsync("  " + reloadedTokenizer.Decode([.. greedyIds]));
        await Console.Out.WriteLineAsync();

        // 9b. Temperature-sampled continuation.
        await Console.Out.WriteLineAsync("[sampled] 'Wherefore art thou Romeo' →");
        using var generator = new StandardGenerator<KVCacherBuilder>(
            reloaded, reloadedTokenizer, addBos: false, addEos: false, seed: Seed);
        var sampling = new SamplingConfig { Temperature = 0.7f, TopK = 40, TopP = 0.9f };
        var generation = GenerationConfig.Completion with { MaxNewTokens = GenTokens, RepetitionPenalty = 1.1f };
        await Console.Out.WriteAsync("  ");
        await foreach (var fragment in generator.GenerateAsync("Wherefore art thou Romeo", sampling, generation))
            await Console.Out.WriteAsync(fragment);
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync();

        // 9c. Chat formatter — built from the chat template stored in the .SMM,
        //     exactly as a real consumer would resolve it, so a user prompt
        //     produces an (crude) response.
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
