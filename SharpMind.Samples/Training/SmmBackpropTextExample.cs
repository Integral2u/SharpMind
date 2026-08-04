using System.Runtime.CompilerServices;
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
using SharpMind.Training.Schedulers;

namespace SharpMind.Samples.Training;

/// <summary>
/// End-to-end .SMM training example on <em>real text</em> using full backprop.
/// Trains a much larger transformer than the finite-difference baseline
/// (<see cref="SmmRealTextExample"/>) on <c>c:\temp\tinyshakespeare.txt</c>
/// because gradients now come from <see cref="BackpropEngine"/>
/// (one forward + one backward per step, instead of O(parameters) forwards),
/// then exports the weights to <c>c:\temp\tinyshakespeare-backprop.smm</c>,
/// reloads it via <see cref="SmmTrainingPipeline.LoadForInference"/> and
/// demonstrates greedy continuation, temperature sampling, and a
/// <see cref="ChatSession"/> built from the chat template stored in the file.
///
/// The corpus is streamed repeatedly (multi-epoch) so the loop can run the
/// requested number of steps even though <c>tinyshakespeare.txt</c> is small.
///
/// Overnight settings (Option B): H128/L4, Batch 8 x Seq 128, 12,000 steps
/// (~2 s/step ≈ 6-7 h on CPU). Checkpoints are written every 1,000 steps to
/// <c>c:\temp\tinyshakespeare-bp-checkpoints</c>; to resume after an
/// interruption, set <see cref="TrainConfig.ResumeFrom"/> to the latest
/// <c>step-XXXXXXX</c> directory.
/// </summary>
public static class SmmBackpropTextExample
{
    public const string Name = "tinyshakespeare-backprop";
    private const string CorpusPath = @$"c:\temp\tinyshakespeare.txt";
    private const string TokenizerPath = @$"c:\temp\tinyshakespeare.tokenizer.json";
    private const string SavePath = @$"c:\temp\{Name}.smm";
    private const string CheckpointDir = @"c:\temp\tinyshakespeare-bp-checkpoints";

    private const int TargetVocabSize = 1024;
    private const int BatchSize = 8;
    private const int SeqLen = 128;
    private const int MaxContextLen = 512;
    private const int GenTokens = 48;
    private const int Steps = 12_000;
    private const int LogInterval = 200;
    private const int Seed = 1234;

    // ChatML-style Jinja template, embedded in the .SMM on export and read back
    // to build the chat formatter on reload (see step 8 / 9c).
    private const string ChatTemplate =
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}";

    public static async Task RunAsync()
    {
        await Console.Out.WriteLineAsync("== SharpMind real-text backprop SMM training example ==");
        await Console.Out.WriteLineAsync();

        // 1. Tokenizer — BPE trained on the corpus, cached on disk. Shared with
        //    the finite-difference baseline (same corpus → same vocabulary).
        Tokenizer tokenizer = File.Exists(TokenizerPath)
            ? TokenizationPipeline.Load(TokenizerPath)
            : await TokenizationPipeline.TrainAndSaveAsync(new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc), TokenizerPath, TargetVocabSize);
        await Console.Out.WriteLineAsync($"Tokenizer: vocab={tokenizer.VocabSize} " +
                                         $"unk={tokenizer.UnkId} bos={tokenizer.BosId} eos={tokenizer.EosId} pad={tokenizer.PadId}");
        await Console.Out.WriteLineAsync();

        // 2. Model size — a real transformer now that backprop replaces the
        //    O(parameters)-forward finite-difference loop.
        var modelConfig = new ModelConfig
        {
            VocabSize = tokenizer.VocabSize,
            HiddenDim = 128,
            NumLayers = 4,
            NumHeads = 8,
            NumKvHeads = 8,
            FfnDim = 512,
            MaxSeqLen = MaxContextLen,
        };
        await Console.Out.WriteLineAsync($"Training config: {modelConfig}");
        await Console.Out.WriteLineAsync();

        // 3. Data pipeline — lines → clean → tokenise → packed TrainingBatches.
        //    The corpus source repeats forever so the loop can reach `Steps`
        //    regardless of how small tinyshakespeare.txt is.
        var source = new RepeatingDataSource(new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc));
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

        // 5. Train with AdamW over backprop gradients.
        var parameters = model.Parameters().ToList();
        var ops = TrainingOpsFactory.Create(sharpConfig);
        using var optimizer = new AdamW(parameters, ops, lr: 6e-4f, weightDecay: 0.1f);
        var scheduler = new CosineWithWarmup(maxLr: 6e-4f, minLr: 6e-5f, warmupSteps: 800, decaySteps: Steps);
        var loop = new TrainLoop(
            model: model,
            parameters: parameters,
            loader: loader,
            optimizer: optimizer,
            scheduler: scheduler,
            ops: ops,
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = Steps,
                LogInterval = LogInterval,
                GradClipNorm = 1.0f,
                CheckpointInterval = 1000,
                CheckpointDir = CheckpointDir,
            });

        await Console.Out.WriteLineAsync($"Training {Steps} steps (random-init loss ≈ ln({modelConfig.VocabSize}) = {MathF.Log(modelConfig.VocabSize):F2})...");
        float lastLoss = float.NaN;
        var wall = System.Diagnostics.Stopwatch.StartNew();
        await loop.RunAsync(
            onStep: r => { lastLoss = r.Loss; Console.WriteLine($"step {r.Step,5}/{Steps}: loss = {r.Loss:F4}  gradNorm = {r.GradNorm:F3}  {r.StepTime.TotalSeconds:F1}s"); },
            progress: new Progress<float>(p => Console.Write($"\rTraining: {p,6:P0}")));
        wall.Stop();
        Console.WriteLine();
        await Console.Out.WriteLineAsync($"Run summary: {Steps} steps in {wall.Elapsed.TotalMinutes:F1} min, final loss = {lastLoss:F4}");

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
        //     exactly as a real consumer would resolve it.
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

    /// <summary>
    /// Cycles the wrapped source forever so a small corpus can feed arbitrarily
    /// long training runs (each pass re-enumerates the underlying source).
    /// </summary>
    private sealed class RepeatingDataSource(IDataSource inner) : IDataSource
    {
        public long? EstimatedCount => null;

        public string Description => $"{inner.Description} (repeated for multi-epoch training)";

        public async IAsyncEnumerable<string> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                await foreach (var doc in inner.ReadAsync(cancellationToken))
                    yield return doc;
            }
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
