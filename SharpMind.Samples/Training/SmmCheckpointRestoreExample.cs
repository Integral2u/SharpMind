using System.Diagnostics;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Inference;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Loss;

namespace SharpMind.Samples.Training;

/// <summary>
/// Restores a training checkpoint and re-exports it to a .SMM without retraining.
///
/// After a long run diverges late in the schedule (e.g. tinyshakespeare-backprop
/// at 3,000 steps: gradient norm exploded 5 → 235k from step 1,600 onward and
/// the exported final state scored eval ppl 1435.8), the last <em>healthy</em>
/// state is preserved in a periodic checkpoint. This example loads such a
/// checkpoint back into a freshly built model and writes a clean .SMM, so a
/// diverged run still yields its best-observed weights.
///
/// It is intentionally train-free: no optimizer, no loader loop — just
/// <see cref="Checkpoint.Load"/> (weights + a fresh-pass eval) then
/// <see cref="SmmTrainingExporter.Export"/>. Running it is fast (seconds to
/// minutes), unlike the multi-hour training sample.
///
/// It rebuilds the model with the exact <see cref="ModelConfig"/>/hardware of
/// <see cref="SmmBackpropTextExample"/> so the parameter names/shapes match the
/// checkpoint 1:1 and <see cref="Checkpoint.Load"/> lands values in place.
/// </summary>
public static class SmmCheckpointRestoreExample
{
    public const string Name = "tinyshakespeare-backprop-restore";

    private const string TokenizerPath = @$"c:\temp\tinyshakespeare.tokenizer.json";
    private const string CorpusPath = @$"c:\temp\tinyshakespeare.txt";
    private const string SavePath = @$"c:\temp\tinyshakespeare-backprop.smm";
    private const string CheckpointDir = @"c:\temp\tinyshakespeare-bp-checkpoints";
    private const string RestoreStep = "step-0001400";

    private const int TargetVocabSize = 1024;
    private const int BatchSize = 16;
    private const int SeqLen = 128;
    private const int MaxContextLen = 512;
    private const int GenTokens = 48;
    private const int Seed = 1234;

    // ChatML-style Jinja template (must match the trained example's so the
    // restored .SMM carries the same chat format).
    private const string ChatTemplate =
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}";

    public static async Task RunAsync()
    {
        var timer = new Stopwatch();
        timer.Start();
        await Console.Out.WriteLineAsync("== .SMM checkpoint restore example ==");
        await Console.Out.WriteLineAsync();

        // 1. Tokenizer — must be the same cached tokenizer used by the training run.
        Tokenizer tokenizer = File.Exists(TokenizerPath)
            ? TokenizationPipeline.Load(TokenizerPath)
            : await TokenizationPipeline.TrainAndSaveAsync(new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc), TokenizerPath, TargetVocabSize);
        await Console.Out.WriteLineAsync($"Tokenizer: vocab={tokenizer.VocabSize} " +
                                         $"unk={tokenizer.UnkId} bos={tokenizer.BosId} eos={tokenizer.EosId} pad={tokenizer.PadId}");
        await Console.Out.WriteLineAsync();

        // 2. Model size — must reproduce the training run's fixture exactly so the
        //    checkpoint's parameter names map 1:1 onto the rebuilt parameters.
        var modelConfig = new ModelConfig
        {
            VocabSize = tokenizer.VocabSize,
            HiddenDim = 128,
            NumLayers = 4,
            NumHeads = 8,
            NumKvHeads = 8,
            FfnDim = 512,
            MaxSeqLen = MaxContextLen,
            NormEps = 1e-3f,
        };
        await Console.Out.WriteLineAsync($"Restored model config: {modelConfig}");
        await Console.Out.WriteLineAsync();

        // 3. Model — empty host weights; Checkpoint.Load overwrites their values
        //    in place. The parameter name set comes from model.Parameters().
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var parameters = model.Parameters().ToList();

        // 4. Load the healthy checkpoint into the parameters (in-place → into weights).
        string checkpointPath = Path.Combine(CheckpointDir, RestoreStep);
        var meta = Checkpoint.Load(checkpointPath, parameters);
        await Console.Out.WriteLineAsync($"Loaded checkpoint: {checkpointPath}");
        await Console.Out.WriteLineAsync($"  saved step={meta.Step}  loss={(float.IsNaN(meta.Loss) ? "n/a" : meta.Loss.ToString("F4"))}  note={(string.IsNullOrEmpty(meta.Note) ? "n/a" : meta.Note)}");
        await Console.Out.WriteLineAsync();

        // 5. Fresh-vision eval with a plain cross-entropy loss (no smoothing) —
        //    the true held-out metric; this is the number the diverged final .SMM
        //    never exposed.
        var pipeline = PipelineNode.From(new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc))
            .Pipe(new NormaliseWhitespace())
            .Pipe(new MinLengthFilter(8));
        var tokenise = (string text) => tokenizer.Encode(text);
        var batcher = new PackingBatcher(
            batchSize: BatchSize,
            maxSeqLen: SeqLen,
            eosTokenId: tokenizer.EosId,
            padTokenId: tokenizer.PadId);
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

        // 6. Export the restored weights + tokenizer + chat template to .SMM,
        //    overwriting the diverged final file produced by the training run.
        Directory.CreateDirectory(Path.GetDirectoryName(SavePath)!);
        SmmTrainingExporter.Export(weights, tokenizer, SavePath, new SmmWriteOptions
        {
            Compression = CompressionMode.Auto,
            Source = "training-restore",
        }, chatTemplate: ChatTemplate);
        await Console.Out.WriteLineAsync($"Saved (restored): {SavePath} ({new FileInfo(SavePath).Length:N0} bytes)");
        await Console.Out.WriteLineAsync();

        // 7. Reload and demonstrate quality the same way the training sample does.
        using var reloaded = SmmTrainingPipeline.LoadForInference(SavePath, out var reloadedTokenizer, out var reloadedConfig);
        await Console.Out.WriteLineAsync($"Reloaded config: {reloadedConfig}");
        await Console.Out.WriteLineAsync($"Reloaded tokenizer vocab size: {reloadedTokenizer.VocabSize}");
        await Console.Out.WriteLineAsync();

        await Console.Out.WriteLineAsync("[greedy] 'To be, or not to be' →");
        var prompt = tokenizer.Encode("To be, or not to be");
        int greedySteps = Math.Max(1, Math.Min(GenTokens, reloadedConfig.MaxSeqLen - 1 - prompt.Length));
        var greedyIds = SmmTrainingPipeline.GenerateGreedy(reloaded, prompt, reloadedConfig.VocabSize, steps: greedySteps);
        await Console.Out.WriteLineAsync("  " + reloadedTokenizer.Decode([.. greedyIds]));
        await Console.Out.WriteLineAsync();

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

        await Console.Out.WriteLineAsync($"Restore executed in: {timer.Elapsed.TotalSeconds:F2}s");
        timer.Stop();
    }
}