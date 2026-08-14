using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace SharpMind.Samples.Training;

/// <summary>
/// Quantization-aware training (.SMM) example. Same corpus + pipeline as
/// <see cref="SmmBackpropTextExample"/>, but sets
/// <see cref="TrainConfig.QuantAwareTraining"/> so every linear-layer forward
/// (attention projections, FFN, and the tied LM head) runs through the
/// quantized dtype while gradients flow straight through to the F32 master
/// weights. The saved .SMM therefore loads with that dtype already baked in.
///
/// Choose the target via <see cref="QuantTarget"/>:
/// <list type="bullet">
///   <item>K-quants (<see cref="QuantDType.Q6_K"/>/<see cref="QuantDType.Q4_K"/>/
///   <see cref="QuantDType.Q2_K"/>, etc.) — best quality-per-byte for fake-quant
///   training; require each layer's InFeatures to be a multiple of 128 and the
///   flattened weight length a multiple of 256 (this example's dims qualify:
///   1024×128 tied head, 128/512-wide attention and FFN projections).</item>
///   <item><see cref="QuantDType.Q8_0"/> / <see cref="QuantDType.Q4_0"/> —
///   legacy block formats; require every weight dimension to be a multiple of 32.</item>
///   <item><see cref="QuantDType.F16"/> — always safe on any shape.</item>
///   <item><see cref="QuantDType.F32"/> or null — disables QAT, identical to
///   the plain backprop example.</item>
/// </list>
/// </summary>
public static class SmmQuantAwareTrainingExample
{
    public const string Name = "qat";
    private const string CorpusPath = @"c:\temp\tinyshakespeare.txt";
    private const string TokenizerPath = @"c:\temp\tinyshakespeare.tokenizer.json";
    private const string SavePath = @"c:\temp\qat.smm";

    private const int TargetVocabSize = 1024;
    private const int BatchSize = 8;
    private const int SeqLen = 64;
    private const int MaxContextLen = 512;
    private const int Steps = 300;
    private const int LogInterval = 50;
    private const int Seed = 1234;

    /// <summary>The quantized dtype to fake-quantize forwards to while training.</summary>
    private const QuantDType QuantTarget = QuantDType.Q6_K;

    public static async Task RunAsync()
    {
        await Console.Out.WriteLineAsync($"== SharpMind QAT SMM training example (target: {QuantTarget}) ==");
        await Console.Out.WriteLineAsync();

        // 1. Tokenizer — BPE trained on the corpus, cached on disk (shared with
        //    the plain backprop example, same corpus → same vocabulary).
        Tokenizer tokenizer = File.Exists(TokenizerPath)
            ? TokenizationPipeline.Load(TokenizerPath)
            : await TokenizationPipeline.TrainAndSaveAsync(new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc), TokenizerPath, TargetVocabSize);
        await Console.Out.WriteLineAsync($"Tokenizer: vocab={tokenizer.VocabSize}");

        // 2. Model — every weight dim is a multiple of 32 so block targets
        //    (Q8_0/Q4_0) are layout-correct; InFeatures are multiples of 128 and
        //    flattened lengths multiples of 256 so K-quant targets are correct
        //    for the per-column VecDot sub-scale addressing.
        var modelConfig = new ModelConfig
        {
            VocabSize = tokenizer.VocabSize, // 1024 % 32 == 0, 1024*128 % 256 == 0
            HiddenDim = 128,                 // 128  % 32 == 0 and % 128 == 0
            NumLayers = 4,
            NumHeads = 8,
            NumKvHeads = 8,
            FfnDim = 512,                    // 512  % 32 == 0, % 128 == 0, 512*128 % 256 == 0
            MaxSeqLen = MaxContextLen,
            NormEps = 1e-3f,
        };
        await Console.Out.WriteLineAsync($"Training config: {modelConfig}");

        // 3. Data pipeline — lines → clean → tokenise → packed TrainingBatches.
        var source = new TextFileSource(CorpusPath, TextFileSource.DocumentMode.LinePerDoc);
        var pipeline = PipelineNode.From(source)
            .Pipe(new NormaliseWhitespace())
            .Pipe(new MinLengthFilter(8));
        var batcher = new PackingBatcher(
            batchSize: BatchSize,
            maxSeqLen: SeqLen,
            eosTokenId: tokenizer.EosId,
            padTokenId: tokenizer.PadId);
        var loader = new DataLoader(pipeline, s => tokenizer.Encode(s), batcher, prefetchBuffer: 4);

        // 4. Model — empty float weights, randomised so gradients are meaningful.
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, Seed);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        // 5. Train with AdamW. QuantAwareTraining wires fake-quantization into
        //    every linear layer + the head automatically (TrainLoop calls
        //    model.EnableQuantAwareTraining).
        var parameters = model.Parameters().ToList();
        var ops = TrainingOpsFactory.Create(sharpConfig);
        using var optimizer = new AdamW(parameters, ops, lr: 8e-5f, weightDecay: 0.1f);
        var scheduler = new CosineWithWarmup(maxLr: 8e-5f, minLr: 3e-5f, warmupSteps: 40, decaySteps: 200);
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
                LabelSmoothing = 0.1f,
                QuantAwareTraining = QuantTarget,
            });

        await Console.Out.WriteLineAsync($"Training {Steps} steps with QAT [{QuantTarget}] " +
                                         $"(random-init loss ≈ {MathF.Log(modelConfig.VocabSize):F2})...");
        float lastLoss = float.NaN;
        await loop.RunAsync(onStep: r =>
        {
            lastLoss = r.Loss;
            Console.WriteLine($"step {r.Step,4}/{Steps}: loss = {r.Loss:F4}  gradNorm = {r.GradNorm:F3}");
        });
        await Console.Out.WriteLineAsync($"Final loss = {lastLoss:F4}");
        await Console.Out.WriteLineAsync();

        // 6. Export the trained weights + tokenizer to .SMM. The exporter
        //    quantizes each weight to the target dtype; because training already
        //    fake-quantized forwards to the same dtype, the round trip is exact.
        Directory.CreateDirectory(Path.GetDirectoryName(SavePath)!);
        SmmTrainingExporter.Export(weights, tokenizer, SavePath, new SmmWriteOptions
        {
            Source = "training",
        }, model: model);
        await Console.Out.WriteLineAsync($"Saved: {SavePath} ({new FileInfo(SavePath).Length:N0} bytes)");

        // 7. Reload the .SMM and rebuild the inference transformer.
        using var reloaded = SmmTrainingPipeline.LoadForInference(SavePath, out var reloadedTokenizer, out var reloadedConfig);
        await Console.Out.WriteLineAsync($"Reloaded config: {reloadedConfig}");
        await Console.Out.WriteLineAsync($"Reloaded tokenizer vocab size: {reloadedTokenizer.VocabSize}");
        await Console.Out.WriteLineAsync($"Export dtype per tensor: see .SMM metadata on reload.");
        await Console.Out.WriteLineAsync();
    }
}
