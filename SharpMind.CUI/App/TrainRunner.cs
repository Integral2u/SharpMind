using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Metadata;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace SharpMind.CUI.App;

/// <summary>Result of a full training run.</summary>
public sealed class TrainRunResult
{
    public bool Success { get; init; }
    public string? Error { get; init; }
    public string? ExportPath { get; init; }
    public int FinalStep { get; init; }
}

/// <summary>
/// Runs one training job end-to-end: rebuilds the source + stage chain from
/// <see cref="TrainJobSettings"/>, trains (or loads) the BPE tokenizer,
/// constructs the model, runs <see cref="TrainLoop"/> with checkpoints and QAT,
/// exports each retained checkpoint to a testable .smm, and finally exports the
/// trained model. Must be called from a background task; the caller marshals
/// progress callbacks onto the UI thread.
/// </summary>
public static class TrainRunner
{
    public const int Seed = 1234;

    /// <summary>
    /// Runs <paramref name="job"/> off the UI thread. <paramref name="status"/>
    /// receives free-form log lines; <paramref name="progress"/> receives a 0..1
    /// overall completion figure; <paramref name="onStep"/> receives each logged
    /// optimizer step. Returns the export path on success.
    /// </summary>
    public static async Task<TrainRunResult> RunAsync(
        TrainJobSettings job,
        string pluginsFolder,
        IProgress<string> status,
        IProgress<float> progress,
        Action<TrainStepResult>? onStep = null,
        CancellationToken cancellationToken = default)
    {
        void Log(string line) => status.Report(line);

        try
        {
            Log("Resolving data pipeline…");
            var components = ComponentRegistry.ScanFolder(pluginsFolder, out var pluginWarnings);
            foreach (var w in pluginWarnings) Log($"Plugin: {w}");

            if (job.Sources.Count == 0)
                throw new InvalidOperationException("No data sources configured for this job.");
            var sources = job.Sources.Select(s => BuildSource(s.Component, components)).ToList();
            var sourceNodes = new List<PipelineNode>();
            for (int i = 0; i < job.Sources.Count; i++)
            {
                var node = PipelineNode.From(sources[i]);
                foreach (var stage in job.Sources[i].Stages)
                {
                    var descriptor = ComponentRegistry.Find(stage.TypeName, components)
                        ?? throw new InvalidOperationException($"Unknown stage '{stage.TypeName}'.");
                    node = node.Pipe((ICleaningStage)ComponentRegistry.Build<ICleaningStage>(descriptor, stage.Args));
                }
                sourceNodes.Add(node);
            }
            PipelineNode pipeline = sourceNodes.Count == 1
                ? sourceNodes[0]
                : CleaningPipeline.Merge(sourceNodes);
            foreach (var g in job.GlobalStages)
            {
                var descriptor = ComponentRegistry.Find(g.TypeName, components)
                    ?? throw new InvalidOperationException($"Unknown stage '{g.TypeName}'.");
                pipeline = pipeline.Pipe((ICleaningStage)ComponentRegistry.Build<ICleaningStage>(descriptor, g.Args));
            }

            var compositeSource = job.Sources.Count == 1 ? sources[0] : new CompositeSource(sources);
            Log($"Sources: {string.Join(", ", sources.Select(s => s.Description))}");
            progress.Report(0.05f);

            // 1. Tokenizer — BPE trained on the corpus, cached on disk.
            string tokenizerPath = job.TokenizerCachePath
                ?? Path.Combine(TrainJobSettings.DefaultFolder, Sanitize(job.Name) + ".tokenizer.json");
            var tokenizer = File.Exists(tokenizerPath)
                ? TokenizationPipeline.Load(tokenizerPath)
                : await TokenizationPipeline.TrainAndSaveAsync(compositeSource, tokenizerPath, job.TokenizerVocabSize);
            Log($"Tokenizer: vocab={tokenizer.VocabSize}");
            progress.Report(0.1f);

            // 2. Model config.
            var modelConfig = new ModelConfig
            {
                VocabSize = tokenizer.VocabSize,
                HiddenDim = job.HiddenDim,
                NumLayers = job.NumLayers,
                NumHeads = job.NumHeads,
                NumKvHeads = job.NumKvHeads,
                FfnDim = job.FfnDim,
                MaxSeqLen = job.MaxSeqLen,
                NormEps = job.NormEps,
            };
            Log($"Model: H={job.HiddenDim} L={job.NumLayers} heads={job.NumHeads} ffn={job.FfnDim}");

            // 3. Data pipeline — clean → tokenise → packed TrainingBatches.
            var batcher = new PackingBatcher(
                batchSize: job.BatchSize,
                maxSeqLen: job.SeqLen,
                eosTokenId: tokenizer.EosId,
                padTokenId: tokenizer.PadId);
            var loader = new DataLoader(pipeline, s => tokenizer.Encode(s), batcher, prefetchBuffer: 4,
                maxBatches: job.TotalSteps * job.GradAccumSteps);
            Log($"Data pipeline: {loader.Describe()}");

            // 4. Model — empty float weights, randomised unless resuming.
            var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
            var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
            if (string.IsNullOrWhiteSpace(job.ResumeFrom))
            {
                WeightInitializer.InitializeRandomly(weights, Seed);
                Log("Weights initialised.");
            }
            using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
            var parameters = model.Parameters().ToList();
            var ops = TrainingOpsFactory.Create(sharpConfig);

            // 5. Optimizer + scheduler + loop.
            using var optimizer = new AdamW(parameters, ops, lr: job.LearningRate, weightDecay: job.WeightDecay);
            var scheduler = new CosineWithWarmup(
                maxLr: job.LearningRate, minLr: job.MinLr,
                warmupSteps: job.WarmupSteps, decaySteps: job.TotalSteps);

            var checkpointDir = job.CheckpointDir;
            Directory.CreateDirectory(checkpointDir);

            var qat = ParseQat(job.QuantAwareTraining);
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
                    TotalSteps = job.TotalSteps,
                    GradAccumSteps = job.GradAccumSteps,
                    GradClipNorm = job.GradClipNorm,
                    LabelSmoothing = job.LabelSmoothing,
                    LogInterval = Math.Max(1, job.LogInterval),
                    CheckpointInterval = job.CheckpointInterval,
                    CheckpointDir = checkpointDir,
                    ResumeFrom = job.ResumeFrom,
                    KeepRecent = job.KeepRecent,
                    QuantAwareTraining = qat,
                });

            Log(qat is { } q
                ? $"Training {job.TotalSteps} steps with QAT [{q}] — full backprop (loss → gradients → AdamW)…"
                : $"Training {job.TotalSteps} steps (float32) — full backprop (loss → gradients → AdamW)…");
            progress.Report(0.12f);

            int lastStep = 0;
            await loop.RunAsync(
                onStep: r =>
                {
                    lastStep = r.Step;
                    onStep?.Invoke(r);
                },
                progress: new Progress<float>(p => progress.Report(0.12f + 0.78f * p)),
                cancellationToken: cancellationToken,
                onCheckpoint: dir => ExportCheckpoint(dir, job, modelConfig, sharpConfig, tokenizer, Log));

            // 6. Export the trained weights + tokenizer to .SMM.
            string exportPath = job.ExportFilePath;
            Directory.CreateDirectory(Path.GetDirectoryName(exportPath)!);
            SmmTrainingExporter.Export(weights, tokenizer, exportPath, BuildEmbedOptions(job, "training"));
            progress.Report(1f);
            Log($"Saved: {exportPath} ({new FileInfo(exportPath).Length:N0} bytes)");

            return new TrainRunResult { Success = true, ExportPath = exportPath, FinalStep = lastStep };
        }
        catch (OperationCanceledException)
        {
            Log("Training interrupted.");
            return new TrainRunResult { Success = false, Error = "cancelled" };
        }
        catch (Exception ex)
        {
            Log($"Training failed: {ex.Message}");
            return new TrainRunResult { Success = false, Error = ex.Message };
        }
    }

    private static IDataSource BuildSource(JobComponent component, IReadOnlyList<ComponentDescriptor> registry)
    {
        if (component is null)
            throw new InvalidOperationException("No data source configured for this job.");
        var descriptor = ComponentRegistry.Find(component.TypeName, registry)
            ?? throw new InvalidOperationException($"Unknown data source '{component.TypeName}'.");
        return (IDataSource)ComponentRegistry.Build<IDataSource>(descriptor, component.Args);
    }

    /// <summary>Exports a saved checkpoint directory as a testable .smm inside it.</summary>
    private static void ExportCheckpoint(
        string dir, TrainJobSettings job, ModelConfig config, SharpMindConfig sharpConfig, Tokenizer tokenizer, Action<string> log)
    {
        try
        {
            var weights = ModelFactory.CreateForTraining(config, sharpConfig);
            using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
            var parameters = model.Parameters().ToList();
            var meta = Checkpoint.Load(dir, parameters, optimizer: null);

            string smmPath = Path.Combine(dir, "model.smm");
            SmmTrainingExporter.Export(weights, tokenizer, smmPath, BuildEmbedOptions(job, "checkpoint"));
            log($"Checkpoint {Path.GetFileName(dir)} → {Path.GetFileName(smmPath)} (step {meta.Step})");
        }
        catch (Exception ex)
        {
            log($"Checkpoint export failed for {dir}: {ex.Message}");
        }
    }

    /// <summary>
    /// Builds the <see cref="SmmWriteOptions"/> that embed the job's configured
    /// system prompt file, skills folder, and plugin DLLs into the .smm. Files
    /// are read eagerly, so missing input at export time fails the run loudly
    /// instead of silently producing an unannotated model.
    /// </summary>
    private static SmmWriteOptions BuildEmbedOptions(TrainJobSettings job, string source)
    {
        string? systemPrompt = null;
        if (!string.IsNullOrWhiteSpace(job.SystemPromptPath))
        {
            if (!File.Exists(job.SystemPromptPath))
                throw new InvalidOperationException($"System prompt file not found: {job.SystemPromptPath}");
            systemPrompt = File.ReadAllText(job.SystemPromptPath).Trim();
        }

        List<string>? skills = null;
        if (!string.IsNullOrWhiteSpace(job.SkillsFolder))
        {
            if (!Directory.Exists(job.SkillsFolder))
                throw new InvalidOperationException($"Skills folder not found: {job.SkillsFolder}");
            skills = Directory.GetFiles(job.SkillsFolder, "*.md", SearchOption.TopDirectoryOnly)
                .OrderBy(f => Path.GetFileName(f))
                .Select(File.ReadAllText)
                .ToList();
            if (skills.Count == 0)
                throw new InvalidOperationException($"No *.md skill files found in: {job.SkillsFolder}");
        }

        List<SmmPluginEntry>? plugins = null;
        if (job.PluginDllPaths is { Count: > 0 })
        {
            plugins = [];
            foreach (var dll in job.PluginDllPaths)
            {
                if (!File.Exists(dll))
                    throw new InvalidOperationException($"Plugin DLL not found: {dll}");
                plugins.Add(new SmmPluginEntry { Name = Path.GetFileName(dll), AssemblyBytes = File.ReadAllBytes(dll) });
            }
        }

        return new SmmWriteOptions
        {
            Source = source,
            Outputs = plugins is { Count: > 0 } ? SmmOutputs.Default | SmmOutputs.Plugins : SmmOutputs.Default,
            SystemPrompt = systemPrompt,
            Skills = skills,
            Plugins = plugins,
        };
    }

    private static QuantDType? ParseQat(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        if (raw.Equals("F32", StringComparison.OrdinalIgnoreCase)) return null;
        return Enum.TryParse<QuantDType>(raw, ignoreCase: true, out var q) ? q : null;
    }

    private static string Sanitize(string name)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return string.IsNullOrWhiteSpace(name) ? "job" : name.Trim();
    }
}