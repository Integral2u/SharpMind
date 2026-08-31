using SharpMind.Core;
using SharpMind.Core.Plugins;
using SharpMind.Core.Quantization;
using SharpMind.Core.Training;
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
using SharpMind.Training.Loss;
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

    /// <summary>
    /// True when an incremental run found no new or changed files to train on;
    /// the run never entered the training loop and produced no export.
    /// </summary>
    public bool NothingToTrain { get; init; }

    /// <summary>
    /// Set when an accelerator was explicitly requested (its sanity-checked name in
    /// <see cref="AcceleratorRequested"/>) but its engine factory declined — e.g. the ILGPU plugin was
    /// chosen but no CUDA/OpenCL device is present. The CUI turns this into a consent picker (CPU +
    /// every other capable plugin, from <see cref="AcceleratorReason"/> as the why) rather than failing
    /// the run outright. Never set alongside <see cref="Success"/>.
    /// </summary>
    public string? AcceleratorRequested { get; init; }
    public string? AcceleratorReason { get; init; }
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
        Action<int>? onResume = null,
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

            // Incremental mode decides which sources (and which files within a
            // file-based source) feed this run. The plan carries, per source, the
            // rebuilt IDataSource restricted to its new+changed files; unchanged
            // sources contribute no node at all. Deltas are computed by diffing
            // the current per-file hashes against the previous run's map.
            // "Force resume latest" is applied later in the resume-resolution.
            if (job.IncrementalMode && job.SourceFileHashes.Count == 0)
            {
                var persisted = IncrementalStore.Load(job);
                if (persisted.Count > 0)
                {
                    job.SourceFileHashes = persisted;
                    Log($"Incremental: loaded {persisted.Count} source hash map(s) recorded by a previous run.");
                }
            }

            var plan = IncrementalPlanner.Build(job, components, Log);
            if (plan.NothingToTrain)
            {
                Log($"Incremental: no new or changed files since the last run — nothing to train.");
                return new TrainRunResult { Success = true, NothingToTrain = true };
            }

            var sources = plan.Sources;
            var sourceNodes = new List<PipelineNode>();
            for (int i = 0; i < job.Sources.Count; i++)
            {
                if (plan.SkipSource[i]) continue;
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

            var compositeSource = sources.Count == 1 ? sources[0] : new CompositeSource(sources);
            Log($"Sources: {string.Join(", ", sources.Select(s => s.Description))}");

            // Source fingerprints: record what this job trains on, warn when it
            // has changed since the last run (the tokenizer cache is reused on
            // path alone, so a stale corpus would silently poison the mapping),
            // and remember them for stamping the exported model's metadata.
            var sourceHashes = SourceHasher.Compute(job.Sources);
            var changed = job.SourceHashes.Count > 0
                && sourceHashes.Count > 0
                && !job.SourceHashes.OrderBy(kv => kv.Key).SequenceEqual(sourceHashes.OrderBy(kv => kv.Key));
            foreach (var kv in sourceHashes.Where(kv => kv.Value is not null))
                Log($"Source hash {kv.Key}: {kv.Value![..Math.Min(12, kv.Value == null ? 0 : kv.Value.Length)]}…");
            if (changed)
                Log("WARNING: training sources changed since the last run — the cached tokenizer may be stale for this corpus.");
            if (sourceHashes.Count > 0)
                job.SourceHashes = sourceHashes;

            progress.Report(0.05f);

            // 1. Tokenizer — BPE trained on the corpus and cached on disk, or a
            // character-level tokenizer derived from the corpus characters. The
            // vocab must always reflect the FULL corpus, never just the deltas of
            // an incremental run, so when the cache is missing we train it over the
            // complete configured sources (a no-op for fresh jobs, where the
            // deltas ARE the whole corpus anyway).
            Tokenizer tokenizer;
            if (job.UsesCharacterTokenizer)
            {
                tokenizer = await TokenizationPipeline.TrainCharacterAsync(
                    job.IncrementalMode ? FullCorpusComposite(job, components) : compositeSource);
            }
            else
            {
                string tokenizerPath = job.TokenizerCachePath
                    ?? Path.Combine(TrainJobSettings.DefaultFolder, Sanitize(job.Name) + ".tokenizer.json");
                tokenizer = File.Exists(tokenizerPath)
                    ? TokenizationPipeline.Load(tokenizerPath)
                    : await TokenizationPipeline.TrainAndSaveAsync(
                        job.IncrementalMode ? FullCorpusComposite(job, components) : compositeSource,
                        tokenizerPath, job.TokenizerVocabSize);
            }
            Log($"Tokenizer: vocab={tokenizer.VocabSize} ({(job.UsesCharacterTokenizer ? "char" : "bpe")})");
            progress.Report(0.1f);

            // 2. Model config.
            var modelConfig = TrainingModelOptions.ResolveModelConfig(job, tokenizer.VocabSize);
            Log($"Model: H={job.HiddenDim} L={job.NumLayers} heads={job.NumHeads} ffn={job.FfnDim}");

            // 3. Data pipeline — clean → tokenise → TrainingBatches. Random-window
            // batching samples contiguous windows from one flat corpus stream
            // (nanoGPT-style); PackingBatcher is the default document packer.
            IBatchStrategy batcher = job.UsesRandomWindowBatching
                ? new RandomWindowBatcher(batchSize: job.BatchSize, seqLen: job.SeqLen, seed: Seed)
                : new PackingBatcher(
                    batchSize: job.BatchSize,
                    maxSeqLen: job.SeqLen,
                    eosTokenId: tokenizer.EosId,
                    padTokenId: tokenizer.PadId);
            var loader = new DataLoader(pipeline, s => tokenizer.Encode(s), batcher, prefetchBuffer: 4,
                maxBatches: job.TotalSteps * job.GradAccumSteps);
            Log($"Data pipeline: {loader.Describe()}");

            // 4. Model — empty float weights, randomised unless resuming.
            var sharpConfig = TrainingModelOptions.ResolveSharpConfig(job);

            // Resume resolution: an explicit ResumeFrom wins; otherwise if any
            // checkpoint exists under the job's derived checkpoint folder we
            // auto-resume the latest one, so an interrupted run can always be
            // picked back up without re-entering the path manually. StartFresh
            // forces a from-scratch run regardless of what exists on disk.
            // Incremental runs ALWAYS continue from the newest checkpoint —
            // "force resume latest" — ignoring StartFresh/ResumeFrom, because
            // the delta pipeline only delivers unseen data: building on top of
            // the latest weights is the entire point.
            string checkpointDir = job.CheckpointDir;
            string? resumeDir;
            if (job.IncrementalMode)
            {
                resumeDir = Checkpoint.FindLatest(checkpointDir);
                if (resumeDir is null)
                    Log("Incremental: no prior checkpoint found — starting from random weights on the new/changed files only.");
            }
            else
            {
                resumeDir = job.StartFresh ? null
                    : !string.IsNullOrWhiteSpace(job.ResumeFrom) ? job.ResumeFrom
                    : Checkpoint.FindLatest(checkpointDir);
            }
            if (resumeDir is not null)
            {
                job.ResumeFrom = resumeDir;
                var resumeMeta = Checkpoint.ReadMeta(resumeDir);
                Log($"Resuming from checkpoint {Path.GetFileName(resumeDir)} (step {resumeMeta.Step})");
                onResume?.Invoke(resumeMeta.Step);
            }
            else
            {
                onResume?.Invoke(0);
            }

            var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
            if (resumeDir is null)
            {
                WeightInitializer.InitializeRandomly(weights, Seed);
                Log("Weights initialised.");
            }
            using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
            if (resumeDir is null && model.Config.NumExperts > 0)
            {
                WeightInitializer.InitializeModelMoE(model, Seed + 1013);
                Log("MoE router/expert weights initialised.");
            }
            var parameters = model.Parameters().ToList();
            var ops = TrainingOpsFactory.Create(sharpConfig);

            // Accelerator plugins live in the same folder as data-pipeline plugins.
            // An explicit accelerator that cannot be honoured (no device, unsupported shape) is
            // reported back to the UI, which shows a consent picker (CPU + every other capable
            // plugin) rather than failing the run — never a silent CPU fallback. Only genuinely
            // unfindable/non-capable names fail here with an error.
            var accelerators = AcceleratorLoader.LoadFrom(pluginsFolder, out var acceleratorWarnings);
            foreach (var w in acceleratorWarnings) Log($"Accelerator: {w}");
            var loss = new CrossEntropyLoss(labelSmoothing: job.LabelSmoothing);
            var mapping = GradientMappingFactory.Create(sharpConfig);

            (ITrainingEngine? Engine, string? Refusal, string Name) ResolveEngine()
            {
                string name = job.Accelerator?.Trim() ?? "";
                try
                {
                    var engine = TrainingEngineResolver.Resolve(name, accelerators,
                        new TrainingEngineContext(model, parameters, mapping, sharpConfig, loss,
                            BatchSize: job.BatchSize, SeqLen: job.SeqLen, LabelSmoothing: job.LabelSmoothing));
                    return (engine, null, name);
                }
                catch (AcceleratorUnavailableException ex)
                {
                    return (null, ex.Reason, name);
                }
            }

            var resolved = ResolveEngine();
            if (resolved.Refusal is not null)
                return new TrainRunResult { Success = false, AcceleratorRequested = resolved.Name, AcceleratorReason = resolved.Refusal };

            using var engine = resolved.Engine;
            if (engine is not null)
            {
                // Re-derived from the same name Resolve just matched — FirstOrDefault
                // rather than First so a future drift between the two lookups logs
                // nothing instead of throwing "Sequence contains no matching element"
                // into the generic catch around this run.
                var chosen = accelerators.FirstOrDefault(p => string.Equals(p.Name, resolved.Name, StringComparison.OrdinalIgnoreCase));
                if (chosen is not null)
                    Log($"Accelerator: {chosen.Name} — {chosen.Description}");
                Log($"Engine: {engine.Description}");
            }

            // 5. Optimizer + scheduler + loop.
            using IOptimizer optimizer = TrainingModelOptions.UsesSgd(job)
                ? new SGD(parameters, lr: job.LearningRate, momentum: job.SgdMomentum, weightDecay: job.WeightDecay)
                : new AdamW(parameters, ops, lr: job.LearningRate, weightDecay: job.WeightDecay);
            var scheduler = new CosineWithWarmup(
                maxLr: job.LearningRate, minLr: job.MinLr,
                warmupSteps: job.WarmupSteps, decaySteps: job.TotalSteps);

            Directory.CreateDirectory(checkpointDir);

            var qat = ParseQat(job.QuantAwareTraining);
            var loop = new TrainLoop(
                model: model,
                parameters: parameters,
                loader: loader,
                optimizer: optimizer,
                scheduler: scheduler,
                ops: ops,
                loss: loss,
                engine: engine,
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
                    ResumeFrom = resumeDir,
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
            SmmTrainingExporter.Export(weights, tokenizer, exportPath, BuildEmbedOptions(job, "training", SourceHasher.Combined(job.Sources)), model: model);
            progress.Report(1f);
            Log($"Saved: {exportPath} ({new FileInfo(exportPath).Length:N0} bytes)");

            // Record the corpus as trained-that-far so the next incremental run
            // diffs against EXACTLY what these weights have seen (not what the
            // run started with). Persisted to both the job (in-memory, saved
            // when the wizard saves) and the checkpoint folder (survives app
            // restart even without a manual Save).
            if (job.IncrementalMode)
            {
                job.SourceFileHashes = plan.CurrentFileHashes;
                IncrementalStore.Save(job);
                Log("Incremental: recorded per-file hashes for the corpus just trained.");
            }

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

    /// <summary>
    /// A composite over every configured source, unrestricted — used to train the
    /// tokenizer over the complete corpus even when the training pipeline of an
    /// incremental run is restricted to the delta files.
    /// </summary>
    private static IDataSource FullCorpusComposite(TrainJobSettings job, IReadOnlyList<ComponentDescriptor> registry)
    {
        var all = job.Sources.Select(s => BuildSource(s.Component, registry)).ToList();
        return all.Count == 1 ? all[0] : new CompositeSource(all);
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
            SmmTrainingExporter.Export(weights, tokenizer, smmPath, BuildEmbedOptions(job, "checkpoint", SourceHasher.Combined(job.Sources)), model: model);
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
    private static SmmWriteOptions BuildEmbedOptions(TrainJobSettings job, string source, string? checksum = null)
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
            skills = [.. Directory.GetFiles(job.SkillsFolder, "*.md", SearchOption.TopDirectoryOnly)
                .OrderBy(f => Path.GetFileName(f))
                .Select(File.ReadAllText)];
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
            Checksum = checksum,
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