using SharpMind;
using SharpMind.AgentTools;
using SharpMind.Core.Quantization;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.CUI.App;

/// <summary>
/// The expensive, shareable part of launching a session: a GGUF file fully
/// read into a real Transformer + Tokenizer. Kept distinct from any one
/// ChatSession built on top of it so that opening a second named chat
/// session against the same model file can reuse this instead of reading
/// gigabytes of weights a second time. RefCount tracks how many open chat
/// sessions are currently using it; the owner (see ModelCache) is
/// responsible for actually disposing the Transformer once that count
/// reaches zero.
/// </summary>
public sealed class LoadedModel
{
    public required string ModelPath { get; init; }
    public required Transformer Model { get; init; }
    public required Tokenizer Tokenizer { get; init; }
    public required ModelMetaData Meta { get; init; }
    public required HardwareTier HardwareTier { get; init; }
    public required bool UseGpu { get; init; }
    public int RefCount;
}

/// <summary>Result of attempting to load a model file (the expensive, shareable phase).</summary>
public sealed class ModelLoadResult
{
    public LoadedModel? Loaded { get; init; }
    public string? Error { get; init; }
    public bool Success => Error is null && Loaded is not null;
}

/// <summary>Result of attempting to build a session on top of an already-loaded model (the cheap phase).</summary>
public sealed class LaunchResult
{
    public IChatSession? Session { get; init; }
    public IAgentBuilder? Agent { get; init; }
    public CuiToolContext? CuiContext { get; init; }
    public List<string> Warnings { get; init; } = [];
    public string? Error { get; init; }
    public bool Success => Error is null && (Session is not null || IsDebugMode);

    /// <summary>
    /// True when GeneratorStrategy.UIDebug was selected. In this case Session
    /// is deliberately null — there is no real model, tokenizer, or
    /// transformer involved — and the caller should drive the chat screen
    /// with a DebugChatBridge instead of a ChatSessionBridge.
    /// </summary>
    public bool IsDebugMode { get; init; }
}

/// <summary>
/// Turns a <see cref="SessionOptions"/> into a running session. This is the
/// one place that knows how to map the UI's plain enums onto the engine's
/// generic ChatSession&lt;T,K&gt; combinatorics — everything upstream of here
/// only ever deals with GeneratorStrategy/CacheStrategy, never with the
/// generic types themselves.
///
/// Split into two phases — <see cref="LoadModelAsync"/> and
/// <see cref="BuildSession"/> — specifically so a second chat session
/// against the same GGUF file doesn't have to re-read it. The caller (see
/// ModelCache in MainWindow) is responsible for deciding when a
/// <see cref="LoadedModel"/> can be reused versus when option changes
/// (different hardware tier, different GPU setting) mean it actually needs
/// a fresh load — those choices change what gets baked into the Transformer
/// itself, not just session-level behaviour, so they can't share a
/// LoadedModel even though the file path is the same.
/// </summary>
public static class SessionLauncher
{
    /// <summary>The expensive phase: read a GGUF file into a real, ready-to-use Transformer + Tokenizer.</summary>
    public static async Task<ModelLoadResult> LoadModelAsync(SessionOptions options, IProgress<string>? status = null, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(options.ModelPath) || !File.Exists(options.ModelPath))
            return new ModelLoadResult { Error = $"Model file not found: {options.ModelPath}" };
        var fmt = ModelFormatHelpers.GetFormatForExtension(options.ModelPath);
        if(fmt==null) return new ModelLoadResult { Error = $"Model type not supported: {options.ModelPath}" };
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);

        status?.Report("Reading model metadata...");
        ModelMetaData meta;
        ModelConfig modelConfig;
        Tokenizer? tokenizer;
        try
        {
            (meta, modelConfig, tokenizer) = await Task.Run(() =>
            {
                metaHelper.Load(options.ModelPath, null, out var m, out var c, out var t);
                return (m, c, t);
            });
        }
        catch (Exception ex)
        {
            return new ModelLoadResult { Error = $"Failed to read model: {ex.Message}" };
        }

        if (tokenizer is null)
            return new ModelLoadResult { Error = "Model file has no embedded tokenizer data and no fallback tokenizer path was given." };

        status?.Report("Assembling model...");
        await Task.Yield();
        var sharpConfig = modelConfig.ForModel(hw: options.HardwareTier)
        ;

        // Build a single combined mapping dictionary that includes both model-level
        // operations (pointwise, gate, softmax, etc.) and quantization operations
        // (vecdot, qmatmul, read, etc.). SharpMindConfig.ToJigSawMapping() now produces
        // both sets. The GPU path uses MappingBuilder.WithGpu() which additionally
        // overrides selected entries with GPU-kernel values — the same dictionary
        // shape is ultimately passed to ModelFactory.CreateSession.
        Dictionary<string, string> mapping = options.UseGpu
            ? new MappingBuilder(options.HardwareTier).ApplyPreset(sharpConfig).ApplyQuantPreset(parallel: options.UseParallelKernels).WithGpu(nonQuant: options.GpuNonQuant, vecDot: options.GpuVecDot, matMul: options.GpuMatMul).Build()
            : new MappingBuilder(options.HardwareTier).ApplyPreset(sharpConfig).ApplyQuantPreset(parallel: options.UseParallelKernels).Build();

        status?.Report("Loading weights...");
        TransformerWeights weights;
        try
        {
            var progress = status is null
                ? null
                : new Progress<float>(p => status.Report($"Loading weights... {p:P0}"));

            var qOps = QuantizationFactory.Create(mapping);
            weights = await Task.Run(() =>
            {
                var w = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, options.ModelPath);
                w.InitializeWeights(progress);
                return w;
            });
        }
        catch (Exception ex)
        {
            return new ModelLoadResult { Error = $"Failed to load weights: {ex.Message}" };
        }

        
        status?.Report("Creating session...");
        await Task.Yield();
        var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        return new ModelLoadResult
        {
            Loaded = new LoadedModel
            {
                ModelPath = options.ModelPath,
                Model = model,
                Tokenizer = tokenizer,
                Meta = meta,
                HardwareTier = sharpConfig.ResolvedHardware,
                UseGpu = options.UseGpu
            }
        };
    }

    /// <summary>The cheap phase: build a ChatSession (or a debug bridge context) on top of an already-loaded model.</summary>
    public static LaunchResult BuildSession(SessionOptions options, LoadedModel? loaded, Func<ToolPermissionContext, Task<ToolPermission>>? permissions)
    {
        var warnings = new List<string>();

        if (options.Generator == GeneratorStrategy.UIDebug)
        {
            // No model required at all — that's the entire point of this mode.
            // A CuiToolContext is still created so TestOptions can exercise the
            // same choice-dialog path a real model's UIShowOptionSelection call
            // would use.
            return new LaunchResult { IsDebugMode = true, CuiContext = new CuiToolContext(), Warnings = warnings };
        }

        if (loaded is null)
            return new LaunchResult { Error = "No loaded model was provided for a non-debug session." };

        // --- Tools / skills / agent -----------------------------------------
        var resolvedToolPaths = ResolveToolAssemblyPaths(options);
        var cuiContext = new CuiToolContext();

        var builder = new AgentBuilder(options.AgentName, options.Sampling);
        builder.DisabledTools = options.DisabledTools;
        builder.WithTools(new CuiTools(cuiContext));
        builder.WithTools(new WeatherTool());
        builder.WithTools(new FileSystemTool(options.ProjectPath ?? Directory.GetCurrentDirectory()));

        foreach (var folder in options.SkillFolders)
            builder.WithSkills(folder);

        if (resolvedToolPaths.Count > 0)
        {
            var (toolInstances, toolWarnings) = ToolAssemblyLoader.Load(resolvedToolPaths);
            warnings.AddRange(toolWarnings);
            if (toolInstances.Count > 0)
                builder.WithTools(toolInstances.ToArray());
        }

        if (options.AgentsEnabled)
            builder.WithAgents(options.MaxAgentDepth);

        IAgentBuilder agentBuilder = builder;

        // --- Resolve generator/cache type combo and build the session ------
        Type generatorBuilderDef = options.Generator switch
        {
            GeneratorStrategy.Standard => typeof(StandardGeneratorBuilder<>),
            GeneratorStrategy.Speculative => typeof(SpeculativeGeneratorBuilder<>),
            GeneratorStrategy.Medusa => typeof(MedusaGeneratorBuilder<>),
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };

        Type cacheBuilder = options.Cache switch
        {
            CacheStrategy.Standard => typeof(KVCacherBuilder),
            CacheStrategy.Paged => typeof(PagedKVCacherBuilder),
            CacheStrategy.Quantized => typeof(QuantizedKVCacherBuilder),
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };

        IChatSession session;
        try
        {
            session = ChatSessionFactory.CreateChatSession(
                generatorBuilderDef, cacheBuilder, loaded.Model, loaded.Tokenizer, loaded.Meta, agentBuilder,
                preProcessor: null, compactor: null, permissions: permissions,
                seed: options.Sampling.Seed);
        }
        catch (Exception ex)
        {
            return new LaunchResult { Error = $"Failed to start session with {options.Generator}/{options.Cache}: {ex.Message}", Warnings = warnings };
        }

        session.MaxTokens = options.Generation.MaxNewTokens;
        session.Temperature = options.Sampling.Temperature;
        session.TopK = options.Sampling.TopK;
        session.TopP = options.Sampling.TopP;
        session.RepetitionPenalty = options.Generation.RepetitionPenalty;
        session.RepetitionWindow = options.Generation.RepetitionWindow;
        session.ShowThinking = options.ShowThinking;
        if (options.Generation.StopTokenIds.Count > 0)
            session.StopTokenIds = options.Generation.StopTokenIds;

        return new LaunchResult { Session = session, Agent = agentBuilder, CuiContext = cuiContext, Warnings = warnings };
    }

    /// <summary>Discovers all tools that would be registered for the given options, ignoring the disabled set.</summary>
    public static List<string> GetAvailableTools(SessionOptions options)
    {
        var resolvedToolPaths = ResolveToolAssemblyPaths(options);
        var cuiContext = new CuiToolContext();
        var builder = new AgentBuilder();
        
        // We pass an empty DisabledTools set because we want ALL possible tools
        builder.DisabledTools = [];
        
        builder.WithTools(new CuiTools(cuiContext));
        builder.WithTools(new WeatherTool());
        builder.WithTools(new FileSystemTool(options.ProjectPath ?? Directory.GetCurrentDirectory()));

        if (resolvedToolPaths.Count > 0)
        {
            var (toolInstances, _) = ToolAssemblyLoader.Load(resolvedToolPaths);
            if (toolInstances.Count > 0)
                builder.WithTools(toolInstances.ToArray());
        }

        return builder.RegisteredToolNames.ToList();
    }

    /// <summary>
    /// Combines the explicit tool DLL paths with a fresh scan of
    /// <see cref="SessionOptions.ToolsFolder"/> at launch time. The folder is
    /// deliberately re-read here rather than ever being snapshotted into
    /// <see cref="SessionOptions.ToolAssemblyPaths"/> — that's what makes
    /// dropping a new tool DLL into the folder between sessions (or even
    /// while sitting on the Options screen before pressing Launch) actually
    /// take effect, instead of requiring the path list to be manually
    /// re-typed or the app restarted.
    /// </summary>
    private static List<string> ResolveToolAssemblyPaths(SessionOptions options)
    {
        var paths = new List<string>(options.ToolAssemblyPaths);

        if (!string.IsNullOrWhiteSpace(options.ToolsFolder) && Directory.Exists(options.ToolsFolder))
        {
            paths.AddRange(Directory.GetFiles(options.ToolsFolder, "*.dll"));
        }

        return [.. paths
            .Select(p => Path.GetFullPath(p))
            .Distinct(StringComparer.OrdinalIgnoreCase)];
    }
}
