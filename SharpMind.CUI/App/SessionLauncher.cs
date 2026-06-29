using SharpMind;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.CUI.App;

/// <summary>Result of attempting to launch a session: either it worked, or it didn't and here's why.</summary>
public sealed class LaunchResult
{
    public IChatSession? Session { get; init; }
    public IAgentBuilder? Agent { get; init; }
    public ModelMetaData? Meta { get; init; }
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
/// </summary>
public static class SessionLauncher
{
    public static async Task<LaunchResult> LaunchAsync(SessionOptions options, IProgress<string>? status = null, CancellationToken ct = default)
    {
        var warnings = new List<string>();

        if (options.Generator == GeneratorStrategy.UIDebug)
        {
            // No model file check, no GGUF load, no tokenizer, no transformer —
            // that's the entire point of this mode. A CuiToolContext is still
            // created so TestOptions can exercise the same choice-dialog path
            // a real model's UIShowOptionSelection call would use.
            status?.Report("Starting debug session (no model)...");
            await Task.Yield(); // keep this genuinely async-shaped rather than synchronous-but-typed-as-Task
            return new LaunchResult { IsDebugMode = true, CuiContext = new CuiToolContext(), Warnings = warnings };
        }

        if (string.IsNullOrWhiteSpace(options.ModelPath) || !File.Exists(options.ModelPath))
            return new LaunchResult { Error = $"Model file not found: {options.ModelPath}" };

        // --- Load model -----------------------------------------------------
        status?.Report("Reading model metadata...");
        ModelMetaData meta;
        ModelConfig modelConfig;
        Tokenizer? tokenizer;
        try
        {
            GgufLoader.Load(options.ModelPath, null, out meta, out modelConfig, out tokenizer);
        }
        catch (Exception ex)
        {
            return new LaunchResult { Error = $"Failed to read model: {ex.Message}" };
        }

        if (tokenizer is null)
            return new LaunchResult { Error = "Model file has no embedded tokenizer data and no fallback tokenizer path was given." };

        status?.Report("Loading weights...");
        TransformerWeights weights;
        try
        {
            var progress = status is null
                ? null
                : new Progress<float>(p => status.Report($"Loading weights... {p:P0}"));
            weights = GgufLoader.LoadWeightsToTransformerWeights(options.ModelPath, modelConfig, progress);
        }
        catch (Exception ex)
        {
            return new LaunchResult { Error = $"Failed to load weights: {ex.Message}" };
        }

        status?.Report("Assembling model...");
        var sharpConfig = modelConfig.ForModel(hw: options.HardwareTier);

        // GPU path needs the mapping built manually via MappingBuilder so
        // WithGpu() can be chained in — SharpMindConfig.ToJigSawMapping()
        // (the simpler CPU-only path) has no GPU-aware overload. Both paths
        // ultimately produce the same Dictionary<string,string> shape
        // ModelFactory.CreateSession accepts; this only decides how that
        // dictionary gets built, mirroring the engine's own QwenOnGpu sample
        // exactly for the GPU case.
        //
        // Requires a project reference to SharpMind.GPU for WithGpu() to
        // resolve at all — if that reference isn't present, UseGpu in
        // SessionOptions simply won't compile against this method, which is
        // the correct failure mode (a missing capability should fail to
        // build, not silently no-op at runtime).
        Dictionary<string, string> mapping = options.UseGpu
            ? new MappingBuilder(options.HardwareTier).ApplyPreset(sharpConfig).WithGpu().Build()
            : sharpConfig.ToJigSawMapping();

        var model = ModelFactory.CreateSession(weights, sharpConfig, mapping);

        // --- Tools / skills / agent -----------------------------------------
        var resolvedToolPaths = ResolveToolAssemblyPaths(options);
        var cuiContext = new CuiToolContext();

        // CuiTools is always registered — every session gets it regardless of
        // what other tools, skills, or sub-agent settings the user configured.
        // This is what makes a model that's never seen this UI before still
        // able to present a choice dialog or ask about the host machine the
        // moment it's loaded, rather than that depending on the user having
        // remembered to add a tool DLL for it.
        var builder = new AgentBuilder(options.AgentName, options.Sampling);
        builder.WithTools(new CuiTools(cuiContext));

        foreach (var folder in options.SkillFolders)
            builder.WithSkills(folder);

        if (resolvedToolPaths.Count > 0)
        {
            status?.Report("Loading tool assemblies...");
            var (toolInstances, toolWarnings) = ToolAssemblyLoader.Load(resolvedToolPaths);
            warnings.AddRange(toolWarnings);
            if (toolInstances.Count > 0)
                builder.WithTools(toolInstances.ToArray());
        }

        if (options.AgentsEnabled)
            builder.WithAgents(options.MaxAgentDepth);

        IAgentBuilder agentBuilder = builder;

        // --- Resolve generator/cache type combo and build the session ------
        status?.Report("Starting session...");
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
                generatorBuilderDef, cacheBuilder, model, tokenizer, meta, agentBuilder,
                preProcessor: null, compactor: null, permissions: null,
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
        if (options.Generation.StopTokenIds.Count > 0)
            session.StopTokenIds = options.Generation.StopTokenIds;

        // Note: MaxToolCallsPerTurn and MaxAgentDepth live on ChatSession<T,K>
        // directly, not on IChatSession, so they can't be set through this
        // non-generic handle. They fall back to the engine's own defaults
        // (10 and 2 respectively) until those two properties are promoted to
        // the interface.

        return new LaunchResult { Session = session, Agent = agentBuilder, Meta = meta, CuiContext = cuiContext, Warnings = warnings };
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
            // Top-level only, not recursive: a tools folder containing nested
            // project subfolders (e.g. a cloned tool repo with its own
            // obj/bin structure) would otherwise pull in unrelated build
            // output DLLs that happen to also carry a .dll extension.
            paths.AddRange(Directory.GetFiles(options.ToolsFolder, "*.dll"));
        }

        return paths
            .Select(p => Path.GetFullPath(p))
            .Distinct(StringComparer.OrdinalIgnoreCase) // Windows paths are case-insensitive; harmless to dedupe this way on Unix too
            .ToList();
    }
}
