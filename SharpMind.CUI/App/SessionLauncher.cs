using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.Core.Plugins;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
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
    /// <summary>Architecture/kernel config the model was built with — needed by accelerator engines to pick matching kernels.</summary>
    public required SharpMindConfig Config { get; init; }
    public int RefCount;
}

/// <summary>Result of attempting to load a model file (the expensive, shareable phase).</summary>
public sealed class ModelLoadResult
{
    public LoadedModel? Loaded { get; init; }
    public string? Error { get; init; }
    public bool Success => Error is null && Loaded is not null;
}

/// <summary>
/// Plugins embedded inside an SMM model container, materialized once so the
/// permission gate and the session builder agree on the exact same set.
/// </summary>
public sealed class EmbeddedPluginInfo
{
    /// <summary>Assembly file names as recorded in the container (e.g. "SharpMind.Plugins.Weather.dll").</summary>
    public required IReadOnlyList<string> AssemblyNames { get; init; }

    /// <summary>Tool method names contributed by the embedded plugin tools.</summary>
    public required IReadOnlySet<string> ToolNames { get; init; }

    /// <summary>Materialized components (tools, compactors, pre/post processors, generators).</summary>
    public required PluginLoadResult Plugins { get; init; }

    public bool HasPlugins => AssemblyNames.Count > 0;
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
    /// The backend actually used for inference, for the chat status sidebar — e.g. <c>"CPU"</c>
    /// (the default, no accelerator) or a device string like <c>"[Cuda] GTX 1060, cuBLAS 12.8"</c>
    /// (the resolved engine's <see cref="IInferenceEngine.Description"/>).
    /// </summary>
    public string? EngineDescription { get; init; }

    /// <summary>
    /// Set when the requested accelerator exists but its engine factory declined (no device,
    /// unsupported shape): the factory's human-readable reason. When present the caller should show
    /// <see cref="AcceleratorPicker"/> (CPU + other capable plugins) rather than treating it as a
    /// hard launch failure. <see cref="Success"/> stays false; there is no <see cref="Session"/>.
    /// </summary>
    public string? AcceleratorRefusal { get; init; }

    /// <summary>
    /// True when GeneratorStrategy.UIDebug was selected. In this case Session
    /// is deliberately null - there is no real model, tokenizer, or
    /// transformer involved - and the caller should drive the chat screen
    /// with a DebugChatBridge instead of a ChatSessionBridge.
    /// </summary>
    public bool IsDebugMode { get; init; }
}

/// <summary>
/// Turns a <see cref="SessionOptions"/> into a running session. This is the
/// one place that knows how to map the UI's plain enums onto the engine's
/// generic ChatSession&lt;T,K&gt; combinatorics - everything upstream of here
/// only ever deals with GeneratorStrategy/CacheStrategy, never with the
/// generic types themselves.
///
/// Split into two phases - <see cref="LoadModelAsync"/> and
/// <see cref="BuildSession"/> - specifically so a second chat session
/// against the same GGUF file doesn't have to re-read it. The caller (see
/// ModelCache in MainWindow) is responsible for deciding when a
/// <see cref="LoadedModel"/> can be reused versus when option changes
/// (different hardware tier) mean it actually needs a fresh load - those
/// choices change what gets baked into the Transformer
/// itself, not just session-level behaviour, so they can't share a
/// LoadedModel even though the file path is the same.
/// </summary>
public static class SessionLauncher
{
    /// <summary>The expensive phase: read a GGUF file into a real, ready-to-use Transformer + Tokenizer.</summary>
    public static async Task<ModelLoadResult> LoadModelAsync(SessionOptions options, IProgress<string>? status = null,
        string? pluginsFolder = null, CancellationToken ct = default)
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
            },ct);
        }
        catch (OperationCanceledException)
        {
            return new ModelLoadResult { Error = $"Operation cancelled" };
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

        // Build the full mapping via SharpMindConfig.ToJigSawMapping().
        Dictionary<string, string> mapping = sharpConfig.ToJigSawMapping(parallel: options.UseParallelKernels);

        // Fail fast before paying for the weight load: an accelerator that can never run this
        // model's quant dtypes / shape is refused from the metadata alone (the refusal reason is
        // the same one BuildSession would surface after loading). Otherwise we would read the
        // whole file only to refuse it at engine creation.
        if (!string.IsNullOrWhiteSpace(options.InferenceAccelerator)
            && !options.InferenceAccelerator.Trim().Equals(InferenceEngineResolver.CpuName, StringComparison.OrdinalIgnoreCase))
        {
            var refusal = CheckAcceleratorCompatibility(options.InferenceAccelerator, meta, modelConfig, sharpConfig, pluginsFolder);
            if (refusal is not null)
                return new ModelLoadResult { Error = refusal };
        }

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
                // Chat only ever runs the quantized forward, which reads the raw
                // bytes, so skip the dequantized F32 copy of every layer.
                var w = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, options.ModelPath, options.LoadMode,
                    quantizedResident: true);
                w.InitializeWeights(progress);
                return w;
            },ct);
        }
        catch (OperationCanceledException)
        {
            return new ModelLoadResult { Error = $"Operation cancelled" };
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
                Config = sharpConfig
            }
        };
    }

    /// <summary>
    /// Reads and materializes any plugins embedded inside an SMM model file.
    /// Returns null when the path is not an SMM container or carries no plugin
    /// section, so callers can treat "no embedded plugins" exactly like "no
    /// plugins". The returned info drives both the Options screen's "(Plugin
    /// Embedded)" tags and the permission gate's restricted-tool set.
    /// </summary>
    public static EmbeddedPluginInfo? LoadEmbeddedPlugins(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath)) return null;
        if (ModelFormatHelpers.GetFormatForExtension(modelPath) != ModelFormat.Smm) return null;

        List<SmmPluginEntry> entries;
        try
        {
            entries = SmmLoader.LoadPlugins(modelPath);
        }
        catch (Exception)
        {
            return null;
        }
        if (entries.Count == 0) return null;

        var result = PluginLoader.LoadFromBytes(entries.Select(e => (e.Name, e.AssemblyBytes)));

        // Tool names are the method names the model will actually call - reuse
        // AgentBuilder's registration so the set exactly matches live tools.
        var toolNames = new AgentBuilder()
            .WithTools([.. result.Tools])
            .RegisteredToolNames;

        return new EmbeddedPluginInfo
        {
            AssemblyNames = [.. entries.Select(e => e.Name)],
            ToolNames = toolNames.ToHashSet(StringComparer.Ordinal),
            Plugins = result
        };
    }

    /// <summary>
    /// Pre-weight-load accelerator compatibility gate. Resolves the named accelerator plugin and
    /// asks its inference capability <i>from the metadata alone</i> whether this model can run on
    /// it. Returns a refusal reason (or null when compatible / the plugin does not implement the
    /// metadata check, in which case the authoritative per-layer check at BuildSession still runs).
    /// </summary>
    private static string? CheckAcceleratorCompatibility(string accelerator, ModelMetaData meta,
        ModelConfig modelConfig, SharpMindConfig sharpConfig, string? pluginsFolder)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(pluginsFolder))
                return "No accelerator plugins folder is configured; cannot resolve inference accelerator " +
                    $"'{accelerator}' to check compatibility.";

            var accelerators = AcceleratorLoader.LoadFrom(pluginsFolder, out _);
            string wanted = AcceleratorNames.Canonicalize(accelerator.Trim());
            var plugin = accelerators.FirstOrDefault(p => string.Equals(p.Name, wanted, StringComparison.OrdinalIgnoreCase));
            if (plugin is null)
                return $"Accelerator '{wanted}' was not found in the plugins folder (available: " +
                    $"{(accelerators.Count == 0 ? "none loaded" : string.Join(", ", accelerators.Select(p => p.Name)))}).";

            var factory = plugin.Capabilities.OfType<IInferenceEngineFactory>().FirstOrDefault();
            if (factory is null) return null;   // no inference capability -> let BuildSession resolve

            return factory.CheckSupported(meta, modelConfig, sharpConfig);
        }
        catch (Exception ex)
        {
            // Never block the load on a plugin failure; the engine creation at BuildSession is the
            // authoritative gate and reports underlying plugin problems there.
            return null;
        }
    }

    /// <summary>The cheap phase: build a ChatSession (or a debug bridge context) on top of an already-loaded model.</summary>
    public static LaunchResult BuildSession(SessionOptions options, LoadedModel? loaded, Func<ToolPermissionContext, Task<ToolPermission>>? permissions, EmbeddedPluginInfo? embedded = null, string? pluginsFolder = null)
    {
        var warnings = new List<string>();

        if (options.Generator == GeneratorStrategy.UIDebug)
        {
            // No model required at all - that's the entire point of this mode.
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

        var builder = new AgentBuilder(options.AgentName, options.Sampling)
        {
            DisabledTools = options.DisabledTools
        };
        if (!options.DisableTools)
        {
            builder.WithTools(new CuiTools(cuiContext));
            builder.WithTools(new WeatherTool());
            builder.WithTools(new FileSystemTool(options.ProjectPath ?? Directory.GetCurrentDirectory()));
        }

            foreach (var folder in options.SkillFolders)
            builder.WithSkills(folder);

        // --- Embedded skills / default system prompt from the model (SMM) ---
        // Silently auto-applied so a self-contained .SMM carries its own agent
        // context. Skills load into the standard Skills section; the system
        // prompt is inserted as an additional system message at the top of the
        // history (before the synthesized agent prompt).
        if (loaded.Meta is not null && ModelFormatHelpers.GetFormatForExtension(loaded.ModelPath) == ModelFormat.Smm)
        {
            foreach (string skillContent in SmmLoader.LoadSkillsFromMeta(loaded.Meta))
                builder.WithSkillContent(skillContent);
            if (SmmLoader.LoadSystemPromptFromMeta(loaded.Meta) is { } systemPrompt)
                builder.WithAdditionalSystemPrompt(systemPrompt);
        }

        if (resolvedToolPaths.Count > 0 && !options.DisableTools)
        {
            var (toolInstances, toolWarnings) = ToolAssemblyLoader.Load(resolvedToolPaths);
            warnings.AddRange(toolWarnings);
            if (toolInstances.Count > 0)
                builder.WithTools([.. toolInstances]);
        }

        if (options.AgentsEnabled)
            builder.WithAgents(options.MaxAgentDepth);

        // --- Load plugins from ./plugins/ ----------------------------------
        var pluginResult = PluginLoader.LoadFrom(Path.Combine(AppContext.BaseDirectory, "plugins"));
        warnings.AddRange(pluginResult.Warnings);

        var pluginCompactors = new List<IContextCompactor>(pluginResult.Compactors);
        var pluginPreProcessors = new List<IPromptPreProcessor>(pluginResult.PreProcessors);
        var pluginPostProcessors = new List<IPromptPostProcessor>(pluginResult.PostProcessors);
        var pluginGenerators = new List<PluginGeneratorInfo>(pluginResult.Generators);

        // --- Merge plugins embedded inside the model file (SMM) -------------
        // Registered through the exact same pathways as on-disk plugins, so they
        // are gated identically. Their tools are additionally forced through the
        // Ask permission flow (see PermissionGate / PermissionPolicy).
        if (embedded is not null && embedded.HasPlugins)
        {
            pluginCompactors.AddRange(embedded.Plugins.Compactors);
            pluginPreProcessors.AddRange(embedded.Plugins.PreProcessors);
            pluginPostProcessors.AddRange(embedded.Plugins.PostProcessors);
            pluginGenerators.AddRange(embedded.Plugins.Generators);
            if (embedded.Plugins.Tools.Count > 0 && !options.DisableTools)
                builder.WithTools([.. embedded.Plugins.Tools]);
        }

        builder.PluginCompactors = pluginCompactors;
        builder.PluginPreProcessors = pluginPreProcessors;
        builder.PluginPostProcessors = pluginPostProcessors;
        if (pluginResult.Tools.Count > 0 && !options.DisableTools)
            builder.WithTools([.. pluginResult.Tools]);

        // Resolve compactor: plugin compactor takes priority, then built-in strategy
        builder.Compactor = options.PluginCompactorName is not null
            ? pluginCompactors.FirstOrDefault(c =>
                string.Equals(c.Name, options.PluginCompactorName, StringComparison.OrdinalIgnoreCase))
            : options.Compactor switch
            {
                CompactorStrategy.Truncate => new TruncatingCompactor(),
                CompactorStrategy.Summarize => new SummarizingCompactor(),
                _ => null
            };

        // SkipAgentPrompt drops the whole agent layer — no synthesized agent
        // prompt, no sub-agents, no tool-call loop. DisableTools is the narrower
        // control that keeps the agent prompt but registers no tools.
        IAgentBuilder? agentBuilder = options.SkipAgentPrompt ? null : builder;

        // --- Resolve generator/cache type combo and build the session ------
        Type cacheBuilder = options.Cache switch
        {
            CacheStrategy.Standard => typeof(KVCacherBuilder),
            CacheStrategy.Paged => typeof(PagedKVCacherBuilder),
            CacheStrategy.Quantized => typeof(QuantizedKVCacherBuilder),
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };

        // Check plugin generators first; fall back to built-in strategies
        Type generatorBuilderDef;
        var matchedPluginGen = pluginGenerators.FirstOrDefault(g => g.CacheBuilderType == cacheBuilder);
        if (matchedPluginGen is not null)
        {
            generatorBuilderDef = matchedPluginGen.BuilderType;
        }
        else
        {
            generatorBuilderDef = options.Generator switch
            {
                GeneratorStrategy.Standard => typeof(StandardGeneratorBuilder<>),
                GeneratorStrategy.Speculative => typeof(SpeculativeGeneratorBuilder<>),
                GeneratorStrategy.Medusa => typeof(MedusaGeneratorBuilder<>),
                _ => throw new ArgumentOutOfRangeException(nameof(options))
            };
        }
        string? tmpl = loaded.Meta?.GetChatTemplate();
        IChatPromptFormatter? formatter = options.Formatter switch
        {
            FormatterStrategy.Auto => null,
            FormatterStrategy.Jinja => string.IsNullOrWhiteSpace(tmpl) ? new SimpleFormatter() :  new JinjaTemplateFormatter(tmpl),
            FormatterStrategy.ChatML =>  string.IsNullOrWhiteSpace(tmpl) ? new SimpleFormatter() : new ChatMLFormatter(tmpl),
            FormatterStrategy.Simple => new SimpleFormatter(),
            FormatterStrategy.QuestionAnswer => new QuestionAnswerFormatter(),
            FormatterStrategy.Alpaca => new AlpacaFormatter(),
            FormatterStrategy.Llama3 => new Llama3Formatter(),
            FormatterStrategy.Raw => new RawTemplateFormatter(),
            _ => throw new ArgumentOutOfRangeException(nameof(options))
        };

        // An explicitly chosen formatter can be flatly wrong for the model — the
        // Llama-3 formatter on a Qwen checkpoint frames every turn with tokens
        // that vocabulary does not contain. Nothing downstream fails, it just
        // generates badly, so say so here. Auto is exempt: it picks from this
        // model's own template and vocab.
        if (formatter is not null &&
            ChatPromptFormatterFactory.DescribeVocabMismatch(formatter, loaded.Tokenizer.Vocab.Contains) is { } mismatch)
            warnings.Add(mismatch);
        // Build available pre/post processors (built-in + plugin), filtered by disabled set
        var allPreProcessors = new List<IPromptPreProcessor>
        {
            new SimpleArtifactPromptPreProcessor()
        };
        allPreProcessors.AddRange(pluginPreProcessors);
        IPromptPreProcessor? preProcessor = allPreProcessors.FirstOrDefault(p => !options.DisabledPreProcessors.Contains(p.Name));

        var allPostProcessors = new List<IPromptPostProcessor>(pluginPostProcessors);
        IPromptPostProcessor? postProcessor = allPostProcessors.FirstOrDefault(p => !options.DisabledPostProcessors.Contains(p.Name));


        IChatSession session;
        IInferenceEngine? engine = null;
        try
        {
            // Resolve an accelerator-backed inference engine when the session asks for one.
            // Mirrors the training path (TrainRunner): plugins are scanned from the plugins
            // folder and the choice is honoured or the launch fails — never a silent CPU fallback.
            if (!string.IsNullOrWhiteSpace(options.InferenceAccelerator))
            {
                if (string.IsNullOrWhiteSpace(pluginsFolder))
                    throw new InvalidOperationException(
                        "No accelerator plugins folder is configured; cannot resolve inference accelerator "
                        + $"'{options.InferenceAccelerator}'.");
                var accelerators = AcceleratorLoader.LoadFrom(pluginsFolder, out _);
                int maxCache = ModelConfig.ComputeMaxCacheLength(loaded.Model.Config);
                engine = InferenceEngineResolver.Resolve(options.InferenceAccelerator, accelerators,
                    new InferenceEngineContext(loaded.Model, loaded.Config, maxCache));
            }

            session = ChatSessionFactory.CreateChatSession(
                generatorBuilderDef, cacheBuilder, loaded.Model, loaded.Tokenizer, loaded.Meta, agentBuilder,
                preProcessor: preProcessor, postProcessor: postProcessor, progress: null, permissions: permissions, formatter: formatter,
                seed: options.Sampling.Seed, engine: engine);
        }
        catch (AcceleratorUnavailableException ex)
        {
            // The chosen accelerator exists but can't run here (no device / unsupported shape).
            // Don't fail the launch — the caller shows a consent picker offering the CPU and any
            // other plug-in that can do the job, then re-runs BuildSession with the choice.
            engine?.Dispose();
            return new LaunchResult { AcceleratorRefusal = ex.Reason, Warnings = warnings };
        }
        catch (Exception ex)
        {
            engine?.Dispose();
            return new LaunchResult { Error = $"Failed to start session with {options.Generator}/{options.Cache}: {ex.Message}", Warnings = warnings };
        }

        // MaxNewTokens is the per-turn generation length cap; MaxTokens is the
        // context window that ChatSession.TrimToFitContext budgets against.
        // MaxTokens = EffectiveInferenceCacheLength (the model's actual usable
        // context window, which may be smaller than MaxSeqLen for sliding-window
        // architectures) so long conversations are kept intact rather than
        // trimmed to a token count that silently drops the agent/tool system
        // prompt and memory.
        //
        // The generator prefills in chunks (see Prefill.ForwardLastLogitsChunked),
        // each sized to fit the workspace's ~128-token prefill budget, so a full
        // context window no longer overflows the workspace the way the old
        // single-shot prefill did (a 3k-token agent prompt once blew the bump
        // allocator and surfaced as an empty reply). The cost is that a long
        // first turn prefills slowly — progress is surfaced to the UI as
        // "Prefilling NN.NN%" via IGenerator.PrefillProgress.
        //
        // Options.MaxTokens opts back into a truncated context window (clamped to
        // [1, EffectiveInferenceCacheLength]) to mimic the pre-restore fast-prefill
        // behavior; when null, the full EffectiveInferenceCacheLength context is used.
        session.MaxNewTokens = options.Generation.MaxNewTokens;
        int effectiveCacheLen = ModelConfig.ComputeMaxCacheLength(loaded.Model.Config);
        session.MaxTokens = options.MaxTokens is int maxTokens && maxTokens > 0
            ? Math.Min(maxTokens, effectiveCacheLen)
            : effectiveCacheLen;
        session.Temperature = options.Sampling.Temperature;
        session.TopK = options.Sampling.TopK;
        session.TopP = options.Sampling.TopP;
        session.RepetitionPenalty = options.Generation.RepetitionPenalty;
        session.RepetitionWindow = options.Generation.RepetitionWindow;
        session.ShowThinking = options.ShowThinking;
        session.EnableThinking = options.EnableThinking;
        session.UserName = options.UserName;
        if (options.Generation.StopTokenIds.Count > 0)
            session.StopTokenIds = options.Generation.StopTokenIds;

        return new LaunchResult { Session = session, Agent = agentBuilder, CuiContext = cuiContext, Warnings = warnings, EngineDescription = engine?.Description ?? InferenceEngineResolver.CpuName };
    }

    /// <summary>Discovers all tools that would be registered for the given options, ignoring the disabled set.</summary>
    public static List<string> GetAvailableTools(SessionOptions options)
    {
        var resolvedToolPaths = ResolveToolAssemblyPaths(options);
        var cuiContext = new CuiToolContext();
        var builder = new AgentBuilder
        {
            // We pass an empty DisabledTools set because we want ALL possible tools
            DisabledTools = []
        };

        builder.WithTools(new CuiTools(cuiContext));
        builder.WithTools(new WeatherTool());
        builder.WithTools(new FileSystemTool(options.ProjectPath ?? Directory.GetCurrentDirectory()));

        if (resolvedToolPaths.Count > 0)
        {
            var (toolInstances, _) = ToolAssemblyLoader.Load(resolvedToolPaths);
            if (toolInstances.Count > 0)
                builder.WithTools([.. toolInstances]);
        }

        // Also include plugin tools from the plugins folder.
        var pluginResult = PluginLoader.LoadFrom(Path.Combine(AppContext.BaseDirectory, "plugins"));
        if (pluginResult.Tools.Count > 0)
            builder.WithTools([.. pluginResult.Tools]);

        return [.. builder.RegisteredToolNames];
    }

    public static List<IPromptPreProcessor> GetAvailablePreProcessors()
    {
        var result = new List<IPromptPreProcessor>();
        var pluginResult = PluginLoader.LoadFrom(Path.Combine(AppContext.BaseDirectory, "plugins"));
        result.AddRange(pluginResult.PreProcessors);
        return result;
    }

    public static List<IPromptPostProcessor> GetAvailablePostProcessors()
    {
        var result = new List<IPromptPostProcessor>();
        var pluginResult = PluginLoader.LoadFrom(Path.Combine(AppContext.BaseDirectory, "plugins"));
        result.AddRange(pluginResult.PostProcessors);
        return result;
    }

    /// <summary>
    /// Combines the explicit tool DLL paths with a fresh scan of
    /// <see cref="SessionOptions.ToolsFolder"/> at launch time. The folder is
    /// deliberately re-read here rather than ever being snapshotted into
    /// <see cref="SessionOptions.ToolAssemblyPaths"/> - that's what makes
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
