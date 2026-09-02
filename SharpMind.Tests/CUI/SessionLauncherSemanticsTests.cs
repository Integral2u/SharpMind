using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.Core.Plugins;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Verifies the session token semantics set by <see cref="SessionLauncher.BuildSession"/>:
/// <see cref="IChatSession.MaxTokens"/> must be the model's full context window
/// (<see cref="SharpMind.Model.Transformer.Config.MaxSeqLen"/>) so long
/// conversations are not silently trimmed away, while
/// <see cref="IChatSession.MaxNewTokens"/> stays the per-turn generation cap.
///
/// Driven by <see cref="TinyReferenceModel"/> — a deterministic, seed-fixed,
/// millisecond-to-build reference .SMM — so the whole session path is exercised
/// without loading a real model file in tests.
/// </summary>
public sealed class SessionLauncherSemanticsTests
{
    private static async Task<LoadedModel> Load(TempDirectory temp, SessionOptions options)
    {
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;
        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");
        return load.Loaded!;
    }

    private static SessionOptions SwsOptions() => new()
    {
        AgentsEnabled = false,
        FileAccess = ToolPermission.Always,
        NetworkAccess = ToolPermission.Always,
        Generation = new GenerationConfig { MaxNewTokens = 96 },
    };

    [Fact]
    public async Task BuildSession_MaxTokensIsModelContextWindow_MaxNewTokensStaysPerTurnCap()
    {
        using var temp = new TempDirectory();
        var options = SwsOptions();
        var loaded = await Load(temp, options);
        try
        {
            var result = SessionLauncher.BuildSession(options, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            var session = result.Session!;
            int maxSeqLen = loaded.Model.Config.MaxSeqLen;

            // The whole point: context window is the model's full capacity,
            // not trimmed to the per-turn generation cap.
            Assert.Equal(maxSeqLen, session.MaxTokens);
            Assert.Equal(96, session.MaxNewTokens);
            Assert.True(session.MaxTokens >= session.MaxNewTokens,
                "Context window must be able to hold at least one generated response.");
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Theory]
    [InlineData(512)]
    [InlineData(1)]
    public async Task BuildSession_MaxTokensOverride_TruncatesContext(int overrideTokens)
    {
        using var temp = new TempDirectory();
        var options = SwsOptions();
        options.MaxTokens = overrideTokens;
        var loaded = await Load(temp, options);
        try
        {
            var result = SessionLauncher.BuildSession(options, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            var session = result.Session!;
            int maxSeqLen = loaded.Model.Config.MaxSeqLen;

            Assert.Equal(overrideTokens, session.MaxTokens);
            Assert.True(session.MaxTokens <= maxSeqLen,
                "Override must never exceed the model's context window.");
            Assert.Equal(96, session.MaxNewTokens);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_MaxTokensOverride_AboveMaxSeqLen_ClampsToMaxSeqLen()
    {
        using var temp = new TempDirectory();
        var options = SwsOptions();
        options.MaxTokens = 100_000;
        var loaded = await Load(temp, options);
        try
        {
            var result = SessionLauncher.BuildSession(options, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            int maxSeqLen = loaded.Model.Config.MaxSeqLen;
            Assert.Equal(maxSeqLen, result.Session!.MaxTokens);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_DisableTools_RegistersNoTools()
    {
        using var temp = new TempDirectory();
        var options = SwsOptions();
        options.DisableTools = true;
        var loaded = await Load(temp, options);
        try
        {
            var result = SessionLauncher.BuildSession(options, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            // Agent prompt stays but no tools are registered, so the tool-call
            // loop cannot fire even if the model hallucinates a <tool_call> tag.
            Assert.NotNull(result.Agent);
            Assert.Empty(result.Agent!.RegisteredToolNames);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_SkipAgentPrompt_DropsAgentLayer()
    {
        using var temp = new TempDirectory();
        var options = SwsOptions();
        options.SkipAgentPrompt = true;
        var loaded = await Load(temp, options);
        try
        {
            var result = SessionLauncher.BuildSession(options, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            // Skipping the agent prompt drops the whole agent layer (prompt + tools).
            Assert.Null(result.Agent);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_AfterClone_AppliesSkipPromptDisableToolsAndMaxTokensOverride()
    {
        // Mirrors the CUI launch path: MainWindow clones the Options-screen
        // SessionOptions before BuildSession. Regression for the bug where the
        // clone dropped MaxTokens/SkipAgentPrompt/DisableTools, so every CUI
        // launch silently launched with the full context window, the full agent
        // prompt, and tools enabled.
        using var temp = new TempDirectory();
        var options = SwsOptions();
        options.MaxTokens = 255;
        options.SkipAgentPrompt = true;
        options.DisableTools = true;
        var launchOptions = options.Clone();

        var loaded = await Load(temp, launchOptions);
        try
        {
            var result = SessionLauncher.BuildSession(launchOptions, loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            // SkipAgentPrompt → the agent layer (prompt + tools) is dropped.
            Assert.Null(result.Agent);
            // MaxTokens override survives the clone and is applied.
            Assert.Equal(255, result.Session!.MaxTokens);
            Assert.Equal(96, result.Session.MaxNewTokens);
        }
        finally
        {
            loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task LoadModelAsync_CpuFallbackConsent_RefusesBeforeLoad_ThenLoadsAfterConsent()
    {
        // A quant with no on-device kernel (the fake factory reports Q8_K host fallback) must be
        // refused from the metadata alone — before any weight is read — as a consent signal
        // (CpuFallbackWarning), never as a hard Error, until the caller sets AllowCpuFallback.
        using var temp = new TempDirectory();
        string pluginsDir = CopyTestAssembly(temp, nameof(CpuFallbackConsentAcceleratorPlugin));
        var options = SwsOptions();
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;
        options.InferenceAccelerator = "consentacc";

        var first = await SessionLauncher.LoadModelAsync(options, pluginsFolder: pluginsDir);
        Assert.False(first.Success);
        Assert.Null(first.Error);
        Assert.Null(first.Loaded);
        Assert.Null(first.AcceleratorRefusal);
        Assert.NotNull(first.CpuFallbackWarning);
        Assert.Contains("Q8_K", first.CpuFallbackWarning, StringComparison.OrdinalIgnoreCase);

        // The same load with consent (what the picker's "Allow CPU fallback" sentinel sets) retries
        // past the metadata gate and proceeds to a real load.
        options.AllowCpuFallback = true;
        var second = await SessionLauncher.LoadModelAsync(options, pluginsFolder: pluginsDir);
        Assert.True(second.Success, second.Error ?? "load failed");
        using (second.Loaded!.Model) { }
    }

    [Fact]
    public async Task LoadModelAsync_ArchitectureRefusal_IsNotAnError_AndIgnoresConsent()
    {
        // An architecture the accelerator can never run (the fake factory reports MoE) is refused
        // from the metadata alone as AcceleratorRefusal — the picker signal — and the CPU-fallback
        // consent flag must not mask it.
        using var temp = new TempDirectory();
        string pluginsDir = CopyTestAssembly(temp, nameof(ArchRefusingAcceleratorPlugin));
        var options = SwsOptions();
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;
        options.InferenceAccelerator = "refusingacc";
        options.AllowCpuFallback = true;   // irrelevant for an arch refusal

        var result = await SessionLauncher.LoadModelAsync(options, pluginsFolder: pluginsDir);
        Assert.False(result.Success);
        Assert.Null(result.Error);
        Assert.Null(result.Loaded);
        Assert.Null(result.CpuFallbackWarning);
        Assert.NotNull(result.AcceleratorRefusal);
        Assert.Contains("MoE", result.AcceleratorRefusal, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task LoadModelAsync_CpuAccelerator_SkipsTheMetadataGate()
    {
        // null / "CPU" take the CPU path — the consent and refusal gates exist only for a named
        // accelerator, so a plain CPU load must not be blocked by a fake refusing plugin.
        using var temp = new TempDirectory();
        string pluginsDir = CopyTestAssembly(temp, nameof(CpuFallbackConsentAcceleratorPlugin));
        var options = SwsOptions();
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;
        options.InferenceAccelerator = null;

        var result = await SessionLauncher.LoadModelAsync(options, pluginsFolder: pluginsDir);
        Assert.True(result.Success, result.Error ?? "load failed");
        using (result.Loaded!.Model) { }
    }

    private static string CopyTestAssembly(TempDirectory temp, string _)
    {
        string pluginsDir = Path.Combine(temp.Path, "plugins");
        Directory.CreateDirectory(pluginsDir);
        File.Copy(typeof(CpuFallbackConsentAcceleratorPlugin).Assembly.Location,
            Path.Combine(pluginsDir, Path.GetFileName(typeof(CpuFallbackConsentAcceleratorPlugin).Assembly.Location)));
        return pluginsDir;
    }

    /// <summary>
    /// Fake accelerator discovered from SharpMind.Tests.dll by the launcher's metadata gate: its
    /// inference factory runs the model but reports a per-tensor CPU-fallback (Q8_K) from the
    /// metadata alone — the consent signal <see cref="ModelLoadResult.CpuFallbackWarning"/>.
    /// </summary>
    public sealed class CpuFallbackConsentAcceleratorPlugin : IAcceleratorPlugin
    {
        public string Name => "consentacc";
        public string Description => "fake accelerator with host-fallback consent";
        public IReadOnlyList<object> Capabilities { get; } = [new ConsentFactory()];

        private sealed class ConsentFactory : IInferenceEngineFactory
        {
            public IInferenceEngine? TryCreate(InferenceEngineContext context, out string? reason) { reason = null; return null; }
            public string? CheckSupported(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config) => null;
            public string? DescribeCpuFallback(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config)
                => "will run on the CPU: Q8_K weights (block linears); the rest of the model stays on the GPU.";
        }
    }

    /// <summary>Fake accelerator whose inference factory refuses the model's architecture outright.</summary>
    public sealed class ArchRefusingAcceleratorPlugin : IAcceleratorPlugin
    {
        public string Name => "refusingacc";
        public string Description => "fake accelerator that refuses the architecture";
        public IReadOnlyList<object> Capabilities { get; } = [new RefusingFactory()];

        private sealed class RefusingFactory : IInferenceEngineFactory
        {
            public IInferenceEngine? TryCreate(InferenceEngineContext context, out string? reason) { reason = null; return null; }
            public string? CheckSupported(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config)
                => "GPU inference engine does not support MoE; use CPU inference, which does.";
            public string? DescribeCpuFallback(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config) => null;
        }
    }
}