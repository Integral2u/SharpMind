using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
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
}