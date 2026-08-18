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
/// </summary>
public sealed class SessionLauncherSemanticsTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";

    [Fact]
    public async Task BuildSession_MaxTokensIsModelContextWindow_MaxNewTokensStaysPerTurnCap()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded!,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            var session = result.Session!;
            int maxSeqLen = load.Loaded!.Model.Config.MaxSeqLen;

            // The whole point: context window is the model's full capacity,
            // not trimmed to the per-turn generation cap.
            Assert.Equal(maxSeqLen, session.MaxTokens);
            Assert.Equal(96, session.MaxNewTokens);
            Assert.True(session.MaxTokens >= session.MaxNewTokens,
                "Context window must be able to hold at least one generated response.");
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }

    [Theory]
    [InlineData(512)]
    [InlineData(1)]
    public async Task BuildSession_MaxTokensOverride_TruncatesContext(int overrideTokens)
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
            MaxTokens = overrideTokens,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded!,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            var session = result.Session!;
            int maxSeqLen = load.Loaded!.Model.Config.MaxSeqLen;

            Assert.Equal(overrideTokens, session.MaxTokens);
            Assert.True(session.MaxTokens <= maxSeqLen,
                "Override must never exceed the model's context window.");
            Assert.Equal(96, session.MaxNewTokens);
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_MaxTokensOverride_AboveMaxSeqLen_ClampsToMaxSeqLen()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
            MaxTokens = 100_000,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded!,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            int maxSeqLen = load.Loaded!.Model.Config.MaxSeqLen;
            Assert.Equal(maxSeqLen, result.Session!.MaxTokens);
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_DisableTools_RegistersNoTools()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
            DisableTools = true,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded!,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            // Agent prompt stays but no tools are registered, so the tool-call
            // loop cannot fire even if the model hallucinates a <tool_call> tag.
            Assert.NotNull(result.Agent);
            Assert.Empty(result.Agent!.RegisteredToolNames);
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_SkipAgentPrompt_DropsAgentLayer()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
            SkipAgentPrompt = true,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded!,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Assert.True(result.Success, result.Error ?? "build failed");

            // Skipping the agent prompt drops the whole agent layer (prompt + tools).
            Assert.Null(result.Agent);
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }

    [Fact]
    public async Task BuildSession_AfterClone_AppliesSkipPromptDisableToolsAndMaxTokensOverride()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        // Mirrors the CUI launch path: MainWindow clones the Options-screen
        // SessionOptions before BuildSession. Regression for the bug where the
        // clone dropped MaxTokens/SkipAgentPrompt/DisableTools, so every CUI
        // launch silently launched with the full context window, the full agent
        // prompt, and tools enabled.
        var launchOptions = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            Generation = new GenerationConfig { MaxNewTokens = 96 },
            MaxTokens = 255,
            SkipAgentPrompt = true,
            DisableTools = true,
        }.Clone();

        var load = await SessionLauncher.LoadModelAsync(launchOptions);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(launchOptions, load.Loaded!,
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
            load.Loaded!.Model.Dispose();
        }
    }
}