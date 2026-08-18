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
}