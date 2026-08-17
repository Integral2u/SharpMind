using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Headless reproduction of the CUI's session path: the exact
/// SessionLauncher.LoadModelAsync + BuildSession construction (agent builder
/// with tools, Auto formatter, MaxTokens = MaxSeqLen, SimpleArtifactPrompt
/// pre-processor, permission callback) driven via StartChatAsync the same
/// way ChatSessionBridge drives it. Diagnoses whether the first turn streams
/// entries or produces none at all — the "stuck at Thinking..." symptom.
/// </summary>
public sealed class CuiSessionReproTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";

    private static void Log(string message) =>
        File.AppendAllText(Path.Combine(Path.GetTempPath(), "cui_session_repro.log"),
            $"{DateTime.Now:HH:mm:ss.fff} {message}{Environment.NewLine}");

    [Fact]
    public async Task CuiSession_FirstTurn_StreamsOrCompletes()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        Log($"start model load maxseq check");

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            ShowThinking = true,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Log($"loaded, maxseqlen={load.Loaded!.Model.Config.MaxSeqLen}");
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var result = SessionLauncher.BuildSession(options, load.Loaded,
                permissions: _ => Task.FromResult(ToolPermission.Always));
            Log($"session built");
            Assert.True(result.Success, result.Error ?? "build failed");

            var session = result.Session!;
            session.MaxNewTokens = 32;
            Log($"maxTokens={session.MaxTokens} maxNewTokens={session.MaxNewTokens}");

            var entries = new List<ChatStreamEntry>();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));
            var first = true;

            var run = session.StartChatAsync(
                () =>
                {
                    if (first)
                    {
                        first = false;
                        return Task.FromResult(ChatMessage.User("Hello! Please answer in one short sentence: 2 + 2 = ?"));
                    }
                    return Task.FromResult(ChatMessage.User("exit"));
                },
                entries.Add,
                cts.Token);
            Log("startchat launched");

            var completed = await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(90)));
            Log($"whenany done, run={run.IsCompleted}");
            if (!ReferenceEquals(completed, run))
                cts.Cancel();
            await Task.WhenAll(run);
            Log($"loop finished, entry count={entries.Count}");
            Assert.Same(run, completed); // if the delay won, the session loop hung with no entry

            // First turn must have produced streamed or terminal entries.
            Assert.Contains(entries, e =>
                e.Status is ChatStatus.Thinking or ChatStatus.Responding or ChatStatus.Complete);

            bool completedCleanly = entries.Any(e => e.IsComplete || e.Status is ChatStatus.Complete or ChatStatus.Interrupted);
            if (entries.Count > 0 && !completedCleanly)
                Assert.Fail("First turn produced entries but never completed.");
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }
}