using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Headless reproduction of the CUI's session path: the exact
/// SessionLauncher.LoadModelAsync + BuildSession construction (agent builder
/// with tools, Auto formatter, MaxTokens = MaxSeqLen, SimpleArtifactPrompt
/// pre-processor, permission callback) driven via StartChatAsync the same
/// way ChatSessionBridge drives it.
///
/// The first turn is expected to (a) surface chunked-prefill progress as
/// "Prefilling NN.NN%" entries — the fix for the "stuck at Thinking..."
/// symptom, where a slow engine plus a long agent prompt looked like a hang —
/// and (b) actually stream a response within a generous window.
///
/// Driven by <see cref="TinyReferenceModel"/> (deterministic, seed-fixed,
/// millisecond-to-build reference .SMM) so the whole session plumbing is
/// exercised end-to-end without loading a real model file.
/// </summary>
public sealed class CuiSessionReproTests
{
    private static void Log(string message) =>
        File.AppendAllText(Path.Combine(Path.GetTempPath(), "cui_session_repro.log"),
            $"{DateTime.Now:HH:mm:ss.fff} {message}{Environment.NewLine}");

    [Fact]
    public async Task CuiSession_FirstTurn_StreamsWithPrefillProgress()
    {
        using var temp = new TempDirectory();
        Log($"start model load");

        var options = new SessionOptions
        {
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
            ShowThinking = true,
        };
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;

        // Disable every tool: an untrained model on a trivial question tries to
        // call tools, and each tool iteration re-prefills the whole growing
        // conversation (the Auto formatter forces a full rebuild; the tool call
        // resets the KV cache). That ballooned the first turn to many minutes
        // and made the harness look like a prefill hang when it was actually a
        // runaway tool loop. This test targets prefill + progress, so tools off.
        options.DisabledTools = SessionLauncher.GetAvailableTools(options).ToHashSet(StringComparer.Ordinal);

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

            // A long context message simulates real history so the first turn's
            // prompt exceeds one prefill chunk (progress entries require a
            // multi-chunk prompt; a bare 65-token prompt is a single shot).
            var filler = new System.Text.StringBuilder();
            while (true)
            {
                filler.Append("The quick brown fox jumps over the lazy dog. ");
                int tokens = load.Loaded!.Tokenizer.Encode(filler.ToString(), addBos: false, addEos: false).Length;
                if (tokens >= 480) break;
            }
            session.AddMessage(ChatRole.System, filler.ToString());
            Log($"maxTokens={session.MaxTokens} maxNewTokens={session.MaxNewTokens} fillerTokens={load.Loaded.Tokenizer.Encode(filler.ToString(), addBos: false, addEos: false).Length}");

            var entries = new System.Collections.Concurrent.ConcurrentQueue<ChatStreamEntry>();
            using var cts = new CancellationTokenSource();

            // First call supplies the real user message; the loop then asks for
            // the next input and this provider simply waits for the test to
            // cancel (the session loop has no "exit" sentinel — it generates for
            // whatever it's handed, so feeding it more text would re-prefill the
            // growing history forever).
            bool first = true;
            async Task<ChatMessage> NextMessage()
            {
                if (first)
                {
                    first = false;
                    return ChatMessage.User("Hello! Please answer in one short sentence: 2 + 2 = ?");
                }
                await Task.Delay(Timeout.Infinite, cts.Token);
                return ChatMessage.User(string.Empty);
            }

            var run = session.StartChatAsync(NextMessage, entries.Enqueue, cts.Token);
            Log("startchat launched");

            // Watch for progress + first streamed fragment, then stop the turn.
            // The reference model prefills in milliseconds, so the deadline only
            // guards a regression that turns prefill back into a hang.
            var deadline = DateTime.UtcNow.AddSeconds(120);
            bool sawPrefill = false;
            bool sawRespond = false;
            while (DateTime.UtcNow < deadline && !sawRespond)
            {
                await Task.Delay(100);
                foreach (var e in entries)
                {
                    if (e.Status == ChatStatus.Updating && e.Token?.StartsWith("Prefilling") == true)
                        sawPrefill = true;
                    if (e.Status == ChatStatus.Responding)
                        sawRespond = true;
                }
            }
            Log($"watch done sawPrefill={sawPrefill} sawRespond={sawRespond} entries={entries.Count}");

            cts.Cancel();
            await Task.WhenAny(run, Task.Delay(TimeSpan.FromSeconds(15)));

            // The prefill must not look like a hang: progress entries appear
            // while the prompt is being processed, and a response streams.
            Assert.True(sawPrefill, "Expected 'Prefilling NN.NN%' progress entries during the first turn.");
            Assert.True(sawRespond, "Expected the first turn to stream a response.");
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }
}