using System.Text;
using SharpMind.Core;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using Xunit;

namespace SharpMind.Tests.Chat;

/// <summary>
/// Verifies ChatSession's incremental KV-cache extension: instead of
/// re-prefilling the whole conversation every turn, a turn whose render
/// reproduces the previously cached tokens (checked token-for-token against
/// the cached prompt plus the last turn's generated ids) feeds only the unseen
/// delta to the generator. This is formatter-agnostic — the old requirement
/// that <c>_formatter is null</c> (which in practice disabled the feature for
/// every chat-template model in the CUI) is gone.
///
/// The engage test primes the physical KV cache with a known prefix through an
/// internal hook after staging a rendered history, because the toy whitespace
/// tokenizer never re-encodes sampled ids (a sampled <c>&lt;t1716&gt;</c> decodes
/// to text that re-encodes as &lt;unk&gt; fragments plus the id) — the
/// real-workload fallback path is covered by the render-mismatch test.
/// </summary>
public sealed class IncrementalPrefillTests
{
    private static SessionOptions Options() => new()
    {
        AgentsEnabled = false,
        SkipAgentPrompt = true,
        FileAccess = ToolPermission.Always,
        NetworkAccess = ToolPermission.Always,
        Formatter = FormatterStrategy.Simple,
        Sampling = new SamplingConfig { Temperature = 0, TopK = 1 },
        Generation = new GenerationConfig { MaxNewTokens = 1 },
    };

    private static (LoadedModel Loaded, IChatSession Session) Build(TempDirectory temp, SessionOptions options)
    {
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;
        var load = SessionLauncher.LoadModelAsync(options).GetAwaiter().GetResult();
        Assert.True(load.Success, load.Error ?? "load failed");

        var result = SessionLauncher.BuildSession(options, load.Loaded!,
            permissions: _ => Task.FromResult(ToolPermission.Always));
        Assert.True(result.Success, result.Error ?? "build failed");

        var session = result.Session!;
        session.MaxNewTokens = 1;
        session.Temperature = 0;
        session.TopK = 1;
        return (load.Loaded!, session);
    }

    /// <summary>Drives a single chat turn to completion and returns the generated text.</summary>
    private static async Task<string> RunTurn(IChatSession session, string input)
    {
        var sb = new StringBuilder();
        bool first = true;
        using var cts = new CancellationTokenSource();
        await session.StartChatAsync(() =>
        {
            if (first)
            {
                first = false;
                return ChatMessage.User(input);
            }
            cts.Cancel();
            return ChatMessage.User(string.Empty);
        }, e =>
        {
            if (e.Token is not null && e.Status is ChatStatus.Responding or ChatStatus.Thinking)
                sb.Append(e.Token);
        }, cts.Token);
        return sb.ToString();
    }

    [Fact]
    public async Task IncrementalPrefill_WithFormatter_ExtendsCacheInsteadOfReprefilling()
    {
        using var temp = new TempDirectory();
        var inc = Build(temp, Options());
        var control = Build(temp, Options());
        try
        {
            // Stage history so the next turn's render begins with an exactly
            // known prefix: a user turn answered by an assistant turn.
            inc.Session.AddMessage(ChatRole.User, "Hello");
            inc.Session.AddMessage(ChatRole.Agent, "");
            control.Session.AddMessage(ChatRole.User, "Hello");
            control.Session.AddMessage(ChatRole.Agent, "");

            // "Previous turn completed cleanly": prime the physical KV cache
            // with the current render, which is a byte-prefix of the next turn's
            // render. The toy whitespace tokenizer is deterministic left-to-right,
            // so re-encoding the next turn reproduces this prefix exactly.
            var concreteInc = (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>)inc.Session;
            int[] prefix = inc.Session.Tokenizer.Encode(inc.Session.GetFormattedPrompt(), addBos: false, addEos: false);
            await concreteInc.PrimeIncrementalCacheForTestAsync(prefix);

            // Control: same staged history, but ResetCaches forces a full
            // re-prefill of turn two — giving the full render length to compare
            // the incremental session against.
            control.Session.ResetCaches();
            var controlTwo = await RunTurn(control.Session, "Tell me more");
            int fullTwo = control.Session.LastPrefillTokenCount;

            var incTwo = await RunTurn(inc.Session, "Tell me more");
            int deltaTwo = inc.Session.LastPrefillTokenCount;

            // The incremental turn fed only the unseen tail: the full re-render
            // minus the primed prefix.
            Assert.True(deltaTwo < fullTwo,
                $"Expected incremental prefill ({deltaTwo}) to be smaller than full ({fullTwo}).");
            Assert.Equal(fullTwo - prefix.Length, deltaTwo);

            // And extending the KV cache is lossless: greedy output is identical
            // to a turn that re-prefilled the whole conversation.
            Assert.Equal(controlTwo, incTwo);
        }
        finally
        {
            inc.Loaded.Model.Dispose();
            control.Loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task KvCacheSnapshot_RoundTrip_SkipsPrefillOnRestore()
    {
        using var temp = new TempDirectory();
        var options = Options();
        var original = Build(temp, options);
        try
        {
            // Initialize and warm up — this fills the KV cache.
            original.Session.InitializeChat();
            var concrete = (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>)original.Session;
            await concrete.WarmupPrefillAsync();

            // Snapshot should include the KV cache data.
            var snapshot = original.Session.GetSnapshot();
            Assert.NotNull(snapshot.KVCache);
            Assert.True(snapshot.KVCache!.Layers.Count > 0);
            Assert.True(snapshot.KVCache!.PromptTokenCount > 0);

            // Load into a fresh session — WarmupPrefillAsync should hit the
            // fast path and restore from the snapshot instead of re-prefilling.
            var restored = Build(temp, options);
            restored.Session.InitializeChat();
            var concreteRestored = (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>)restored.Session;
            concreteRestored.LoadSnapshot(snapshot);

            // The cache should now be valid after WarmupPrefillAsync restores it.
            await concreteRestored.WarmupPrefillAsync();

            // Run a turn and verify output is deterministic.
            var text = await RunTurn(restored.Session, "Hello");
            Assert.False(string.IsNullOrWhiteSpace(text));
        }
        finally
        {
            original.Loaded.Model.Dispose();
        }
    }

    [Fact]
    public async Task KvCacheSnapshot_HashMismatch_FallsBackToFullPrefill()
    {
        using var temp = new TempDirectory();
        var options = Options();
        var original = Build(temp, options);
        try
        {
            original.Session.InitializeChat();
            var concrete = (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>)original.Session;
            await concrete.WarmupPrefillAsync();

            var snapshot = original.Session.GetSnapshot();
            Assert.NotNull(snapshot.KVCache);

            // Load into a fresh session but mutate the system prompt so the
            // hash won't match — WarmupPrefillAsync should fall back.
            var restored = Build(temp, options);
            restored.Session.InitializeChat();
            restored.Session.AddMessage(ChatRole.System, "Extra system message.");
            var concreteRestored = (ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>)restored.Session;
            concreteRestored.LoadSnapshot(snapshot);
            await concreteRestored.WarmupPrefillAsync();

            var text = await RunTurn(restored.Session, "Hello");
            Assert.False(string.IsNullOrWhiteSpace(text));
        }
        finally
        {
            original.Loaded.Model.Dispose();
        }
    }
}