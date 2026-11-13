using System.Text.Json.Nodes;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Delegation is a normal tool: delegate(agent, task). Its agents come from the builder,
/// and the old {{agent:...}} text form still works.
/// </summary>
public sealed class DelegationTests
{
    private static IAgentBuilder AgentWithReviewer()
    {
        var agent = new AgentBuilder("Test");
        agent.CreateAgent(new AgentConfig
        {
            Name = "Reviewer",
            SystemPrompt = "You review code for bugs.\nBe brief.",
            Purpose = "Reviews code and lists bugs",
            Temperature = 0.2f,
            Seed = 7,
        });
        agent.WithAgents(depth: 2);
        agent.RegisterDelegateTool();
        return agent;
    }

    private static string DelegateCall(string agent, string task) =>
        "<tool_call>{\"name\":\"delegate\",\"arguments\":{\"agent\":\"" + agent + "\",\"task\":\"" + task + "\"}}</tool_call>";

    private static async Task<string> Reply(ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder> session)
    {
        var text = new System.Text.StringBuilder();
        await foreach (var e in session.GetResponseStreamAsync("hi"))
            if (e.Status == ChatStatus.Responding) text.Append(e.Token);
        return text.ToString();
    }

    [Fact]
    public void DelegateTool_HasAgentEnumAndPurposes()
    {
        var agent = AgentWithReviewer();
        var tool = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "delegate");
        var props = tool["parameters"]!["properties"]!;

        Assert.Equal(["Reviewer"], props["agent"]!["enum"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.Contains("Reviews code and lists bugs", props["agent"]!["description"]!.GetValue<string>());
        Assert.Equal(["agent", "task"], tool["parameters"]!["required"]!.AsArray().Select(n => n!.GetValue<string>()));
    }

    [Fact]
    public void DelegateTool_HasNoModelChosenTemperatureOrSeed()
    {
        var agent = AgentWithReviewer();
        string schema = agent.ToolDefinitions.ToJsonString();
        Assert.DoesNotContain("temp", schema);
        Assert.DoesNotContain("seed", schema);
    }

    [Fact]
    public void AgentPurpose_FallsBackToTheFirstLineOfTheSystemPrompt()
    {
        var agent = new AgentBuilder("Test");
        var plain = agent.CreateAgent(new AgentConfig { Name = "Plain", SystemPrompt = "First line.\nSecond." });
        Assert.Equal("First line.", AgentBuilder.AgentPurpose(plain));
    }

    [Fact]
    public async Task DelegateCall_RunsTheSubAgent_AndFeedsBackItsAnswer()
    {
        // The parent delegates; the sub-agent answers from the next scripted reply.
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            DelegateCall("Reviewer", "Check this loop."),   // parent turn 1: the delegate call
            "Found one bug: off-by-one.",                   // sub-agent reply
            "The reviewer found an off-by-one bug.");       // parent turn 2: final answer

        var reply = await Reply(session);
        Assert.Contains("off-by-one", reply);
        // The result header names the tool and its key arguments, then the sub-agent's answer.
        Assert.Contains("[delegate agent=Reviewer task=Check this loop.] Found one bug: off-by-one.",
            string.Concat(session.RawHistoryForTests.Select(m => m.Content)));
    }

    [Fact]
    public async Task UnknownAgent_ReturnsAClearErrorListingTheValidOnes()
    {
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            DelegateCall("Nobody", "anything"),
            "Sorry, I could not find that specialist.");

        var reply = await Reply(session);
        string history = string.Concat(session.RawHistoryForTests.Select(m => m.Content));
        Assert.Contains("Unknown agent 'Nobody'. Valid agents: Reviewer", history);
        Assert.Contains("could not find", reply);
    }

    [Fact]
    public async Task DepthLimit_RefusesNestedDelegation()
    {
        var agent = new AgentBuilder("Test");
        agent.CreateAgent(new AgentConfig { Name = "Reviewer", SystemPrompt = "Reviews.", Purpose = "Reviews" });
        agent.WithAgents(depth: 0);      // no nesting allowed
        agent.RegisterDelegateTool();

        await using var session = ScriptedSession.Create(agent, ToolCallFormat.SharpMind,
            DelegateCall("Reviewer", "x"),
            "No specialist available; here is my own answer.");

        var reply = await Reply(session);
        string history = string.Concat(session.RawHistoryForTests.Select(m => m.Content));
        Assert.Contains("Maximum agent depth (0) reached", history);
        Assert.Contains("my own answer", reply);
    }

    [Fact]
    public async Task EmptyTask_IsRejectedWithAClearError()
    {
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            DelegateCall("Reviewer", ""),
            "Okay.");

        await Reply(session);
        string history = string.Concat(session.RawHistoryForTests.Select(m => m.Content));
        Assert.Contains("The task is empty", history);
    }

    [Fact]
    public async Task TextForm_StillParses_WithItsLegacyTempAndSeed()
    {
        // The {{agent:Name:temp=..:seed=..:query}} form remains accepted for older prompts.
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            "{{agent:Reviewer:temp=0.1:seed=3:check the code}}",
            "Legacy answer.",
            "Done.");

        var reply = await Reply(session);
        Assert.Contains("Done.", reply);
    }

    [Fact]
    public async Task Delegation_UsesASeparateGenerator_AndDoesNotResetTheParent()
    {
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            DelegateCall("Reviewer", "Check this loop."),
            "Found one bug: off-by-one.",
            "The reviewer found an off-by-one bug.");

        var parent = Assert.IsType<ScriptedGenerator>(session.Generator);
        int parentResetsAtStart = parent.ResetCalls;

        var reply = await Reply(session);

        Assert.Contains("off-by-one", reply);
        var sub = Assert.IsType<ScriptedGenerator>(session.SubAgentGeneratorForTests);
        Assert.NotSame(parent, sub);
        Assert.Equal(1, sub.ResetCalls);

        // Two parent model calls (the delegate call and the final answer). The
        // old shared-cache path added a third reset when the sub-agent ran.
        Assert.Equal(parentResetsAtStart + 2, parent.ResetCalls);
    }

    [Fact]
    public async Task TwoDelegations_ReuseOneSubAgentGenerator()
    {
        await using var session = ScriptedSession.Create(AgentWithReviewer(), ToolCallFormat.SharpMind,
            DelegateCall("Reviewer", "First task."),
            "First answer.",
            DelegateCall("Reviewer", "Second task."),
            "Second answer.",
            "Both done.");

        var reply = await Reply(session);

        Assert.Contains("Both done.", reply);
        var sub = Assert.IsType<ScriptedGenerator>(session.SubAgentGeneratorForTests);
        // One instance, reset before each one-shot task.
        Assert.Equal(2, sub.ResetCalls);
    }

    [Fact]
    public async Task EnginePath_ExportsAndRestoresTheParentCache()
    {
        var generator = new CacheTrackingGenerator();
        var engine = new FakeInferenceEngine();
        int[] tokens = [11, 12, 13];

        await ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>.PreserveParentCacheAsync(
            generator,
            engine,
            tokens,
            () =>
            {
                generator.ResetCache();
                return Task.CompletedTask;
            });

        Assert.Equal(1, engine.ExportCalls);
        Assert.Equal(1, engine.ImportCalls);
        Assert.Equal(tokens, generator.CacheTokens);
    }

    [Fact]
    public async Task EnginePath_RestoresTheParentCacheWhenTheSubAgentThrows()
    {
        var generator = new CacheTrackingGenerator();
        var engine = new FakeInferenceEngine();
        int[] tokens = [21, 22];

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>.PreserveParentCacheAsync(
                generator,
                engine,
                tokens,
                () =>
                {
                    generator.ResetCache();
                    throw new InvalidOperationException("sub-agent failed");
                }));

        Assert.Equal(1, engine.ImportCalls);
        Assert.Equal(tokens, generator.CacheTokens);
    }

    private sealed class CacheTrackingGenerator : IGenerator<KVCacherBuilder>
    {
        public string Name => "tracking";
        public Action<double>? PrefillProgress { get; set; }
        public float CacheFillRatio => 0;
        public float? TokensPerSecond => null;
        public float? CumulativeTokensPerSecond => null;
        public float? TimeToFirstToken => null;
        public IReadOnlyList<IKVCache> Caches => [];
        public IReadOnlyList<int>? CacheTokens { get; private set; }

        public void ResetCache() => CacheTokens = [];
        public void SetCacheTokens(IReadOnlyList<int> tokens) => CacheTokens = [.. tokens];
        public void Dispose() { }

        public IAsyncEnumerable<string> GenerateAsync(
            string prompt,
            SamplingConfig? sampling = null,
            GenerationConfig? generation = null,
            CancellationToken cancellationToken = default) => EmptyAsync();

        public IAsyncEnumerable<string> GenerateFromTokensAsync(
            int[] promptIds,
            SamplingConfig? sampling = null,
            GenerationConfig? generation = null,
            CancellationToken cancellationToken = default) => EmptyAsync();

        private static async IAsyncEnumerable<string> EmptyAsync()
        {
            await Task.CompletedTask;
            yield break;
        }
    }

    private sealed class FakeInferenceEngine : IInferenceEngine
    {
        public string Description => "fake";
        public int CachedLength => 0;
        public int MaxCacheLength => 1024;
        public bool IsCacheFull => false;
        public int ExportCalls { get; private set; }
        public int ImportCalls { get; private set; }

        public ReadOnlyMemory<float> Prefill(
            ReadOnlySpan<int> tokenIds,
            Action<double>? onChunkProgress = null,
            CancellationToken cancellationToken = default) => new float[2];

        public ReadOnlyMemory<float> DecodeStep(
            int tokenId,
            CancellationToken cancellationToken = default) => new float[2];

        public void TruncateCache(int length) { }
        public void TrimToLast(int keep) { }
        public void ResetCache() { }

        public KVCacheSnapshot ExportCache(int[] promptTokenIds)
        {
            ExportCalls++;
            return new KVCacheSnapshot
            {
                PromptHash = KVCacheSnapshot.HashPromptTokens(promptTokenIds),
                PromptTokenCount = promptTokenIds.Length,
                Layers = [],
            };
        }

        public void ImportCache(KVCacheSnapshot snapshot) => ImportCalls++;
        public void Dispose() { }
    }
}
