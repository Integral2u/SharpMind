using System.Text.Json.Nodes;
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
}
