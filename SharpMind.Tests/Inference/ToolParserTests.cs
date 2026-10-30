using System.Text.Json.Nodes;
using SharpMind.Core.AgentTools;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The call parser must stay lenient: every format and every key spelling that worked
/// before must still dispatch. These tests drive the real session parser end to end.
/// </summary>
public sealed class ToolParserTests
{
    private static IAgentBuilder Agent()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTool("Measure", "Measures.",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject { ["label"] = new JsonObject { ["type"] = "string" } },
                ["required"] = new JsonArray("label")
            },
            args => Task.FromResult("ok"));
        return agent;
    }

    /// <summary>Runs one turn whose model reply is <paramref name="call"/>, returns dispatched calls.</summary>
    private static async Task<List<JsonObject>> Dispatched(ToolCallFormat format, string call)
    {
        await using var session = ScriptedSession.Create(Agent(), format, call, " Done.");
        var calls = new List<JsonObject>();
        session.ProcessToolRequest = (name, args, _) =>
        {
            calls.Add(new JsonObject { ["tool"] = name, ["arguments"] = JsonNode.Parse(args.ToJsonString()) });
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };
        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }
        return calls;
    }

    [Theory]
    [InlineData("<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}</tool_call>")]
    [InlineData("<tool_call>{\"tool\":\"Measure\",\"arguments\":{\"label\":\"red\"}}</tool_call>")]
    [InlineData("<tool_call>{\"name\":\"Measure\",\"args\":{\"label\":\"red\"}}</tool_call>")]
    public async Task SharpMindTags_AcceptNameOrToolKey(string call)
    {
        var calls = await Dispatched(ToolCallFormat.SharpMind, call);
        Assert.Single(calls);
        Assert.Equal("Measure", calls[0]["tool"]!.GetValue<string>());
        Assert.Equal("red", calls[0]["arguments"]!["label"]!.GetValue<string>());
    }

    [Theory]
    [InlineData(ToolCallFormat.Qwen, "{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}")]
    [InlineData(ToolCallFormat.Llama3, "{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}")]
    [InlineData(ToolCallFormat.Mistral, "[TOOL_CALLS]{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}")]
    [InlineData(ToolCallFormat.Gemma, "{\"name\":\"Measure\",\"args\":{\"label\":\"red\"}}")]
    [InlineData(ToolCallFormat.Gemma, "{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}")]
    [InlineData(ToolCallFormat.QwenXml, "<tool_call>\n<function=Measure>\n<parameter=label>\nred\n</parameter>\n</function>\n</tool_call>")]
    public async Task EveryFormat_DispatchesItsCanonicalCall(ToolCallFormat format, string call)
    {
        var calls = await Dispatched(format, call);
        Assert.Single(calls);
        Assert.Equal("Measure", calls[0]["tool"]!.GetValue<string>());
        Assert.Equal("red", calls[0]["arguments"]!["label"]!.GetValue<string>());
    }
}
