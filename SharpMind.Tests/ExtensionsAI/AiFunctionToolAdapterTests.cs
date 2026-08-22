using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;
using SharpMind.Inference.Agent;

namespace SharpMind.Tests.ExtensionsAI;

/// <summary>
/// Tests for <see cref="AiFunctionToolAdapter"/>: bridging MEAI
/// <see cref="AIFunction"/> instances into SharpMind's
/// <see cref="IAgentBuilder"/> tool infrastructure.
/// </summary>
public sealed class AiFunctionToolAdapterTests
{
    [Fact]
    public void RegisterTools_NullTools_NoOp()
    {
        var builder = new AgentBuilder("Test");
        AiFunctionToolAdapter.RegisterTools(builder, null);
        Assert.Empty(builder.RegisteredToolNames);
    }

    [Fact]
    public void RegisterTools_EmptyList_NoOp()
    {
        var builder = new AgentBuilder("Test");
        AiFunctionToolAdapter.RegisterTools(builder, new List<AITool>());
        Assert.Empty(builder.RegisteredToolNames);
    }

    [Fact]
    public void RegisterTools_RegistersAIFunction()
    {
        var builder = new AgentBuilder("Test");
        var fn = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "Weather",
            description: "Get weather for a city");
        var tools = new List<AITool> { fn };

        AiFunctionToolAdapter.RegisterTools(builder, tools);

        Assert.Contains("Weather", builder.RegisteredToolNames);
        Assert.Single(builder.ToolDefinitions);
        var def = (JsonObject)builder.ToolDefinitions[0]!;
        Assert.Equal("Weather", def["name"]?.GetValue<string>());
        Assert.Equal("Get weather for a city", def["description"]?.GetValue<string>());
    }

    [Fact]
    public void RegisterTools_SkipsNonAIFunctionTools()
    {
        var builder = new AgentBuilder("Test");
        var tools = new List<AITool> { new FakeTool("NotAFunction") };

        AiFunctionToolAdapter.RegisterTools(builder, tools);

        Assert.Empty(builder.RegisteredToolNames);
    }

    [Fact]
    public async Task RegisterTools_InvocationWorks()
    {
        var builder = new AgentBuilder("Test");
        var fn = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "Weather",
            description: "Get weather for a city");
        var tools = new List<AITool> { fn };

        AiFunctionToolAdapter.RegisterTools(builder, tools);

        var toolCall = new JsonObject
        {
            ["tool"] = "Weather",
            ["arguments"] = new JsonObject { ["city"] = "London" }
        };

        var result = await builder.CallToolAsync(toolCall);
        Assert.Equal("success", result["status"]?.GetValue<string>());
        Assert.Equal("Weather in London", result["data"]?.GetValue<string>());
    }

    [Fact]
    public void RegisterTools_DuplicateName_OnlyFirstRegistered()
    {
        var builder = new AgentBuilder("Test");
        var fn1 = AIFunctionFactory.Create(
            () => "first",
            name: "Dup",
            description: "First");
        var fn2 = AIFunctionFactory.Create(
            () => "second",
            name: "Dup",
            description: "Second");
        var tools = new List<AITool> { fn1, fn2 };

        AiFunctionToolAdapter.RegisterTools(builder, tools);

        Assert.Single(builder.RegisteredToolNames);
        Assert.Contains("Dup", builder.RegisteredToolNames);
    }

    private sealed class FakeTool(string name) : AITool
    {
        public override string Name => name;
    }
}
