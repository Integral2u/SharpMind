using Microsoft.Extensions.AI;
using SharpMind.Core;
using SharpMind.Extensions.AI;
using SharpMind.Inference.Agent;

namespace SharpMind.Tests.ExtensionsAI;

/// <summary>
/// Tests for the delegate-based <see cref="IAgentBuilder.WithTool"/> overload
/// added to support the Microsoft.Extensions.AI adapter.
/// </summary>
public sealed class AgentBuilderWithToolTests
{
    [Fact]
    public void WithTool_RegistersToolDefinition()
    {
        var builder = new AgentBuilder("TestAgent");
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject
            {
                ["query"] = new System.Text.Json.Nodes.JsonObject { ["type"] = "string" }
            },
            ["required"] = new System.Text.Json.Nodes.JsonArray("query")
        };

        builder.WithTool("Search", "Search the web", schema,
            args => System.Threading.Tasks.Task.FromResult("result"));

        Assert.Contains("Search", builder.RegisteredToolNames);
        Assert.Single(builder.ToolDefinitions);

        var def = (System.Text.Json.Nodes.JsonObject)builder.ToolDefinitions[0]!;
        Assert.Equal("Search", def["name"]?.GetValue<string>());
        Assert.Equal("Search the web", def["description"]?.GetValue<string>());
    }

    [Fact]
    public async Task WithTool_CanBeInvoked()
    {
        var builder = new AgentBuilder("TestAgent");
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject
            {
                ["name"] = new System.Text.Json.Nodes.JsonObject { ["type"] = "string" }
            },
            ["required"] = new System.Text.Json.Nodes.JsonArray("name")
        };

        builder.WithTool("Greet", "Say hello", schema,
            args => System.Threading.Tasks.Task.FromResult($"Hello, {args["name"]}!"));

        var toolCall = new System.Text.Json.Nodes.JsonObject
        {
            ["tool"] = "Greet",
            ["arguments"] = new System.Text.Json.Nodes.JsonObject { ["name"] = "World" }
        };

        var result = await builder.CallToolAsync(toolCall);
        Assert.Equal("success", result["status"]?.GetValue<string>());
        Assert.Equal("Hello, World!", result["data"]?.GetValue<string>());
    }

    [Fact]
    public async Task WithTool_DuplicateName_IsIgnored()
    {
        var builder = new AgentBuilder("TestAgent");
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject(),
            ["required"] = new System.Text.Json.Nodes.JsonArray()
        };

        builder.WithTool("Echo", "First", schema, _ => System.Threading.Tasks.Task.FromResult("first"));
        builder.WithTool("Echo", "Second", schema, _ => System.Threading.Tasks.Task.FromResult("second"));

        var toolCall = new System.Text.Json.Nodes.JsonObject
        {
            ["tool"] = "Echo",
            ["arguments"] = new System.Text.Json.Nodes.JsonObject()
        };

        var result = await builder.CallToolAsync(toolCall);
        Assert.Equal("first", result["data"]?.GetValue<string>());
    }

    [Fact]
    public void WithTool_DisabledTool_IsNotRegistered()
    {
        var builder = new AgentBuilder("TestAgent")
        {
            DisabledTools = { "Blocked" }
        };
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject(),
            ["required"] = new System.Text.Json.Nodes.JsonArray()
        };

        builder.WithTool("Blocked", "Should not register", schema,
            _ => System.Threading.Tasks.Task.FromResult("nope"));

        Assert.DoesNotContain("Blocked", builder.RegisteredToolNames);
        Assert.Empty(builder.ToolDefinitions);
    }

    [Fact]
    public void WithTool_NullOrEmptyName_Throws()
    {
        var builder = new AgentBuilder("TestAgent");
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject(),
            ["required"] = new System.Text.Json.Nodes.JsonArray()
        };

        Assert.ThrowsAny<ArgumentException>(() =>
            builder.WithTool("", "desc", schema, _ => System.Threading.Tasks.Task.FromResult("")));
        Assert.ThrowsAny<ArgumentException>(() =>
            builder.WithTool(null!, "desc", schema, _ => System.Threading.Tasks.Task.FromResult("")));
    }

    [Fact]
    public async Task WithTool_UnknownTool_ReturnsError()
    {
        var builder = new AgentBuilder("TestAgent");

        var toolCall = new System.Text.Json.Nodes.JsonObject
        {
            ["tool"] = "Nonexistent",
            ["arguments"] = new System.Text.Json.Nodes.JsonObject()
        };

        var result = await builder.CallToolAsync(toolCall);
        Assert.Equal("error", result["status"]?.GetValue<string>());
        Assert.Contains("Unknown tool", result["message"]?.GetValue<string>());
    }

    [Fact]
    public void WithTool_BothReflectionAndDelegate_Coexist()
    {
        var builder = new AgentBuilder("TestAgent");
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject(),
            ["required"] = new System.Text.Json.Nodes.JsonArray()
        };

        builder.WithTool("DelegateTool", "A delegate tool", schema,
            _ => System.Threading.Tasks.Task.FromResult("delegate"));

        builder.WithTools(new TestToolClass());

        Assert.Contains("DelegateTool", builder.RegisteredToolNames);
        Assert.Contains("Echo", builder.RegisteredToolNames);
    }

    private sealed class TestToolClass
    {
        [ToolDesc("Echo input")]
        public static System.Threading.Tasks.Task<string> Echo([ToolDesc("text to echo")] string text)
            => System.Threading.Tasks.Task.FromResult(text);
    }
}
