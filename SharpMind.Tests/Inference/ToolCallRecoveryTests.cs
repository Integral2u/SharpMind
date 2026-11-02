using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Structural recovery for tool calls the strict format parser cannot read.
/// Lenient is the session default; Strict reproduces the old behaviour. A call
/// that already parses strictly is never modified.
/// </summary>
public sealed class ToolCallRecoveryTests
{
    private const string FinalReply = "Done";

    private static IAgentBuilder BuilderWithNativeTool()
        => new AgentBuilder("Test").WithTool(
            "NativeTool",
            "A native SharpMind tool",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["x"] = new JsonObject { ["type"] = "string" }
                },
                ["required"] = new JsonArray("x")
            },
            args => Task.FromResult("took " + (args?["x"]?.GetValue<string>() ?? "?")));

    private static IAgentBuilder BuilderWithChoiceTool()
        => new AgentBuilder("Test").WithTool(
            "present_choices",
            "Present the player's next choices",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["prompt"] = new JsonObject { ["type"] = "string" },
                    ["choices"] = new JsonObject { ["type"] = "string" }
                },
                ["required"] = new JsonArray("prompt", "choices")
            },
            args => Task.FromResult("ok"));

    private static string RespondingText(IEnumerable<ChatStreamEntry> entries)
        => string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));

    private static async Task<(string? Tool, JsonObject? Args, List<ChatStreamEntry> Entries)> RunAsync(
        IAgentBuilder builder,
        string reply,
        params string[] extraReplies)
    {
        string? seenTool = null;
        JsonObject? seenArgs = null;
        await using var session = ScriptedSession.Create(builder, [reply, .. extraReplies]);
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenTool = toolName;
            seenArgs = args.DeepClone().AsObject();
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var entry in session.GetResponseStreamAsync("hi"))
            entries.Add(entry);
        return (seenTool, seenArgs, entries);
    }

    [Fact]
    public async Task ProseBeforeJson_RecoversCall()
    {
        var (tool, args, entries) = await RunAsync(
            BuilderWithNativeTool(),
            "I will check. {\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"9\"}}",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("9", args?["x"]?.GetValue<string>());
        Assert.Contains("I will check.", RespondingText(entries));
        Assert.Contains(FinalReply, RespondingText(entries));
    }

    [Fact]
    public async Task FencedJson_RecoversCall()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            "```json\n{\"name\":\"NativeTool\",\"arguments\":{\"x\":\"1\"}}\n```",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("1", args?["x"]?.GetValue<string>());
    }

    [Fact]
    public async Task StringifiedArguments_RecoversCall()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":"{\"x\":\"2\"}"}""",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("2", args?["x"]?.GetValue<string>());
    }

    [Fact]
    public async Task AliasedToolName_RecoversCall()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"tool":"native_tool","arguments":{"x":"3"}}""",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("3", args?["x"]?.GetValue<string>());
    }

    [Fact]
    public async Task MissingToolNameWithChoicePayload_InfersPresentChoices()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithChoiceTool(),
            """{"prompt":"What now?","choices":"Go north"}""",
            FinalReply);

        Assert.Equal("present_choices", tool);
        Assert.Equal("What now?", args?["prompt"]?.GetValue<string>());
        Assert.Equal("Go north", args?["choices"]?.GetValue<string>());
    }

    [Fact]
    public async Task UniqueSchemaMatch_InfersTool()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"x":"5"}""",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("5", args?["x"]?.GetValue<string>());
    }

    [Fact]
    public async Task AmbiguousSchemaMatch_DoesNotDispatch()
    {
        var builder = BuilderWithNativeTool().WithTool(
            "SecondTool",
            "Also requires x",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["x"] = new JsonObject { ["type"] = "string" }
                },
                ["required"] = new JsonArray("x")
            },
            _ => Task.FromResult("second"));

        var (tool, _, _) = await RunAsync(builder, """{"x":"5"}""", FinalReply);
        Assert.Null(tool);
    }

    [Fact]
    public async Task StrictMode_DoesNotRecover()
    {
        string? seenTool = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            "I will check. {\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"9\"}}",
            FinalReply);
        session.ToolCallRecovery = ToolCallRecoveryMode.Strict;
        session.ProcessToolRequest = (toolName, _, _) =>
        {
            seenTool = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var entry in session.GetResponseStreamAsync("hi"))
            entries.Add(entry);

        Assert.Null(seenTool);
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task ValidStrictCall_IsNotModified()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"tool":"NativeTool","arguments":{"x":"1"}}""",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Single(args!);
        Assert.Equal("1", args?["x"]?.GetValue<string>());
    }

    [Fact]
    public async Task OptionsPayload_InfersPresentChoices()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithChoiceTool(),
            """{"prompt":"What now?","options":["Go north","Wait"]}""",
            FinalReply);

        Assert.Equal("present_choices", tool);
        Assert.Equal(2, args?["options"]?.AsArray().Count);
    }

    [Fact]
    public async Task StrictParseOfUnregisteredTool_StillDispatches()
    {
        // The old behaviour is preserved: a well-formed call naming an unknown
        // tool reaches dispatch so it can be reported, unless recovery resolves
        // it to a registered tool.
        var (tool, _, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"tool":"NoSuchTool","arguments":{"y":"1"}}""",
            FinalReply);

        Assert.Equal("NoSuchTool", tool);
    }

    [Fact]
    public async Task StrictParseOfUnknownName_ResolvesAliasInstead()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithNativeTool(),
            """{"tool":"native_tool","arguments":{"x":"7"}}""",
            FinalReply);

        Assert.Equal("NativeTool", tool);
        Assert.Equal("7", args?["x"]?.GetValue<string>());
    }
}
