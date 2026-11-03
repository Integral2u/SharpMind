using System.Text.Json.Nodes;
using SharpMind.Core;
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

    public sealed class WeatherProbe
    {
        [ToolDesc("Gets the weather.")]
        public string GetWeather(
            [ToolDesc("City.")] string city,
            [ToolDesc("Unit.")] string unit)
            => $"{city} {unit}";
    }

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

    private static JsonObject ChoiceSchema() => new()
    {
        ["type"] = "object",
        ["properties"] = new JsonObject
        {
            ["prompt"] = new JsonObject { ["type"] = "string" },
            ["choices"] = new JsonObject { ["type"] = "string" }
        },
        ["required"] = new JsonArray("prompt", "choices")
    };

    private static IAgentBuilder BuilderWithChoiceTool()
        => new AgentBuilder("Test").WithTool(
            "present_choices",
            "Present the player's next choices",
            ChoiceSchema(),
            args => Task.FromResult("ok"));

    private static string RespondingText(IEnumerable<ChatStreamEntry> entries)
        => string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));

    private static Task<(string? Tool, JsonObject? Args, List<ChatStreamEntry> Entries)> RunAsync(
        IAgentBuilder builder,
        string reply,
        params string[] extraReplies)
        => RunAsync(builder, ToolCallRecoveryMode.Lenient, reply, extraReplies);

    private static async Task<(string? Tool, JsonObject? Args, List<ChatStreamEntry> Entries)> RunAsync(
        IAgentBuilder builder,
        ToolCallRecoveryMode mode,
        string reply,
        params string[] extraReplies)
    {
        string? seenTool = null;
        JsonObject? seenArgs = null;
        await using var session = ScriptedSession.Create(builder, [reply, .. extraReplies]);
        session.ToolCallRecovery = mode;
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

    private static IAgentBuilder BuilderWithNoRequiredTool(string name)
        => new AgentBuilder("Test").WithTool(
            name,
            "Takes no arguments",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject()
            },
            _ => Task.FromResult("ok"));

    private static IAgentBuilder BuilderWithPathTool(string name)
        => new AgentBuilder("Test").WithTool(
            name,
            "Reads a file",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["path"] = new JsonObject { ["type"] = "string" }
                },
                ["required"] = new JsonArray("path")
            },
            _ => Task.FromResult("ok"));

    private static IAgentBuilder BuilderWithAlternateChoiceTool(string name)
        => new AgentBuilder("Test").WithTool(
            name,
            "Present the player's next choices",
            ChoiceSchema(),
            _ => Task.FromResult("ok"));

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

    // ── argument-name normalization ────────────────────────────────

    [Fact]
    public async Task ArgumentKeys_AreNormalizedBeforeDispatch()
    {
        var builder = new AgentBuilder("Test").WithTool(
            "Act",
            "Takes an action type",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["actionType"] = new JsonObject { ["type"] = "string" }
                },
                ["required"] = new JsonArray("actionType")
            },
            _ => Task.FromResult("ok"));

        var (tool, args, _) = await RunAsync(
            builder,
            """{"tool":"Act","arguments":{"action_type":"battle"}}""",
            FinalReply);

        Assert.Equal("Act", tool);
        Assert.Equal("battle", args?["actionType"]?.GetValue<string>());
        Assert.Null(args?["action_type"]);
    }

    // ── choice-tool generalization ─────────────────────────────────

    [Fact]
    public async Task ChoicePayload_InfersDifferentlyNamedChoiceTool()
    {
        var (tool, _, _) = await RunAsync(
            BuilderWithAlternateChoiceTool("offer_choices"),
            """{"prompt":"What now?","options":["Go north","Wait"]}""",
            FinalReply);

        Assert.Equal("offer_choices", tool);
    }

    [Fact]
    public async Task ChoicePayload_PrefersExactPresentChoices()
    {
        var builder = BuilderWithChoiceTool().WithTool(
            "offer_choices",
            "Present choices",
            ChoiceSchema(),
            _ => Task.FromResult("ok"));

        var (tool, _, _) = await RunAsync(
            builder,
            """{"prompt":"What now?","options":["Go north","Wait"]}""",
            FinalReply);

        Assert.Equal("present_choices", tool);
    }

    // ── Loose fuzzy recovery ───────────────────────────────────────

    [Fact]
    public async Task Loose_RecoversFuzzyToolName()
    {
        var (tool, _, _) = await RunAsync(
            BuilderWithNoRequiredTool("ping"),
            ToolCallRecoveryMode.Loose,
            """{"tool":"pnig","arguments":{}}""",
            FinalReply);

        Assert.Equal("ping", tool);
    }

    [Fact]
    public async Task Lenient_DoesNotRecoverFuzzyToolName()
    {
        // The unknown name still reaches dispatch (historical behaviour); it is
        // simply not guessed into the registered tool.
        var (tool, _, _) = await RunAsync(
            BuilderWithNoRequiredTool("ping"),
            """{"tool":"pnig","arguments":{}}""",
            FinalReply);

        Assert.Equal("pnig", tool);
    }

    [Fact]
    public async Task Loose_RepairsFuzzyArgumentName()
    {
        var (tool, args, _) = await RunAsync(
            BuilderWithChoiceTool(),
            ToolCallRecoveryMode.Loose,
            """{"tool":"present_choices","arguments":{"promt":"What now?","choices":"Go north"}}""",
            FinalReply);

        Assert.Equal("present_choices", tool);
        Assert.Equal("What now?", args?["prompt"]?.GetValue<string>());
        Assert.Null(args?["promt"]);
    }

    [Fact]
    public async Task Lenient_LeavesFuzzyArgumentNameAlone()
    {
        var (_, args, _) = await RunAsync(
            BuilderWithChoiceTool(),
            """{"tool":"present_choices","arguments":{"promt":"What now?","choices":"Go north"}}""",
            FinalReply);

        Assert.Null(args?["prompt"]);
        Assert.Equal("What now?", args?["promt"]?.GetValue<string>());
    }

    [Fact]
    public async Task Loose_DoesNotGuessAToolForAPathArgument()
    {
        // The unknown name still reaches dispatch and fails there; it is never
        // resolved to the file tool the argument shape suggests.
        var (tool, _, _) = await RunAsync(
            BuilderWithPathTool("read_file"),
            ToolCallRecoveryMode.Loose,
            """{"tool":"reed_file","arguments":{"path":"C:\\secret.txt"}}""",
            FinalReply);

        Assert.Equal("reed_file", tool);
        Assert.NotEqual("read_file", tool);
    }

    [Fact]
    public async Task Lenient_StillInfersAToolForAPathArgument()
    {
        // The path guard is a Loose safety rule; Lenient keeps the historical
        // argument-shape inference.
        var (tool, _, _) = await RunAsync(
            BuilderWithPathTool("read_file"),
            """{"tool":"reed_file","arguments":{"path":"C:\\secret.txt"}}""",
            FinalReply);

        Assert.Equal("read_file", tool);
    }

    [Fact]
    public async Task Loose_FuzzyToolName_RequiresMatchingArguments()
    {
        var builder = new AgentBuilder("Test").WithTools(new WeatherProbe());
        var (tool, _, _) = await RunAsync(
            builder,
            ToolCallRecoveryMode.Loose,
            """{"tool":"GetWheather","arguments":{"q":"x"}}""",
            FinalReply);

        // A near-miss name alone is not enough: the required args must fit.
        Assert.Equal("GetWheather", tool);
    }

    [Fact]
    public async Task Loose_FuzzyToolName_ResolvesWhenArgumentsFit()
    {
        var builder = new AgentBuilder("Test").WithTools(new WeatherProbe());
        var (tool, args, _) = await RunAsync(
            builder,
            ToolCallRecoveryMode.Loose,
            """{"tool":"GetWheather","arguments":{"city":"Paris","unit":"celsius"}}""",
            FinalReply);

        Assert.Equal("GetWeather", tool);
        Assert.Equal("Paris", args?["city"]?.GetValue<string>());
    }

    [Fact]
    public async Task Loose_ExactToolNameWithPath_StillDispatches()
    {
        var (tool, _, _) = await RunAsync(
            BuilderWithPathTool("read_file"),
            ToolCallRecoveryMode.Loose,
            """{"tool":"read_file","arguments":{"path":"C:\\secret.txt"}}""",
            FinalReply);

        Assert.Equal("read_file", tool);
    }

    [Fact]
    public void DuplicateNormalizedToolNames_AreRefused()
    {
        var builder = (AgentBuilder)new AgentBuilder("Test")
            .WithTool("save_file", "Save", new JsonObject { ["type"] = "object" }, _ => Task.FromResult("ok"))
            .WithTool("save-file", "Save again", new JsonObject { ["type"] = "object" }, _ => Task.FromResult("ok"));

        Assert.Single(builder.RegisteredToolNames);
        Assert.Contains(builder.RegistrationWarnings, warning => warning.Contains("conflicts"));
    }

    // ── retry advice ───────────────────────────────────────────────

    [Fact]
    public async Task MissingArgument_AdviceUsesTheActiveFormat()
    {
        // SharpMind results use the System role, which the public History hides,
        // so format the advice directly (the XML test below covers the loop).
        var builder = new AgentBuilder("Test").WithTools(new WeatherProbe());
        await using var session = ScriptedSession.Create(builder, FinalReply);

        var args = new JsonObject { ["city"] = "Paris" };
        var result = new JsonObject
        {
            ["status"] = "error",
            ["code"] = "invalid_arguments",
            ["message"] = "Required argument 'unit' is missing."
        };

        string text = session.FormatToolResult("GetWeather", args, result);

        Assert.Contains("Did you mean:", text);
        Assert.Contains("<tool_call>", text);
        Assert.Contains("\"name\":\"GetWeather\"", text);
        Assert.Contains("\"city\":\"Paris\"", text);
        Assert.Contains("\"unit\":\"example\"", text);
    }

    [Fact]
    public async Task MissingArgument_AdviceUsesXmlFormat()
    {
        var builder = new AgentBuilder("Test").WithTools(new WeatherProbe());
        await using var session = ScriptedSession.Create(
            builder,
            ToolCallFormat.QwenXml,
            "<tool_call><function=GetWeather><parameter=city>Paris</parameter></function></tool_call>",
            FinalReply);

        await foreach (var _ in session.GetResponseStreamAsync("weather")) { }

        var toolMessage = session.History.FirstOrDefault(m => m.Content.Contains("ERROR:"));
        Assert.NotNull(toolMessage);
        Assert.Contains("<parameter=unit>", toolMessage.Content);
        Assert.Contains("example", toolMessage.Content);
    }

    [Fact]
    public async Task UnknownTool_AdviceSuggestsClosestName()
    {
        var builder = new AgentBuilder("Test").WithTools(new WeatherProbe());
        await using var session = ScriptedSession.Create(builder, FinalReply);

        var result = new JsonObject
        {
            ["status"] = "error",
            ["code"] = "unknown_tool",
            ["message"] = "Unknown tool: 'GetWether'. Valid tools: GetWeather.",
            ["suggestedTool"] = "GetWeather"
        };

        string text = session.FormatToolResult(
            "GetWether", new JsonObject { ["q"] = "x" }, result);

        Assert.Contains("Did you mean:", text);
        Assert.Contains("GetWeather", text);
        Assert.DoesNotContain("\"q\"", text);
    }
}
