using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Synthetic tests for tool-call markup handling in ChatSession:
/// Qwen-instruct's native {"name":...,"arguments":...} shape, stray
/// &lt;tool_call&gt; markup that never parses, and the fabricated
/// {"status":...,"data":...} envelope a 0.5B model prints instead of
/// actually calling a tool. No real model files are referenced.
/// </summary>
public sealed class ToolCallMarkupBehaviorTests
{
    private const string FinalReply = "Done";

    private static IAgentBuilder BuilderWithNativeTool()
        => new AgentBuilder("Test").WithTool(
            "NativeTool",
            "A native SharpMind tool",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject(),
                ["required"] = new JsonArray()
            },
            args => Task.FromResult("took " + (args?["x"]?.GetValue<string>() ?? "?")));

    private static string RespondingText(IEnumerable<ChatStreamEntry> entries)
        => string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));

    /// <summary>
    /// Qwen2-instruct is trained on {"name":...,"arguments":...} rather than
    /// SharpMind's {"tool":...}. TryParseJsonObject now treats "name" as an
    /// alias and normalizes it into "tool", so the call dispatches.
    /// </summary>
    [Fact]
    public async Task QwenNameShape_RawJson_DispatchesTool()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"1"}}""",
            FinalReply);
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.Contains(FinalReply, RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task QwenNameShape_WrappedInToolCallTag_DispatchesTool()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            """<tool_call>{"name":"NativeTool","arguments":{"x":"1"}}</tool_call>""",
            FinalReply);
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.Contains(FinalReply, RespondingText(entries));
    }

    /// <summary>
    /// The user-observed qwen2-0.5B failure: the model replies only with the
    /// tool-result envelope it was told about in "Final Response Format"
    /// instead of genuinely calling a tool. The envelope must not reach the
    /// transcript as prose, and no tool runs at all.
    /// </summary>
    [Fact]
    public async Task FabricatedToolResultEnvelope_IsNeverShownAsProse()
    {
        int dispatchCalls = 0;
        var builder = BuilderWithNativeTool();
        // One reply streamed in several fragments: a model that stalls on the
        // envelope prints it in one continuous generation, not across turns.
        await using var session = ScriptedSession.Create(
            builder,
            "{" +
            "|\"status\":\"success\"," +
            "|\"data\":{\"prompt\":\"What are the three colors?\"," +
            "|\"options\":[\"Red\",\"Green\",\"Blue\"]}}");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            dispatchCalls++;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("can you call a tool to list me 3 options. red, green and blue"))
            entries.Add(e);

        // No tool ran (the JSON was never a call) and no fragment of it leaked.
        Assert.Equal(0, dispatchCalls);
        Assert.DoesNotContain("status", RespondingText(entries));
        Assert.DoesNotContain("Red", RespondingText(entries));
        Assert.DoesNotContain("options", RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task StrayUnclosedToolCallMarkup_IsNotShownAsProse()
    {
        await using var session = ScriptedSession.Create(
            "Intro ",
            "<tool_call>{\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"1\"}}");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("Intro ", RespondingText(entries));
        Assert.DoesNotContain("<tool_call>", RespondingText(entries));
        Assert.DoesNotContain("tool", RespondingText(entries));
    }

    /// <summary>
    /// A legitimate pure-JSON answer (one the user would want shown) must not
    /// be suppressed by the envelope guard.
    /// </summary>
    [Fact]
    public async Task LegitimatePureJsonAnswer_IsStillShown()
    {
        await using var session = ScriptedSession.Create("{\"answer\":\"red\"}");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("{\"answer\":\"red\"}", RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    /// <summary>
    /// When the model emits the envelope and then continues with real prose,
    /// only the prose is shown — the dropped envelope leaves no trace.
    /// </summary>
    /// <summary>
    /// The prompt contract models are now asked to follow: narrate in prose
    /// and embed the call in explicit tags. The call must dispatch and the
    /// surrounding narration must survive as visible prose with the markup
    /// stripped.
    /// </summary>
    [Fact]
    public async Task NarratedToolCall_DispatchesAndKeepsProse()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            "I'll check the inventory. \u003Ctool_call\u003E{\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"7\"}}\u003C/tool_call\u003E",
            " Done checking.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        string prose = RespondingText(entries);
        Assert.Contains("I'll check the inventory.", prose);
        Assert.Contains("Done checking.", prose);
        Assert.DoesNotContain("<tool_call>", prose);
    }

    /// <summary>
    /// The user-observed qwen2-0.5B shape: the delta showed a bare
    /// "<tool_call" (no '>') and no closing tag at all. The tool must still
    /// dispatch, and the tag prefix must not leak into the visible prose.
    /// </summary>
    [Fact]
    public async Task SloppyOpenTagNoClose_DispatchesAndShowsOnlyNarration()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            "I'll call now. \u003Ctool_call {\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"2\"}}",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        string prose = RespondingText(entries);
        Assert.Contains("I'll call now.", prose);
        Assert.Contains(" Done.", prose);
        Assert.DoesNotContain("tool_call", prose);
    }

    /// <summary>
    /// The closing tag is optional at dispatch time: a call that opens
    /// <c>&lt;tool_call&gt;</c> (or even just <c>&lt;tool_call</c>) and emits
    /// the JSON but never closes must still run. See a 0.5B model stopping
    /// right after the payload.
    /// </summary>
    [Fact]
    public async Task ClosedOpenTagNoClose_DispatchesTool()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            "\u003Ctool_call\u003E{\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"4\"}}",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.DoesNotContain("tool_call", RespondingText(entries));
    }

    /// <summary>
    /// The JSON payload itself can be cut off before its closing brace (token
    /// limit / EOS). TryParseJsonObject's truncation repair must still resolve
    /// the call so it dispatches instead of dumping the raw fragment.
    /// </summary>
    [Fact]
    public async Task TruncatedJsonAfterTag_DispatchesTool()
    {
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            "\u003Ctool_call {\"tool\":\"NativeTool\",\"arguments\":{\"x\":\"3\"}",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.DoesNotContain("tool_call", RespondingText(entries));
    }

    /// <summary>
    /// A partial "<tool_call" streaming fragment (the exact delta the user saw)
    /// must never reach the UI as prose — it is held back and discarded once
    /// the turn ends without completing into a call.
    /// </summary>
    [Fact]
    public async Task PartialTagFragment_NeverLeaksAsProse()
    {
        await using var session = ScriptedSession.Create("Sure. |\u003Ctool_call");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        string prose = RespondingText(entries);
        Assert.Equal("Sure. ", prose);
        Assert.DoesNotContain("tool_call", prose);
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    /// <summary>
    /// The observed qwen2-0.5B failure: for a request with no matching tool the
    /// model replies with a bare {"name":...} object that never becomes a call.
    /// It must not reach the transcript as prose; the session corrects the model
    /// and regenerates a plain answer instead.
    /// </summary>
    [Fact]
    public async Task QwenFailedCallShape_NameWithoutArguments_IsNeverShownAndRegenerates()
    {
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCallFormat.Qwen,
            """{"name":"Troll Troll","description":"a towering troll"}""",
            FinalReply);

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("make up a character name for a troll in a story"))
            entries.Add(e);

        string prose = RespondingText(entries);
        Assert.DoesNotContain("Troll Troll", prose);
        Assert.DoesNotContain("description", prose);
        Assert.Contains(FinalReply, prose);
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task EnvelopeFollowedByProse_ShowsOnlyProse()
    {
        await using var session = ScriptedSession.Create(
            "{\"status\":\"success\",\"data\":\"ok\"}| And here is the real prose.");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal(" And here is the real prose.", RespondingText(entries));
        Assert.DoesNotContain("status", RespondingText(entries));
    }
}