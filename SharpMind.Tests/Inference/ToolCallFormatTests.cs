using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Tool-call-format resolution (metadata → None / SharpMind / Qwen) and its
/// effect on the agent prompt and on session capture.
/// </summary>
public sealed class ToolCallFormatTests
{
    private static JsonObject Schema() => new()
    {
        ["type"] = "object",
        ["properties"] = new JsonObject
        {
            ["x"] = new JsonObject
            {
                ["type"] = "string",
                ["description"] = "A value for x"
            }
        },
        ["required"] = new JsonArray()
    };

    private static IAgentBuilder BuilderWithNativeTool()
        => new AgentBuilder("Test").WithTool(
            "NativeTool",
            "A native SharpMind tool",
            Schema(),
            args => Task.FromResult("took " + (args?["x"]?.GetValue<string>() ?? "?")));

    private static ModelMetaData Meta(params (string Key, string Value)[] kvs) => new()
    {
        KvPairs = [.. kvs.Select(k => new KvPair { Key = k.Key, Value = k.Value })]
    };

    private static string RespondingText(IEnumerable<ChatStreamEntry> entries)
        => string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));

    // ── Detector ────────────────────────────────────────────────────────────

    [Fact]
    public void Resolve_NullMeta_IsNone()
        => Assert.Equal(ToolCallFormat.None, ToolCallFormatDetector.Resolve(null));

    [Fact]
    public void Resolve_TemplateWithToolCallBlocks_IsQwen()
    {
        var meta = Meta(("tokenizer.chat_template",
            "<|im_start|>system\n{{p}}\n<|im_end|><|im_start|>user\n{{u}}<|im_end|><|im_start|>tool_calls\n{{c}}<|im_end|>"));
        Assert.Equal(ToolCallFormat.Qwen, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_QwenName_IsQwen()
    {
        var meta = Meta(
            ("general.name", "Qwen2-0.5B-Instruct"),
            ("general.basename", "Qwen2"),
            ("general.finetune", "instruct"));
        Assert.Equal(ToolCallFormat.Qwen, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_PlainName_IsSharpMind()
    {
        var meta = Meta(("general.name", "Delta-1B-Chat"));
        Assert.Equal(ToolCallFormat.SharpMind, ToolCallFormatDetector.Resolve(meta));
    }

    // ── Prompt shape ────────────────────────────────────────────────────────

    [Fact]
    public void SharpMindFormatDefault_StillTeachesTaggedCalls()
    {
        string prompt = BuilderWithNativeTool().BuildAgentPrompt();
        Assert.Contains("tool_call", prompt);
        Assert.Contains("\"tool\":", prompt);
        Assert.Contains("Narrate freely in prose", prompt);
    }

    [Fact]
    public void QwenFormat_PromptTeachesRawJsonOnly()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("{\"name\":", prompt);
        Assert.DoesNotContain("tool_call", prompt);
        Assert.DoesNotContain("Narrate freely in prose", prompt);
    }

    [Fact]
    public void QwenFormat_PromptTreatsToolRequestsAsOrders()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("can you", prompt);
        Assert.Contains("as an order, not a question", prompt);
        Assert.Contains("Never answer a tool request by describing", prompt);
    }

    [Fact]
    public void QwenFormat_ExampleUsesRealRegisteredToolName()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("\"NativeTool\"", prompt);
        Assert.Contains("\"x\":\"...\"", prompt);
        Assert.Contains("IMMEDIATELY emit the tool call JSON", prompt);
    }

    [Fact]
    public void NoneFormat_OmitsToolsSection()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.None;
        string prompt = builder.BuildAgentPrompt();
        Assert.DoesNotContain("## Available Tools", prompt);
        Assert.DoesNotContain("tool_call", prompt);
    }

    // ── Session capture ─────────────────────────────────────────────────────

    [Fact]
    public async Task QwenMetadata_ResultFedBackAsToolRole()
    {
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}""",
            " Received.");
        session.ProcessToolRequest = (toolName, args, ct) =>
            Task.FromResult(ToolRequestResult.Handled("external result"));

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        var toolMsg = session.History.FirstOrDefault(m => m.Role == ChatRole.Tool);
        Assert.NotNull(toolMsg);
        Assert.Contains("external result", toolMsg!.Content);

        // SharpMind-tagged sessions keep the legacy system framing so a Tool
        // message must NOT appear there.
        await using var tagged = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCallFormat.SharpMind,
            """<tool_call>{"tool":"NativeTool","arguments":{"x":"1"}}</tool_call>""",
            " Received.");
        tagged.ProcessToolRequest = (toolName, args, ct) =>
            Task.FromResult(ToolRequestResult.Handled("external result"));
        var taggedEntries = new List<ChatStreamEntry>();
        await foreach (var e in tagged.GetResponseStreamAsync("hi")) taggedEntries.Add(e);
        Assert.DoesNotContain(tagged.History, m => m.Role == ChatRole.Tool);
    }

    [Fact]
    public async Task QwenMetadata_RawJsonCall_Dispatches()
    {
        string? seenName = null;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"), ("general.finetune", "instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}""",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.Contains(" Done.", RespondingText(entries));
    }

    /// <summary>
    /// Qwen mode: the 0.5B model emits the raw call object and then keeps
    /// generating (apology prose). The call must still dispatch, and the raw
    /// JSON must not leak into the visible transcript as prose.
    /// </summary>
    [Fact]
    public async Task QwenMetadata_RawJsonCallFollowedByProse_DispatchesAndHidesMarkup()
    {
        string? seenName = null;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}I'm sorry, but that is not possible.""",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.DoesNotContain("NativeTool", RespondingText(entries));
        Assert.Contains(" Done.", RespondingText(entries));
    }

    /// <summary>
    /// The 0.5B model often closes the call object then trails one or more
    /// stray '}' closers (user-observed: "there was a closing bracket"). Those
    /// must be swallowed with the drop, so the dialogue starts clean.
    /// </summary>
    [Fact]
    public async Task QwenMetadata_StrayClosersAfterCall_DontLeak()
    {
        string? seenName = null;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}} Let me check.""",
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
        Assert.DoesNotContain("}", prose);
        Assert.Contains("Let me check.", prose);
        Assert.Contains(" Done.", prose);
    }

    /// <summary>
    /// User-observed: the second tool call came out malformed/unclosed
    /// (<c>{"name":"UIGetFreeMemory","arguments":{"}}</c>). It must not dispatch
    /// (it is not a call), must not leak, and must not be persisted to history.
    /// </summary>
    [Fact]
    public async Task QwenMetadata_MalformedCall_IsSilent()
    {
        int dispatchCalls = 0;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"}}""");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            dispatchCalls++;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal(0, dispatchCalls);
        Assert.DoesNotContain("NativeTool", RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    /// <summary>
    /// A valid call with an empty arguments object (no-arg tools) must still
    /// dispatch and be hidden from the transcript.
    /// </summary>
    [Fact]
    public async Task QwenMetadata_EmptyArgumentsObject_Dispatches()
    {
        string? seenName = null;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{}}""",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.Contains(" Done.", RespondingText(entries));
    }

    /// <summary>
    /// With no metadata the format resolves to None: even a perfectly formed
    /// tagged call must not dispatch and its markup must not be shown.
    /// </summary>
    [Fact]
    public async Task NoMetadata_DisablesToolCapture()
    {
        int dispatchCalls = 0;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCallFormat.None,
            """<tool_call>{"tool":"NativeTool","arguments":{"x":"1"}}</tool_call>""");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            dispatchCalls++;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal(0, dispatchCalls);
        Assert.DoesNotContain("tool_call", RespondingText(entries));
    }
}