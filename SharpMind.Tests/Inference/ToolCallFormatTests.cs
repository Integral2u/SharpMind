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

    [Fact]
    public void Resolve_TemplateWithToolCallsMarker_IsMistral()
    {
        var meta = Meta(("tokenizer.chat_template",
            "[INST] {{messages[0]['content'] }} [/INST] [AVAILABLE_TOOLS]{{tools}}[/AVAILABLE_TOOLS][TOOL_CALLS]{{tool_calls}}[/TOOL_CALLS]"));
        Assert.Equal(ToolCallFormat.Mistral, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_MistralName_IsMistral()
    {
        var meta = Meta(("general.name", "Ministral-3-3B-Instruct-2512"));
        Assert.Equal(ToolCallFormat.Mistral, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_TemplateWithPythonTag_IsLlama3()
    {
        var meta = Meta(("tokenizer.chat_template",
            "<|start_header_id|>user<|end_header_id|>\n\n{{u}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|python_tag|>{{call}}"));
        Assert.Equal(ToolCallFormat.Llama3, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_LlamaName_IsLlama3()
    {
        var meta = Meta(("general.name", "Llama-3.2-1B-Instruct"));
        Assert.Equal(ToolCallFormat.Llama3, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_TemplateWithFunctionCall_IsGemma()
    {
        var meta = Meta(("tokenizer.chat_template",
            "<start_of_turn>user\n{{u}}<end_of_turn>\n<start_of_turn>model\n{{function_call}}"));
        Assert.Equal(ToolCallFormat.Gemma, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_GemmaName_IsGemma()
    {
        var meta = Meta(("general.name", "functiongemma-270m-it"));
        Assert.Equal(ToolCallFormat.Gemma, ToolCallFormatDetector.Resolve(meta));
    }

    // ── Prompt shape ────────────────────────────────────────────────────────

    [Fact]
    public void SharpMindFormatDefault_StillTeachesTaggedCalls()
    {
        string prompt = BuilderWithNativeTool().BuildAgentPrompt();
        Assert.Contains("tool_call", prompt);
        Assert.Contains("\"tool\":", prompt);
        Assert.Contains("Narrate", prompt);
    }

    [Fact]
    public void QwenFormat_PromptTeachesRawJsonOnly()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("{\"name\":", prompt);
        Assert.DoesNotContain("tool_call", prompt);
        Assert.DoesNotContain("Narrate", prompt);
    }

    [Fact]
    public void QwenFormat_PromptTreatsToolRequestsAsOrders()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("can you", prompt);
        Assert.Contains("as orders, not questions", prompt);
    }

    [Fact]
    public void QwenFormat_ExampleUsesRealRegisteredToolName()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Qwen;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("\"NativeTool\"", prompt);
        Assert.Contains("\"x\":\"...\"", prompt);
        Assert.Contains("ONLY the JSON object", prompt);
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

    [Fact]
    public void MistralFormat_PromptTeachesToolCallsMarker()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Mistral;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("[TOOL_CALLS]", prompt);
        Assert.Contains("{\"name\":", prompt);
        Assert.Contains("NativeTool", prompt);
    }

    [Fact]
    public void Llama3Format_PromptTeachesRawJson()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Llama3;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("{\"name\":", prompt);
        Assert.Contains("NativeTool", prompt);
    }

    [Fact]
    public void GemmaFormat_PromptTeachesArgsKey()
    {
        var builder = BuilderWithNativeTool();
        builder.CallFormat = ToolCallFormat.Gemma;
        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("\"args\"", prompt);
        Assert.Contains("NativeTool", prompt);
        Assert.Contains("not \"arguments\"", prompt);
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
    /// stray '}' closers (user-observed: "there was a closing bracket") plus
    /// pre-result narration. Since the call names a registered tool, the whole
    /// turn is suppressed (the tool loop regenerates after the result) — nothing
    /// leaks, and the dialogue continues cleanly in the next turn.
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
        // Pre-result narration is suppressed along with the call object; the
        // post-result turn is the only text the user sees.
        Assert.DoesNotContain("Let me check.", prose);
        Assert.Contains(" Done.", prose);
    }

    /// <summary>
    /// The 0.5B UX regression: the model calls UIShowOptionSelection and then
    /// ALSO narrates "I recommend: Red, Green, Blue" before the tool result
    /// arrives, so the options appear twice (once in the dialog, once as text).
    /// A registered native-format call suppresses the whole pre-result tail.
    /// </summary>
    [Fact]
    public async Task QwenMetadata_CallFollowedByRedundantNarration_SuppressesTail()
    {
        string? seenName = null;
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}Based on your question, I recommend:\n\n  * Red\n  * Green\n  * Blue""",
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
        Assert.DoesNotContain("Red", prose);
        Assert.DoesNotContain("Green", prose);
        Assert.DoesNotContain("Blue", prose);
        Assert.DoesNotContain("recommend", prose);
        Assert.Contains(" Done.", prose);
    }

    /// <summary>
    /// A call object naming an UNREGISTERED tool still hides the object, but
    /// keeps any surrounding narration visible (the call will not dispatch, so
    /// the model's prose is all the user has).
    /// </summary>
    [Fact]
    public async Task QwenMetadata_CallObjectForUnknownTool_KeepsNarration()
    {
        var qwen = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            qwen,
            BuilderWithNativeTool(),
            """{"name":"NotATool","arguments":{"x":"9"}}} Hold on.""",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
            Task.FromResult(ToolRequestResult.Handled("external result"));

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        string prose = RespondingText(entries);
        Assert.DoesNotContain("NotATool", prose);
        Assert.Contains("Hold on.", prose);
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

    // ── New formats: Mistral ────────────────────────────────────────────────

    [Fact]
    public async Task MistralMetadata_ResultFedBackAsToolRole()
    {
        var mistral = Meta(("general.name", "Ministral-3-3B-Instruct-2512"));
        await using var session = ScriptedSession.CreateForMeta(
            mistral,
            BuilderWithNativeTool(),
            """[TOOL_CALLS]{"name":"NativeTool","arguments":{"x":"9"}}""",
            " Received.");
        session.ProcessToolRequest = (toolName, args, ct) =>
            Task.FromResult(ToolRequestResult.Handled("external result"));

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        var toolMsg = session.History.FirstOrDefault(m => m.Role == ChatRole.Tool);
        Assert.NotNull(toolMsg);
        Assert.Contains("external result", toolMsg!.Content);
    }

    [Fact]
    public async Task MistralMetadata_ToolCallsMarker_Dispatches()
    {
        string? seenName = null;
        var mistral = Meta(("general.name", "Ministral-3-3B-Instruct-2512"));
        await using var session = ScriptedSession.CreateForMeta(
            mistral,
            BuilderWithNativeTool(),
            """[TOOL_CALLS]{"name":"NativeTool","arguments":{"x":"9"}}""",
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
        Assert.DoesNotContain("[TOOL_CALLS]", RespondingText(entries));
    }

    [Fact]
    public async Task MistralMetadata_RawJsonFallback_Dispatches()
    {
        string? seenName = null;
        var mistral = Meta(("general.name", "Ministral-3-3B-Instruct-2512"));
        await using var session = ScriptedSession.CreateForMeta(
            mistral,
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

    // ── New formats: Llama3 ─────────────────────────────────────────────────

    [Fact]
    public async Task Llama3Metadata_PythonTagCall_Dispatches()
    {
        string? seenName = null;
        var llama = Meta(("general.name", "Llama-3.2-1B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            llama,
            BuilderWithNativeTool(),
            """<|python_tag|>{"name":"NativeTool","arguments":{"x":"9"}}""",
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
        // The tool-call JSON payload must never leak into the transcript as
        // prose. (The harness splits script fragments on '|', so the marker's
        // own sub-fragments may briefly surface as '<'/'>' before the complete
        // marker is in the buffer — the payload is the thing that matters.)
        Assert.DoesNotContain("NativeTool", RespondingText(entries));
    }

    [Fact]
    public async Task Llama3Metadata_ResultFedBackAsSystemRole()
    {
        var llama = Meta(("general.name", "Llama-3.2-1B-Instruct"));
        await using var session = ScriptedSession.CreateForMeta(
            llama,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","arguments":{"x":"9"}}""",
            " Received.");
        session.ProcessToolRequest = (toolName, args, ct) =>
            Task.FromResult(ToolRequestResult.Handled("external result"));

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.DoesNotContain(session.History, m => m.Role == ChatRole.Tool);
    }

    // ── New formats: Gemma ──────────────────────────────────────────────────

    [Fact]
    public async Task GemmaMetadata_ArgsKeyCall_Dispatches()
    {
        string? seenName = null;
        JsonObject? seenArgs = null;
        var gemma = Meta(("general.name", "functiongemma-270m-it"));
        await using var session = ScriptedSession.CreateForMeta(
            gemma,
            BuilderWithNativeTool(),
            """{"name":"NativeTool","args":{"x":"9"}}""",
            " Done.");
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            seenName = toolName;
            seenArgs = args;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("NativeTool", seenName);
        Assert.NotNull(seenArgs);
        Assert.Equal("9", seenArgs!["x"]?.GetValue<string>());
        Assert.Contains(" Done.", RespondingText(entries));
    }

    [Fact]
    public async Task GemmaMetadata_FunctionCallWrapper_Dispatches()
    {
        string? seenName = null;
        var gemma = Meta(("general.name", "functiongemma-270m-it"));
        await using var session = ScriptedSession.CreateForMeta(
            gemma,
            BuilderWithNativeTool(),
            """<function_call>{"name":"NativeTool","args":{"x":"9"}}</function_call>""",
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
        Assert.DoesNotContain("function_call", RespondingText(entries));
    }
}