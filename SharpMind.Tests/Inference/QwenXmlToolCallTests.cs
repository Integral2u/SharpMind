using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Qwen3.5 renders tools as XML &lt;function=...&gt;&lt;parameter=...&gt; blocks.
/// These tests cover detection from the template, the XML parser, schema-driven
/// argument types, and the agent prompt that teaches the XML shape.
/// </summary>
public sealed class QwenXmlToolCallTests
{
    private const string Qwen35Template =
        "{%- if tools %}<tool_call>\n<function=example_function_name>\n<parameter=x>\n</parameter>\n</function>\n</tool_call>{%- endif %}";

    private static ModelMetaData Meta(params (string Key, string Value)[] kvs) => new()
    {
        KvPairs = [.. kvs.Select(k => new KvPair { Key = k.Key, Value = k.Value })]
    };

    private static IAgentBuilder TypedBuilder(List<JsonObject>? captured = null)
    {
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["count"] = new JsonObject { ["type"] = "integer" },
                ["enabled"] = new JsonObject { ["type"] = "boolean" },
                ["label"] = new JsonObject { ["type"] = "string" },
                ["tags"] = new JsonObject { ["type"] = "array" },
            },
            ["required"] = new JsonArray()
        };
        return new AgentBuilder("Test").WithTool(
            "Measure", "Measures things", schema,
            args => Task.FromResult("ok"));
    }

    private static string CallWith(string body) =>
        "<tool_call>\n<function=Measure>\n" + body + "</function>\n</tool_call>";

    // ── Detection ────────────────────────────────────────────────────

    [Fact]
    public void Resolve_FunctionTagTemplate_IsQwenXml()
    {
        var meta = Meta(("tokenizer.chat_template", Qwen35Template), ("general.architecture", "qwen35"));
        Assert.Equal(ToolCallFormat.QwenXml, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_JsonToolCallTemplate_StaysQwen()
    {
        // Qwen2.5 wraps JSON in <tool_call>; it must not be treated as XML.
        var meta = Meta(
            ("tokenizer.chat_template", "{%- if tools %}<tool_call>\n{\"name\": \"f\"}\n</tool_call>{%- endif %}"),
            ("general.architecture", "qwen2"));
        Assert.Equal(ToolCallFormat.Qwen, ToolCallFormatDetector.Resolve(meta));
    }

    [Fact]
    public void Resolve_QwenNameWithoutTemplate_StaysQwen()
    {
        var meta = Meta(("general.name", "Qwen2-0.5B-Instruct"));
        Assert.Equal(ToolCallFormat.Qwen, ToolCallFormatDetector.Resolve(meta));
    }

    // ── Parsing and dispatch ─────────────────────────────────────────

    [Fact]
    public async Task XmlCall_DispatchesWithSchemaTypedArguments()
    {
        string call = CallWith(
            "<parameter=count>\n3\n</parameter>\n" +
            "<parameter=enabled>\ntrue\n</parameter>\n" +
            "<parameter=label>\nred\n</parameter>\n" +
            "<parameter=tags>\n[\"a\",\"b\"]\n</parameter>\n");
        await using var session = ScriptedSession.Create(TypedBuilder(), ToolCallFormat.QwenXml, call, " Done.");

        var dispatched = new List<JsonObject>();
        session.ProcessToolRequest = (name, args, ct) =>
        {
            dispatched.Add(new JsonObject { ["tool"] = name, ["arguments"] = JsonNode.Parse(args.ToJsonString()) });
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("measure it")) entries.Add(e);

        Assert.Single(dispatched);
        var args = dispatched[0]["arguments"]!.AsObject();
        Assert.Equal(3L, args["count"]!.GetValue<long>());
        Assert.True(args["enabled"]!.GetValue<bool>());
        Assert.Equal("red", args["label"]!.GetValue<string>());
        Assert.Equal(2, args["tags"]!.AsArray().Count);
    }

    [Fact]
    public async Task XmlCall_ToolResultIsFedBackAsToolRole()
    {
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            CallWith("<parameter=label>\nred\n</parameter>\n"), " Done.");
        session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.Handled("external result"));

        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }

        var toolMsg = session.History.FirstOrDefault(m => m.Role == ChatRole.Tool);
        Assert.NotNull(toolMsg);
        Assert.Contains("external result", toolMsg!.Content);
    }

    [Fact]
    public async Task XmlCall_WithProseBefore_ShowsProseAndNoMarkup()
    {
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            "Sure. " + CallWith("<parameter=label>\nred\n</parameter>\n"), " Done.");
        session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.Handled("ok"));

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        string prose = string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));
        Assert.Contains("Sure", prose);
        Assert.DoesNotContain("<", prose);
        Assert.DoesNotContain("parameter", prose);
    }

    [Fact]
    public async Task XmlCall_TruncatedBeforeClosingTags_StillDispatches()
    {
        // Model stopped before </function></tool_call>; the parser repairs it.
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            "<tool_call>\n<function=Measure>\n<parameter=label>\nred\n</parameter>\n", " Done.");

        int calls = 0;
        session.ProcessToolRequest = (_, args, _) =>
        {
            calls++;
            Assert.Equal("red", args["label"]!.GetValue<string>());
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };

        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }
        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task XmlCall_UnregisteredToolName_IsSurfacedToHostWithItsName()
    {
        // Same contract as the JSON formats: the host seam sees every parsed call
        // and decides; the name is passed through unchanged, not rewritten.
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            "<tool_call>\n<function=NotATool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
            " Done.");

        var names = new List<string>();
        session.ProcessToolRequest = (name, _, _) =>
        {
            names.Add(name);
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };

        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }
        Assert.Equal(["NotATool"], names);
    }

    [Fact]
    public async Task XmlCall_StringArgumentThatLooksNumeric_StaysString_WhenSchemaSaysString()
    {
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            CallWith("<parameter=label>\n42\n</parameter>\n"), " Done.");

        JsonObject? seen = null;
        session.ProcessToolRequest = (_, args, _) => { seen = args; return Task.FromResult(ToolRequestResult.Handled("ok")); };

        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }
        Assert.NotNull(seen);
        Assert.Equal("42", seen!["label"]!.GetValue<string>());
    }

    [Fact]
    public async Task XmlCall_WithoutWrapper_IsHiddenAndDispatches()
    {
        // The model sometimes drops <tool_call>; the bare block must not leak.
        await using var session = ScriptedSession.Create(
            TypedBuilder(), ToolCallFormat.QwenXml,
            "<function=Measure>\n<parameter=label>\nred\n</parameter>\n</function>",
            " Done.");

        string? label = null;
        session.ProcessToolRequest = (_, args, _) =>
        {
            label = args["label"]!.GetValue<string>();
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal("red", label);
        string prose = string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));
        Assert.DoesNotContain("function", prose);
        Assert.DoesNotContain("parameter", prose);
    }

    // ── Stream filter ────────────────────────────────────────────────

    [Fact]
    public void Filter_BareXmlBlockAndWrapperCloser_AreHidden()
    {
        var filter = new StreamingToolMarkupFilter();
        var output = new System.Text.StringBuilder();
        output.Append(filter.Append("Sure "));
        output.Append(filter.Append("<function=Measure>\n<parameter=label>\nred\n</parameter>\n</function>"));
        output.Append(filter.Append("\n</tool_call>"));
        output.Append(filter.Append(" done"));
        output.Append(filter.Flush());

        Assert.Equal("Sure  done", output.ToString());
    }

    [Fact]
    public void Filter_WrapperCloserSplitAcrossFragments_IsStillHidden()
    {
        var filter = new StreamingToolMarkupFilter();
        var output = new System.Text.StringBuilder();
        output.Append(filter.Append("<function=Measure><parameter=label>red</parameter></function>"));
        output.Append(filter.Append("\n"));
        output.Append(filter.Append("</tool_call>"));
        output.Append(filter.Append("ok"));
        output.Append(filter.Flush());

        Assert.Equal("ok", output.ToString());
    }

    // ── Prompt ───────────────────────────────────────────────────────

    [Fact]
    public void QwenXmlFormat_PromptTeachesOnlyTheXmlContract()
    {
        var agent = TypedBuilder();
        agent.CallFormat = ToolCallFormat.QwenXml;
        string prompt = agent.BuildAgentPrompt();

        // The default branch teaches SharpMind's JSON-inside-tags contract; it
        // must not reach QwenXml, or the rules contradict the XML example.
        Assert.DoesNotContain("JSON object inside", prompt);
        Assert.DoesNotContain("{\"tool\":", prompt);
        Assert.Contains("at most one tool call", prompt);
        Assert.Contains("never write tool names, descriptions or schemas", prompt);
    }

    [Fact]
    public void QwenJsonFormat_KeepsAvailableToolsList()
    {
        var agent = TypedBuilder();
        agent.CallFormat = ToolCallFormat.Qwen;
        Assert.Contains("## Available Tools", agent.BuildAgentPrompt());
    }

    [Fact]
    public void QwenJsonFormat_PromptIsUnchanged_ByQwenXmlBranch()
    {
        var agent = TypedBuilder();
        agent.CallFormat = ToolCallFormat.Qwen;
        string prompt = agent.BuildAgentPrompt();

        Assert.DoesNotContain("<function=", prompt);
        Assert.DoesNotContain("at most one tool call", prompt);
        Assert.Contains("reply with ONLY the JSON object", prompt);
    }

    [Fact]
    public void QwenXmlFormat_PromptTeachesXmlCallBlock()
    {
        var agent = TypedBuilder();
        agent.CallFormat = ToolCallFormat.QwenXml;   // the session sets this from metadata
        var prompt = agent.BuildAgentPrompt();
        Assert.Contains("<tool_call>", prompt);
        Assert.Contains("<function=Measure>", prompt);
        Assert.Contains("<parameter=count>", prompt);
    }
}
