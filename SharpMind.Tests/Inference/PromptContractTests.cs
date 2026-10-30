using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Core.AgentTools;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The agent prompt is what a small model copies from. These tests pin its shape:
/// one call format, one filled-in example, one tool list, and a tool list that
/// stays valid JSON when the budget forces it shorter.
/// </summary>
public sealed class PromptContractTests
{
    private static readonly ToolCallFormat[] AllFormats =
    [
        ToolCallFormat.SharpMind, ToolCallFormat.Qwen, ToolCallFormat.QwenXml,
        ToolCallFormat.Mistral, ToolCallFormat.Llama3, ToolCallFormat.Gemma
    ];

    private const string ToolDescription = "Measures things for the user.";

    private static JsonObject Schema(params (string Name, string Type, bool Required)[] args)
    {
        var props = new JsonObject();
        var required = new JsonArray();
        foreach (var (name, type, req) in args)
        {
            props[name] = new JsonObject { ["type"] = type, ["description"] = $"The {name} value." };
            if (req) required.Add(name);
        }
        return new JsonObject { ["type"] = "object", ["properties"] = props, ["required"] = required };
    }

    private static AgentBuilder Builder(ToolCallFormat format, int? budget = null)
    {
        var agent = new AgentBuilder("Test");
        agent.WithTool("Measure", ToolDescription, Schema(("label", "string", true), ("count", "integer", false)),
            args => Task.FromResult("ok"));
        agent.CallFormat = format;
        if (budget is int b) agent.ToolListTokenBudget = b;
        return agent;
    }

    /// <summary>The text under "## Available Tools" up to the end of the prompt.</summary>
    private static string ToolSection(string prompt)
    {
        string header = "## Available Tools" + Environment.NewLine;
        int at = prompt.IndexOf(header, StringComparison.Ordinal);
        Assert.True(at >= 0, "prompt has no ## Available Tools section");
        return prompt[(at + header.Length)..].Trim();
    }

    /// <summary>The JSON part of the tool section: every line before the first "(" note.</summary>
    private static JsonNode? ToolListJson(string section)
    {
        var lines = section.Split('\n');
        int end = Array.FindIndex(lines, l => l.StartsWith("(", StringComparison.Ordinal));
        string json = string.Join("\n", end < 0 ? lines : lines[..end]).Trim();
        return JsonNode.Parse(json);
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryFormat_HasOneExampleCallAndOneToolList(ToolCallFormat format)
    {
        string prompt = Builder(format).BuildAgentPrompt();

        Assert.Equal(1, Count(prompt, "Example with no tool needed:"));
        Assert.Equal(1, Count(prompt, "## Available Tools" + Environment.NewLine));
        // Count the section headings, not mentions: a rule may refer to a section by name.
        Assert.Equal(1, prompt.Split('\n').Count(l => l.StartsWith("## Tool Call Format", StringComparison.Ordinal)));
        // The description is listed once, in the tool list, and the example call never repeats it.
        string list = prompt.Split("## Available Tools" + Environment.NewLine)[1];
        string example = prompt.Split("## Tool Call Format")[1].Split("## Available Tools")[0];
        Assert.Equal(1, Count(list, ToolDescription));
        Assert.DoesNotContain(ToolDescription, example);
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryFormat_ExampleUsesRealValues_NotPlaceholders(ToolCallFormat format)
    {
        // The coaching rules keep their own "..." notation. The example call, which the model
        // copies, must be filled in with real values.
        string prompt = Builder(format).BuildAgentPrompt();
        string example = prompt.Split("Example with no tool needed:")[0].Split("Example:")[^1];
        Assert.DoesNotContain("\"...\"", example);
        Assert.DoesNotContain("<name>", prompt);
        Assert.Contains("label", example);
        Assert.Contains("example", example);
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void EveryFormat_PromptHasNoSampleToolResult(ToolCallFormat format)
    {
        // A sample result is copied by small models as if the tool had run. The tool-free
        // example's "Reply:" line is a plain prose example and is not a result.
        string prompt = Builder(format).BuildAgentPrompt();
        Assert.DoesNotContain("[Measure result]", prompt);
        Assert.DoesNotContain("[Measure label=", prompt);
        Assert.Contains("Example with no tool needed:", prompt);
    }

    [Fact]
    public void SharpMindExample_UsesNameKey()
    {
        string prompt = Builder(ToolCallFormat.SharpMind).BuildAgentPrompt();
        Assert.Contains("<tool_call>{\"name\":\"Measure\",\"arguments\":", prompt);
    }

    [Theory]
    [InlineData(ToolCallFormat.Gemma, "\"args\":{")]
    [InlineData(ToolCallFormat.Qwen, "\"arguments\":{")]
    [InlineData(ToolCallFormat.Llama3, "\"arguments\":{")]
    [InlineData(ToolCallFormat.Mistral, "[TOOL_CALLS]{\"name\":\"Measure\",\"arguments\":")]
    public void ArgumentKey_MatchesFormat(ToolCallFormat format, string expected)
    {
        Assert.Contains(expected, Builder(format).BuildAgentPrompt());
    }

    [Fact]
    public void Example_PrefersAToolWithRequiredArguments_SkippingEcho()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new EchoTool());
        agent.WithTool("Measure", ToolDescription, Schema(("label", "string", true)), args => Task.FromResult("ok"));
        agent.CallFormat = ToolCallFormat.Qwen;

        string prompt = agent.BuildAgentPrompt();
        Assert.Contains("\"name\":\"Measure\"", prompt.Split("Example:")[1]);
        Assert.DoesNotContain("Example:\n{\"name\":\"Echo", prompt);
    }

    [Fact]
    public void ToolsRenderedByTemplate_ListNamesOnly()
    {
        var agent = Builder(ToolCallFormat.QwenXml);
        agent.ToolListInPrompt = false;
        string prompt = agent.BuildAgentPrompt();

        Assert.Contains("Names: Measure. Their full definitions are provided separately.", prompt);
        Assert.DoesNotContain(ToolDescription, prompt);
        Assert.DoesNotContain("\"properties\"", prompt);
    }

    [Fact]
    public void ToolList_UnderBudget_IsTheFullDefinitionList()
    {
        string section = ToolSection(Builder(ToolCallFormat.Qwen).BuildAgentPrompt());
        var list = ToolListJson(section) as JsonArray;
        Assert.NotNull(list);
        Assert.Equal(ToolDescription, list![0]!["description"]!.GetValue<string>());
    }

    [Fact]
    public void ToolList_OverBudget_StaysValidJson_AndReportsShortening()
    {
        // A tight budget forces the tiered compaction; the list must still parse.
        string prompt = Builder(ToolCallFormat.Qwen, budget: 60).BuildAgentPrompt();
        string section = ToolSection(prompt);

        var list = ToolListJson(section) as JsonArray;
        Assert.NotNull(list);
        Assert.Equal("Measure", list![0]!["name"]!.GetValue<string>());
        Assert.Contains("(Descriptions were shortened", section);
    }

    [Fact]
    public void ToolList_OverBudget_KeepsRequiredArgumentsAndTheirTypes()
    {
        var section = ToolSection(Builder(ToolCallFormat.Qwen, budget: 60).BuildAgentPrompt());
        var tool = (ToolListJson(section) as JsonArray)![0]!.AsObject();

        // A required argument keeps its type (as a bare string or, at the richest tier,
        // an object with a description). Either way the type is "string".
        var label = tool["arguments"]!["label"]!;
        string type = label is JsonObject o ? o["type"]!.GetValue<string>() : label.GetValue<string>();
        Assert.Equal("string", type);
        Assert.Equal("label", tool["required"]![0]!.GetValue<string>());
    }

    [Fact]
    public void ToolList_FarOverBudget_FallsBackToNamesOutsideJson()
    {
        var agent = new AgentBuilder("Test");
        for (int i = 0; i < 40; i++)
            agent.WithTool($"Tool{i:D2}", ToolDescription, Schema(("label", "string", true)), _ => Task.FromResult("ok"));
        agent.CallFormat = ToolCallFormat.Qwen;
        agent.ToolListTokenBudget = 40;

        string section = ToolSection(agent.BuildAgentPrompt());
        var names = ToolListJson(section) as JsonArray;
        Assert.NotNull(names);
        Assert.Equal("Tool00", names![0]!.GetValue<string>());
        Assert.Contains("more tool(s) not listed", section);
    }

    [Fact]
    public void Rules_AreStatedOnceAndKeepTheCoachingThatWorks()
    {
        // The per-format rules are the coaching the models were tuned on; these assertions pin
        // the parts that matter for a call: wait for the result, the order rule, and the
        // prose fallback that does not reference tools.
        string prompt = Builder(ToolCallFormat.Mistral).BuildAgentPrompt();

        Assert.Equal(1, Count(prompt, "\"can you...\""));
        Assert.Contains("wait for its result, then answer directly from it", prompt);
        Assert.Contains("as orders, not questions", prompt);
        Assert.Contains("behave exactly as if tools did not exist", prompt);
    }

    [Fact]
    public void NoTools_OmitsToolSections()
    {
        var agent = new AgentBuilder("Test");
        agent.CallFormat = ToolCallFormat.Qwen;
        string prompt = agent.BuildAgentPrompt();
        Assert.DoesNotContain("## Tool Call Format", prompt);
        Assert.DoesNotContain("## Available Tools", prompt);
    }

    public static IEnumerable<object[]> Formats() => AllFormats.Select(f => new object[] { f });

    private static int Count(string text, string needle)
    {
        int n = 0, at = 0;
        while ((at = text.IndexOf(needle, at, StringComparison.Ordinal)) >= 0) { n++; at += needle.Length; }
        return n;
    }
}
