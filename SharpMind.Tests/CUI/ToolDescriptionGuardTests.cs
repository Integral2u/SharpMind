using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using SharpMind.Core.AgentTools;
using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.CUI;

/// <summary>
/// A small model copies the text it is shown. These tests keep the CUI's built-in
/// tools from carrying an example question in an argument description, and keep the
/// example call for the option dialog built from the choices it will present.
/// </summary>
public sealed class ToolDescriptionGuardTests
{
    private static readonly Regex QuotedQuestionExample = new(@"e\.g\.\s*'[^']*\?'", RegexOptions.Compiled);

    private static AgentBuilder CuiBuilder()
    {
        var builder = new AgentBuilder("Guard") { DisabledTools = [] };
        builder.WithTools(new CuiTools(new CuiToolContext()));
        builder.WithTools(new WeatherTool());
        builder.WithTools(new FileSystemTool(Path.GetTempPath()));
        return builder;
    }

    [Fact]
    public void BuiltInToolArguments_DoNotCarryCopyableQuestionExamples()
    {
        var builder = CuiBuilder();

        foreach (var tool in builder.ToolDefinitions.OfType<JsonObject>())
        {
            if (tool["parameters"]?["properties"] is not JsonObject props) continue;
            foreach (var (name, schema) in props)
            {
                string desc = schema?["description"]?.GetValue<string>() ?? "";
                Assert.False(QuotedQuestionExample.IsMatch(desc),
                    $"{tool["name"]}.{name} carries a copyable question example: {desc}");
            }
        }
    }

    [Fact]
    public void OptionDialog_ExampleCall_UsesTheOptionsItPresents()
    {
        string prompt = CuiBuilder().BuildAgentPrompt();

        Assert.Contains("\"prompt\":\"Red, green or blue?\"", prompt);
        Assert.DoesNotContain("Which file format", prompt);
    }
}
