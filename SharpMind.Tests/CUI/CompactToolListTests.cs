using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Regression cover for <c>AgentBuilder.BuildCompactToolList</c>.
///
/// The compaction pass only runs when the tool JSON exceeds
/// <c>CompactToolBudget</c> (4000 chars) — under it, <c>BuildAgentPrompt</c>
/// returns the verbose dump and never touches the reducing code at all. That
/// threshold is why <see cref="AgentPromptSizeTests"/> passed while the real CUI
/// crashed on every launch: its three-tool set sits under the budget, the CUI's
/// live set does not.
///
/// The specific defect (upstream 3519e0e, fixed here): the reducer copied
/// <c>tool["description"]</c> into a new JsonObject by assignment rather than by
/// <c>DeepClone</c>, and System.Text.Json refuses to re-parent a node that
/// already belongs to a tree — "The node already has a parent". It surfaced as
/// an InvalidOperationException out of ChatSession.InitializeChat, after the
/// weights had finished loading.
/// </summary>
public sealed class CompactToolListTests
{
    /// <summary>
    /// Registers enough tools to push the JSON past the compaction budget, but
    /// few enough that the reduced form lands back under it — so the first pass
    /// runs and <em>keeps</em> descriptions, which is the branch that carried the
    /// defect. Too many tools and the second pass strips descriptions instead,
    /// hiding the very code under test.
    /// </summary>
    private static AgentBuilder BuilderOverBudget(string description, int toolCount = 22)
    {
        var builder = new AgentBuilder("Compact") { DisabledTools = [] };
        for (int i = 0; i < toolCount; i++)
        {
            var schema = new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["path"] = new JsonObject { ["type"] = "string", ["description"] = "a path argument" },
                    ["count"] = new JsonObject { ["type"] = "integer", ["description"] = "how many" },
                },
                ["required"] = new JsonArray("path"),
            };
            builder.WithTool($"tool_{i}", description, schema, _ => Task.FromResult("ok"));
        }
        return builder;
    }

    [Fact]
    public void BuildAgentPrompt_OverBudget_WithShortDescriptions_DoesNotThrow()
    {
        // <= 140 chars: the reducer keeps the description, which is the branch
        // that re-parented a live node before the fix.
        var builder = BuilderOverBudget("Reads a thing.");

        Assert.True(builder.ToolDefinitions.ToJsonString().Length > 4000,
            "fixture must exceed CompactToolBudget or the compaction path never runs");

        string prompt = builder.BuildAgentPrompt();

        Assert.Contains("tool_0", prompt);
        Assert.Contains("Reads a thing.", prompt);
    }

    [Fact]
    public void BuildAgentPrompt_OverBudget_WithLongDescriptions_DoesNotThrow()
    {
        // > 140 chars: the reducer drops the description and reports the count.
        var builder = BuilderOverBudget(new string('x', 200), toolCount: 12);

        string prompt = builder.BuildAgentPrompt();

        Assert.Contains("tool_0", prompt);
        Assert.DoesNotContain(new string('x', 200), prompt);
    }

    [Fact]
    public void BuildAgentPrompt_IsRepeatable()
    {
        // The reducer must not mutate ToolDefinitions: the CUI builds the prompt
        // again on every rebuild, and a destructive first pass would only show up
        // on the second call.
        var builder = BuilderOverBudget("Reads a thing.");

        string first = builder.BuildAgentPrompt();
        string second = builder.BuildAgentPrompt();

        Assert.Equal(first, second);
    }
}
