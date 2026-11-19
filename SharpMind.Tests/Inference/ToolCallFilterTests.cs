using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.Inference;

public sealed class ToolCallFilterTests
{
    private sealed class RefusingFilter : IToolCallFilter
    {
        public int Calls { get; private set; }

        public Task<string?> ReviewAsync(
            string tool,
            JsonObject arguments,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult<string?>("blocked by test");
        }
    }

    [Fact]
    public async Task Filter_BlocksBeforeTheToolRuns()
    {
        int invoked = 0;
        var builder = new AgentBuilder("FilterTest");
        builder.WithTool(
            "echo",
            "Echoes text.",
            new JsonObject { ["type"] = "object" },
            _ =>
            {
                invoked++;
                return Task.FromResult("ok");
            });

        var filter = new RefusingFilter();
        builder.ToolCallFilter = filter;

        JsonObject result = await builder.CallToolAsync(new JsonObject
        {
            ["tool"] = "echo",
            ["arguments"] = new JsonObject { ["text"] = "hi" },
        });

        Assert.Equal(1, filter.Calls);
        Assert.Equal(0, invoked);
        Assert.Equal("blocked by test", result["message"]?.GetValue<string>());
        Assert.Equal("blocked_by_filter", result["code"]?.GetValue<string>());
    }

    [Fact]
    public async Task NoFilter_RunsTheTool()
    {
        int invoked = 0;
        var builder = new AgentBuilder("FilterTest");
        builder.WithTool(
            "echo",
            "Echoes text.",
            new JsonObject { ["type"] = "object" },
            _ =>
            {
                invoked++;
                return Task.FromResult("ok");
            });

        JsonObject result = await builder.CallToolAsync(new JsonObject
        {
            ["tool"] = "echo",
            ["arguments"] = new JsonObject(),
        });

        Assert.Equal(1, invoked);
        Assert.Equal("success", result["status"]?.GetValue<string>());
    }

    [Fact]
    public async Task Filter_AllowingTheCall_RunsTheTool()
    {
        int invoked = 0;
        var builder = new AgentBuilder("FilterTest");
        builder.WithTool(
            "echo",
            "Echoes text.",
            new JsonObject { ["type"] = "object" },
            _ =>
            {
                invoked++;
                return Task.FromResult("ok");
            });

        builder.ToolCallFilter = new AllowFilter();

        JsonObject result = await builder.CallToolAsync(new JsonObject
        {
            ["tool"] = "echo",
            ["arguments"] = new JsonObject(),
        });

        Assert.Equal(1, invoked);
        Assert.Equal("success", result["status"]?.GetValue<string>());
    }

    private sealed class AllowFilter : IToolCallFilter
    {
        public Task<string?> ReviewAsync(
            string tool,
            JsonObject arguments,
            CancellationToken cancellationToken = default) =>
            Task.FromResult<string?>(null);
    }
}
