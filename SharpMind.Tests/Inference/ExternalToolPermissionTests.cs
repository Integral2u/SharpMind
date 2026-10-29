using System.Text.Json.Nodes;
using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>A tool whose code lives outside the built-in assemblies, like a plugin.</summary>
public sealed class ExternalProbeTool
{
    public static int Calls;

    [ToolDesc("Probe tool that records that it ran.")]
    public string Probe([ToolDesc("A label.")] string label)
    {
        Calls++;
        return "probed " + label;
    }
}

/// <summary>
/// External (plugin) tools can do IO outside the interceptors, so the session
/// must ask permission before running them. Built-in tools are not gated here.
/// </summary>
public sealed class ExternalToolPermissionTests
{
    private const string CallProbe =
        "<tool_call>{\"tool\":\"Probe\",\"arguments\":{\"label\":\"x\"}}</tool_call>";

    private static IAgentBuilder ProbeAgent() => new AgentBuilder("Test").WithTools(new ExternalProbeTool());

    [Fact]
    public void IsExternalTool_ClassifiesByDeclaringAssembly()
    {
        var agent = new AgentBuilder("Test").WithTools(new ExternalProbeTool());
        Assert.True(agent.IsExternalTool("Probe"));

        var builtIn = new AgentBuilder("Test");
        builtIn.WithTools(new EchoTool());
        Assert.False(builtIn.IsExternalTool("Echo"));
    }

    [Fact]
    public async Task ExternalTool_IsDeniedWhenPermissionIsNever()
    {
        int before = ExternalProbeTool.Calls;
        await using var session = ScriptedSession.CreateGated(
            ProbeAgent(), ToolCallFormat.SharpMind,
            _ => Task.FromResult(ToolPermission.Never),
            CallProbe, " Done.");

        var text = new System.Text.StringBuilder();
        await foreach (var e in session.GetResponseStreamAsync("hi"))
            if (e.Status == ChatStatus.Responding) text.Append(e.Token);

        // The tool body never ran, and the turn still completed with the reply
        // the model gave after the denied call.
        Assert.Equal(before, ExternalProbeTool.Calls);
        Assert.Contains("Done.", text.ToString());
    }

    [Fact]
    public async Task ExternalTool_RunsWhenPermissionIsAlways()
    {
        int before = ExternalProbeTool.Calls;
        ToolPermissionContext? seen = null;
        await using var session = ScriptedSession.CreateGated(
            ProbeAgent(), ToolCallFormat.SharpMind,
            ctx => { seen = ctx; return Task.FromResult(ToolPermission.Always); },
            CallProbe, " Done.");

        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }

        Assert.Equal(before + 1, ExternalProbeTool.Calls);
        Assert.NotNull(seen);
        Assert.Equal("Probe", seen!.ToolName);
    }
}
