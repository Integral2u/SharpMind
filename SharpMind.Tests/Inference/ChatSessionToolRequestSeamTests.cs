using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Tests for the <see cref="IChatSession.ProcessToolRequest"/> seam: the
/// host-facing interception point that lets a caller decide, per tool call,
/// whether to run it itself (<see cref="ToolRequestOutcome.Handled"/>), let
/// the session's native agent loop run it
/// (<see cref="ToolRequestOutcome.Defer"/>), or take the whole call back and
/// end the turn (<see cref="ToolRequestOutcome.ReturnToCaller"/>).
/// </summary>
public sealed class ChatSessionToolRequestSeamTests
{
    private const string FinalReply = "Done";
    private const string NativeResult = "native result";

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
            args => Task.FromResult(NativeResult));

    private static string ToolCall(string tool, string argsJson)
        => $"<tool_call>{{\"tool\":\"{tool}\",\"arguments\":{argsJson}}}</tool_call>";

    private static string RespondingText(IEnumerable<ChatStreamEntry> entries)
        => string.Concat(entries.Where(e => e.Status == ChatStatus.Responding).Select(e => e.Token));

    [Fact]
    public async Task Handled_FeedsExternalResultBackAndContinuesLoop()
    {
        int calls = 0;
        string? seenName = null;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCall("NativeTool", """{"x":"1"}"""),
            FinalReply);
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            calls++;
            seenName = toolName;
            return Task.FromResult(ToolRequestResult.Handled("external result"));
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal(1, calls);
        Assert.Equal("NativeTool", seenName);
        // The external result is fed back and the loop continues to the final reply
        // (if it hadn't been fed back, the loop would request a 3rd reply and the
        // script would be exhausted before "Done" was streamed).
        Assert.Contains(FinalReply, RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task Defer_DispatchesThroughNativeAgentLoop()
    {
        int calls = 0;
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCall("NativeTool", """{"x":"1"}"""),
            FinalReply);
        session.ProcessToolRequest = (toolName, args, ct) =>
        {
            calls++;
            return Task.FromResult(ToolRequestResult.Defer());
        };

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Equal(1, calls);
        // The native tool ran and its result was fed back, then the loop continued.
        Assert.Contains(FinalReply, RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }

    [Fact]
    public async Task ReturnToCaller_EndsTurnWithToolCallEntry_WithoutDispatching()
    {
        int dispatchCalls = 0;
        var builder = new AgentBuilder("Test");
        builder.WithTool(
            "NativeTool",
            "A native SharpMind tool",
            new JsonObject { ["type"] = "object" },
            args => { dispatchCalls++; return Task.FromResult(NativeResult); });

        await using var session = ScriptedSession.Create(
            builder,
            ToolCall("NativeTool", """{"x":"1"}"""));
        session.ProcessToolRequest = (toolName, args, ct)
            => Task.FromResult(ToolRequestResult.ReturnToCaller());

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        // The native tool never ran and the turn ended at the tool call, without
        // dispatching, without a result feed-back and without a final reply.
        Assert.Equal(0, dispatchCalls);
        Assert.DoesNotContain("Tool result:", RespondingText(entries));
        Assert.DoesNotContain(FinalReply, RespondingText(entries));
        Assert.DoesNotContain(entries, e => e.Status == ChatStatus.Complete);
        // The stream ended with a ToolCall entry carrying the call.
        var toolEntry = entries.LastOrDefault(e => e.Status == ChatStatus.ToolCall);
        Assert.NotNull(toolEntry);
        Assert.Equal("NativeTool", toolEntry!.ToolCall?["tool"]?.GetValue<string>());
        Assert.Equal("1", toolEntry.ToolCall?["arguments"]?["x"]?.GetValue<string>());
        Assert.True(entries.Last().IsComplete);
    }

    [Fact]
    public async Task NullSeam_KeepsBuiltInNativeDispatch()
    {
        await using var session = ScriptedSession.Create(
            BuilderWithNativeTool(),
            ToolCall("NativeTool", """{"x":"1"}"""),
            FinalReply);
        Assert.Null(session.ProcessToolRequest);

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        Assert.Contains(FinalReply, RespondingText(entries));
        Assert.Contains(entries, e => e.Status == ChatStatus.Complete);
    }
}
