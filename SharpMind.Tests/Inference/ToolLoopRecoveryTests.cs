using System.Text.Json.Nodes;
using SharpMind.Core.AgentTools;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The tool loop must always end in a reply, must not repeat itself, and must recover
/// from a call it could not read. Each test drives the real session loop with a
/// scripted model.
/// </summary>
public sealed class ToolLoopRecoveryTests
{
    private const string Call = "<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":\"red\"}}</tool_call>";
    private const string OtherCall = "<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":\"blue\"}}</tool_call>";

    private static IAgentBuilder Agent()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTool("Measure", "Measures.",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject { ["label"] = new JsonObject { ["type"] = "string" } },
                ["required"] = new JsonArray("label")
            },
            args => Task.FromResult("ok"));
        return agent;
    }

    private static async Task<(string Reply, int Dispatches)> Run(
        string[] replies, int? maxCalls = null, Action<ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>>? setup = null)
    {
        await using var session = ScriptedSession.Create(Agent(), ToolCallFormat.SharpMind, replies);
        if (maxCalls is int m) session.MaxToolCallsPerTurn = m;
        setup?.Invoke(session);
        int dispatches = 0;
        session.ProcessToolRequest = (_, _, _) =>
        {
            dispatches++;
            return Task.FromResult(ToolRequestResult.Handled("ok"));
        };

        var text = new System.Text.StringBuilder();
        await foreach (var e in session.GetResponseStreamAsync("hi"))
            if (e.Status == ChatStatus.Responding) text.Append(e.Token);
        return (text.ToString(), dispatches);
    }

    // (d) loop limits end in a non-empty reply

    [Fact]
    public async Task BudgetReached_EndsWithAFinalAnswer()
    {
        // The model keeps calling, with a different argument each time so none is a repeat.
        string[] replies =
        [
            Call,
            OtherCall,
            "<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":\"green\"}}</tool_call>",
            "All done: red, blue and green were measured."
        ];
        var (reply, dispatches) = await Run(replies, maxCalls: 2);

        Assert.Equal(2, dispatches);
        Assert.Contains("All done", reply);
    }

    [Fact]
    public async Task RepeatedIdenticalCall_IsNotRunTwice_AndTheModelIsToldToAnswer()
    {
        string[] replies = [Call, Call, "Red it is."];
        var (reply, dispatches) = await Run(replies);

        Assert.Equal(1, dispatches);
        Assert.Contains("Red it is.", reply);
    }

    [Fact]
    public async Task RepeatedCall_WithDifferentSpacingAndKeyOrder_CountsAsRepeat()
    {
        const string spaced = "<tool_call>{ \"arguments\" : {\"label\":\"red\"}, \"name\":\"Measure\" }</tool_call>";
        var (_, dispatches) = await Run([Call, spaced, "Done."]);
        Assert.Equal(1, dispatches);
    }

    [Fact]
    public void CallKey_IsCanonical()
    {
        var a = JsonNode.Parse("""{"b":1,"a":"x"}""")!.AsObject();
        var b = JsonNode.Parse("""{"a":"x","b":1}""")!.AsObject();
        Assert.Equal(SharpMind.Inference.Chat.ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>.CallKey("T", a),
                     SharpMind.Inference.Chat.ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>.CallKey("T", b));
    }

    // Fabricated results: the model writes a result header itself instead of calling the tool

    [Fact]
    public async Task FabricatedResultHeader_IsNeverShown_AndTheRealCallRuns()
    {
        string[] replies =
        [
            "[Measure label=red] ok",          // invented result, not a call
            Call,                              // the model then makes the real call
            "Red it is."
        ];
        var (reply, dispatches) = await Run(replies);

        Assert.Equal(1, dispatches);
        Assert.DoesNotContain("[Measure", reply);
        Assert.Contains("Red it is.", reply);
    }

    [Fact]
    public async Task FabricatedResultHeader_RepeatedBeyondRetries_IsNotShown()
    {
        const string fake = "[Measure label=red] ok";
        string[] replies = [fake, fake, fake, "Plain answer."];
        var (reply, dispatches) = await Run(replies);

        Assert.Equal(0, dispatches);
        Assert.DoesNotContain("[Measure", reply);
        Assert.Contains("couldn't complete that request", reply);
    }

    [Fact]
    public async Task BracketedProse_IsNotMistakenForAResultHeader()
    {
        // "[Note]" is not a registered tool, so it is ordinary text and is shown.
        var (reply, dispatches) = await Run(["[Note] This is an ordinary answer."]);

        Assert.Equal(0, dispatches);
        Assert.Contains("[Note] This is an ordinary answer.", reply);
    }

    // (d) malformed calls: bounded corrective retries, never an empty turn

    [Fact]
    public async Task MalformedCall_IsCorrectedAndTheModelRecovers()
    {
        // Bad JSON inside the block, then a plain answer after the correction.
        string[] replies =
        [
            "<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":red}}</tool_call>",
            "Sorry, I can answer that directly: hello."
        ];
        var (reply, dispatches) = await Run(replies);

        Assert.Equal(0, dispatches);
        Assert.Contains("hello", reply);
    }

    [Fact]
    public async Task MalformedCall_RetriesAreBounded_AndTheTurnStillReplies()
    {
        // The model never produces a valid call. Two corrective retries are allowed; after
        // that the turn ends with a plain, non-empty reply rather than silence. The fourth
        // scripted reply is never requested, which shows the retries are bounded.
        const string broken = "<tool_call>{\"name\":\"Measure\",\"arguments\":{\"label\":red}}</tool_call>";
        string[] replies = [broken, broken, broken, "Here is my plain answer."];
        var (reply, dispatches) = await Run(replies);

        Assert.Equal(0, dispatches);
        Assert.Contains("couldn't complete that request", reply);
        Assert.DoesNotContain("plain answer", reply);
    }

    // (e) result formatting and truncation

    [Fact]
    public async Task FormatToolResult_HasHeaderWithKeyArguments_AndPlainBody()
    {
        await using var session = ScriptedSession.Create(Agent(), ToolCallFormat.SharpMind, "x");
        var result = new JsonObject { ["status"] = "success", ["data"] = "sunny, 21C" };
        string text = session.FormatToolResult("GetWeather", new JsonObject { ["city"] = "Paris" }, result);

        Assert.Equal("[GetWeather city=Paris] sunny, 21C", text);
    }

    [Fact]
    public async Task FormatToolResult_Error_IsPrefixed()
    {
        await using var session = ScriptedSession.Create(Agent(), ToolCallFormat.SharpMind, "x");
        string text = session.FormatToolResult("ReadFile", new JsonObject { ["path"] = "a.txt" },
            new JsonObject { ["status"] = "error", ["message"] = "file not found" });

        Assert.StartsWith("[ReadFile path=a.txt] ERROR: file not found", text);
    }

    [Fact]
    public async Task FormatToolResult_LongOutput_IsCappedWithNote()
    {
        await using var session = ScriptedSession.Create(Agent(), ToolCallFormat.SharpMind, "x");
        session.MaxToolResultChars = 100;
        string text = session.FormatToolResult("Search", new JsonObject(),
            new JsonObject { ["status"] = "success", ["data"] = new string('a', 500) });

        Assert.True(text.Length < 200);
        Assert.Contains("[truncated: ", text);
        Assert.Contains("offset/limit", text);
    }

    [Fact]
    public async Task ProcessToolRequestResult_UsesTheSameFormatter()
    {
        await using var session = ScriptedSession.Create(Agent(), ToolCallFormat.SharpMind, Call, "Done.");
        session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.Handled("external"));
        await foreach (var _ in session.GetResponseStreamAsync("hi")) { }

        var resultMessage = session.RawHistoryForTests
            .FirstOrDefault(m => m.Content.Contains("external"));
        Assert.NotNull(resultMessage);
        Assert.StartsWith("[Measure label=red]", resultMessage!.Content);
    }
}
