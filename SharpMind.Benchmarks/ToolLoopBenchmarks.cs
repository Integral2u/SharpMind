using System.Text.Json.Nodes;
using BenchmarkDotNet.Attributes;
using SharpMind.Benchmarks.Support;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;

namespace SharpMind.Benchmarks;

/// <summary>
/// Host-side tool-loop latency: one user message that drives three model turns
/// (prompt build, scripted decode, tool-call parse, dispatch, regenerate) and
/// ends in a plain reply. Each format's fixture is first checked to dispatch
/// exactly two calls, so a fixture that stopped parsing fails the run instead
/// of producing a fast-looking number.
/// </summary>
[MemoryDiagnoser]
public class ToolLoopBenchmarks
{
    private const string ToolName = "NativeTool";
    private const string FinalReply = "Done";

    public enum Format { SharpMind, Qwen, Mistral, Llama3, Gemma }

    [Params(Format.SharpMind, Format.Qwen, Format.Mistral, Format.Llama3, Format.Gemma)]
    public Format ToolFormat { get; set; }

    private ToolCallFormat _callFormat;
    private string _call = "";
    private ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>? _session;

    [GlobalSetup]
    public void Setup()
    {
        (_callFormat, _call) = ToolFormat switch
        {
            Format.SharpMind => (ToolCallFormat.SharpMind, """<tool_call>{"tool":"NativeTool","arguments":{"x":"1"}}</tool_call>"""),
            Format.Qwen => (ToolCallFormat.Qwen, """{"tool":"NativeTool","arguments":{"x":"9"}}"""),
            Format.Mistral => (ToolCallFormat.Mistral, """[TOOL_CALLS]{"tool":"NativeTool","arguments":{"x":"9"}}"""),
            Format.Llama3 => (ToolCallFormat.Llama3, """<|python_tag|>{"tool":"NativeTool","arguments":{"x":"9"}}"""),
            Format.Gemma => (ToolCallFormat.Gemma, """<function_call>{"tool":"NativeTool","arguments":{"x":"9"}}</function_call>"""),
            _ => throw new ArgumentOutOfRangeException(nameof(ToolFormat)),
        };

        // Correctness gate: the fixture must produce two dispatched calls and the final reply.
        var check = NewSession();
        try
        {
            int calls = 0;
            check.ProcessToolRequest = (_, _, _) => { calls++; return Task.FromResult(ToolRequestResult.Handled("ok")); };
            string reply = CollectResponding(check, "go").GetAwaiter().GetResult();
            if (calls != 2 || !reply.Contains(FinalReply, StringComparison.Ordinal))
                throw new InvalidOperationException(
                    $"Fixture {ToolFormat} is not a valid tool loop: calls={calls}, reply='{reply}'.");
        }
        finally
        {
            check.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    /// <summary>Fresh session per invocation, so each measured loop starts from the same script.</summary>
    [IterationSetup]
    public void IterationSetup()
    {
        _session = NewSession();
        _session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.Handled("ok"));
    }

    [IterationCleanup]
    public void IterationCleanup()
    {
        _session?.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _session = null;
    }

    /// <summary>Three model turns for one user message. Requires InvocationCount=1 for IterationSetup.</summary>
    [Benchmark]
    public async Task<int> ThreeTurnToolLoop()
    {
        var entries = 0;
        await foreach (var _ in _session!.GetResponseStreamAsync("go"))
            entries++;
        return entries;
    }

    private ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder> NewSession()
        => ScriptedSession.Create(BuilderWithNativeTool(), _callFormat, _call, _call, FinalReply);

    private static async Task<string> CollectResponding(ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder> session, string input)
    {
        var text = new System.Text.StringBuilder();
        await foreach (var e in session.GetResponseStreamAsync(input))
            if (e.Status == ChatStatus.Responding) text.Append(e.Token);
        return text.ToString();
    }

    internal static IAgentBuilder BuilderWithNativeTool()
        => new AgentBuilder("Bench").WithTool(
            ToolName,
            "A native SharpMind tool",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject(),
                ["required"] = new JsonArray()
            },
            args => Task.FromResult("took " + (args?["x"]?.GetValue<string>() ?? "?")));
}
