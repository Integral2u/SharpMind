using BenchmarkDotNet.Attributes;
using SharpMind.Benchmarks.Support;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;

namespace SharpMind.Benchmarks;

/// <summary>
/// Host cost of streaming a long plain-prose reply, one fragment at a time.
/// Before the incremental markup filter, each fragment re-sanitised the whole
/// buffer, so this cost grew quadratically with response length. Scaling across
/// the Length parameter shows whether it is still linear.
/// </summary>
[MemoryDiagnoser]
public class StreamingBenchmarks
{
    // Fragments of ~4 characters, roughly how the decoder emits text.
    private const int FragmentChars = 4;

    [Params(1_000, 4_000, 16_000, 64_000)]
    public int Length { get; set; }

    private string _reply = "";
    private IAgentBuilder _agent = null!;
    private ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>? _session;

    [GlobalSetup]
    public void Setup()
    {
        var text = new System.Text.StringBuilder(Length + FragmentChars);
        var words = new[] { "alpha ", "beta ", "gamma ", "delta ", "epsilon " };
        for (int i = 0; text.Length < Length; i++) text.Append(words[i % words.Length]);
        text.Length = Length;

        // Fragment boundaries are the '|' separators the scripted generator splits on.
        var fragments = new System.Text.StringBuilder(Length + Length / FragmentChars);
        for (int i = 0; i < text.Length; i += FragmentChars)
        {
            if (i > 0) fragments.Append('|');
            fragments.Append(text.ToString(i, Math.Min(FragmentChars, text.Length - i)));
        }
        _reply = fragments.ToString();
        _agent = ToolLoopBenchmarks.BuilderWithNativeTool();
    }

    [IterationSetup]
    public void IterationSetup()
    {
        _session = ScriptedSession.Create(_agent, ToolCallFormat.SharpMind, _reply);
    }

    [IterationCleanup]
    public void IterationCleanup()
    {
        _session?.DisposeAsync().AsTask().GetAwaiter().GetResult();
        _session = null;
    }

    [Benchmark]
    public async Task<int> StreamLongReply()
    {
        int chars = 0;
        await foreach (var e in _session!.GetResponseStreamAsync("summarise"))
            if (e.Status == ChatStatus.Responding) chars += e.Token.Length;
        return chars;
    }
}
