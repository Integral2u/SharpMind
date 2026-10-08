using System.Diagnostics;
using System.Text;

namespace SharpMind.Model;

/// <summary>One stage of a single-token decode step, timed by <see cref="DecodeProfiler"/>.</summary>
public enum DecodeStage
{
    /// <summary>Whole single-token forward: embedding, every block, final norm, lm head.</summary>
    Forward,
    /// <summary>Prompt prefill forward (chunks whose seqLen &gt; 1) — kept separate so a
    /// profiled window can never silently mix prefill into the decode-step total.</summary>
    Prefill,
    /// <summary>Token embedding lookup plus optional embedding scale / position embedding.</summary>
    Embed,
    /// <summary>Every RMSNorm/LayerNorm row: pre/post block norms, Q/K norms, final norm.</summary>
    Norm,
    /// <summary><see cref="Layers.InferenceLinearLayer.Forward"/> — q/k/v/o, FFN, MoE router and
    /// expert projections. Excludes the lm head, which is <see cref="LmHead"/>.</summary>
    Matmul,
    /// <summary>RoPE (or ALiBi slope setup) applied to Q and K.</summary>
    Rope,
    /// <summary>Writing this step's K/V into the cache.</summary>
    Kv,
    /// <summary>Attention itself: the per-head scaled-dot-product loop over the cached K/V.</summary>
    Scores,
    /// <summary>Activation/gating arithmetic between the two FFN matmuls.</summary>
    Act,
    /// <summary>MoE routing: router softmax, top-k selection, weight normalisation.</summary>
    Route,
    /// <summary>Vocabulary projection (<c>LogitOps.Project</c>).</summary>
    LmHead,
    /// <summary>Generator-side per-token work: sampling plus detokenizing the chosen id.</summary>
    Sample,
}

/// <summary>One stage's accumulated cost across the profiled window.</summary>
public readonly record struct DecodeStageTiming(string Name, double TotalMs, long Count);

/// <summary>
/// Per-stage wall-clock breakdown of CPU decode, shared by every mark site in the model
/// (transformer, layers, FFN kernels) and the generator (sampling).
///
/// <para><b>Opt-in and cheap when off.</b> Each site is a <see cref="Begin"/> timestamp and a
/// <see cref="Mark"/> call; while <see cref="Enabled"/> is false <see cref="Begin"/> returns 0
/// without reading the clock, so the default state costs one predictable-branch per site.
/// A profiled run is slower than a real one — take throughput with the profiler off and the
/// breakdown with it on.</para>
///
/// <para><b>Static and thread-safe.</b> One window at a time, accumulating through
/// <see cref="Interlocked"/> so marks fired from inside <c>Parallel.For</c> regions (MoE
/// per-token routing, batched forwards) are counted rather than racing. Nested marks are
/// independent buckets: a MoE expert's matmul is counted in <see cref="Matmul"/>, not in the
/// route scope around its selection, and <see cref="Forward"/> envelopes every decode stage —
/// do not sum a parent with its children.</para>
/// </summary>
public static class DecodeProfiler
{
    private static readonly int StageCount = Enum.GetValues<DecodeStage>().Length;
    private static readonly long[] s_ticks = new long[StageCount];
    private static readonly long[] s_counts = new long[StageCount];
    private static volatile bool s_enabled;

    /// <summary>Whether marks are recorded. Default false; the probe's <c>stages</c> mode and
    /// tests flip it around a measurement window.</summary>
    public static bool Enabled
    {
        get => s_enabled;
        set => s_enabled = value;
    }

    /// <summary>Opens a mark: the clock reading to hand back to <see cref="Mark"/>, or 0 when
    /// profiling is off (which <see cref="Mark"/> then ignores).</summary>
    public static long Begin() => s_enabled ? Stopwatch.GetTimestamp() : 0;

    /// <summary>Closes the interval opened by <see cref="Begin"/> and charges it to
    /// <paramref name="stage"/>. Ignored when profiling was off at <see cref="Begin"/>.</summary>
    public static void Mark(DecodeStage stage, long start)
    {
        if (!s_enabled || start == 0) return;
        long elapsed = Stopwatch.GetTimestamp() - start;
        int i = (int)stage;
        Interlocked.Add(ref s_ticks[i], elapsed);
        Interlocked.Increment(ref s_counts[i]);
    }

    /// <summary>Drops everything recorded so far. Call after the warm-up window: JIT promotion,
    /// the first workspace growth and the prefill all land there.</summary>
    public static void Reset()
    {
        for (int i = 0; i < StageCount; i++)
        {
            Interlocked.Exchange(ref s_ticks[i], 0);
            Interlocked.Exchange(ref s_counts[i], 0);
        }
    }

    /// <summary>Milliseconds accumulated for <paramref name="stage"/> since the last <see cref="Reset"/>.</summary>
    public static double TotalMs(DecodeStage stage) =>
        Interlocked.Read(ref s_ticks[(int)stage]) * 1000.0 / Stopwatch.Frequency;

    /// <summary>How many marks of <paramref name="stage"/> were recorded since the last <see cref="Reset"/>.</summary>
    public static long Count(DecodeStage stage) => Interlocked.Read(ref s_counts[(int)stage]);

    /// <summary>The stages recorded so far, most expensive first.</summary>
    public static IReadOnlyList<DecodeStageTiming> Snapshot()
    {
        var list = new List<DecodeStageTiming>(StageCount);
        for (int i = 0; i < StageCount; i++)
        {
            long ticks = Interlocked.Read(ref s_ticks[i]);
            long count = Interlocked.Read(ref s_counts[i]);
            if (ticks == 0 && count == 0) continue;
            list.Add(new DecodeStageTiming(
                ((DecodeStage)i).ToString(),
                ticks * 1000.0 / Stopwatch.Frequency,
                count));
        }
        return [.. list.OrderByDescending(p => p.TotalMs)];
    }

    /// <summary>
    /// Renders <see cref="Snapshot"/> as a table. Percentages are of <paramref name="wallMs"/>
    /// when the caller supplies one (so unaccounted host time shows up as a gap), otherwise of
    /// the recorded stages themselves.
    /// </summary>
    public static string Format(string title, double wallMs = 0)
    {
        var rows = Snapshot();
        double denominator = wallMs > 0 ? wallMs : rows.Sum(r => r.TotalMs);
        var sb = new StringBuilder();
        sb.Append($"--- {title}  ({(rows.Count > 0 ? "recorded" : "no marks")}, ")
          .AppendLine($"{denominator:F1} ms window) ---");
        sb.Append($"{"stage",-12} {"total ms",10} {"%",7} {"calls",8} {"us/call",10}").AppendLine();
        foreach (var r in rows)
        {
            double pct = denominator > 0 ? 100.0 * r.TotalMs / denominator : 0;
            double usPerCall = r.Count > 0 ? r.TotalMs * 1000.0 / r.Count : 0;
            sb.Append($"{r.Name,-12} {r.TotalMs,10:F1} {pct,6:F1}% {r.Count,8} {usPerCall,10:F1}").AppendLine();
        }
        return sb.ToString();
    }
}
