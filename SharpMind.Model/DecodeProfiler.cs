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
    /// <summary>The quantized/F32 matmul data path <em>inside</em> one <see cref="Matmul"/>
    /// call (the wide transform, the raw quantized kernel, or the float fallback). Sits inside
    /// <see cref="Matmul"/> so its ~1.5&nbsp;MiB-of-allocation-per-call behaviour has a bucket
    /// of its own; <see cref="Matmul"/> minus <see cref="Kernel"/> is the per-layer envelope.</summary>
    Kernel,
    /// <summary><c>Architecture.Forward</c> — the per-block loop shared by every forward.
    /// All its visible work (Matmul, Norm, Rope, Kv, Scores, Act) is staged inside; the
    /// difference between <see cref="Blocks"/> and its children is block glue.</summary>
    Blocks,
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
public readonly record struct DecodeStageTiming(string Name, double TotalMs, long Count, long AllocatedBytes);

/// <summary>
/// The snapshot a <see cref="DecodeProfiler.Begin"/> hands back to <see cref="DecodeProfiler.Mark"/>:
/// the wall-clock start and the process-wide managed-allocation counter at that instant.
/// Both are zero when profiling was off, and <see cref="DecodeProfiler.Mark"/> then ignores it.
/// </summary>
public readonly struct DecodeMeasure
{
    /// <summary>Stopwatch reading captured at <see cref="DecodeProfiler.Begin"/> (0 when disabled).</summary>
    public readonly long Ticks;
    /// <summary><c>GC.GetTotalAllocatedBytes()</c> captured at <see cref="DecodeProfiler.Begin"/> (0 when disabled).</summary>
    public readonly long AllocatedBytes;

    public DecodeMeasure(long ticks, long allocatedBytes)
    {
        Ticks = ticks;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>True when profiling was off: a plain <c>Begin()</c> with nothing valid to measure.</summary>
    public bool IsNoOp => Ticks == 0;
}

/// <summary>
/// Per-stage breakdown of CPU decode — wall clock and managed allocation — shared by every mark
/// site in the model (transformer, layers, FFN kernels) and the generator (sampling).
///
/// <para><b>Opt-in and cheap when off.</b> Each site is a <see cref="Begin"/> snapshot and a
/// <see cref="Mark"/> call; while <see cref="Enabled"/> is false <see cref="Begin"/> returns a
/// no-op token without reading the clock or the allocation counter, so the default state costs
/// one predictable branch per site. A profiled run is slower than a real one — take throughput
/// with the profiler off and the breakdown with it on.</para>
///
/// <para><b>Static and thread-safe.</b> One window at a time, accumulating through
/// <see cref="Interlocked"/> so marks fired from inside <c>Parallel.For</c> regions (MoE
/// per-token routing, batched forwards) are counted rather than racing. Nested marks are
/// independent buckets: a MoE expert's matmul is counted in <see cref="Matmul"/>, not in the
/// route scope around its selection, and <see cref="Forward"/> envelopes every decode stage —
/// do not sum a parent with its children.</para>
///
/// <para><b>Allocation attribution.</b> Each Begin/Mark pair contributes the <em>process-wide</em>
/// managed bytes allocated through its interval (<c>GC.GetTotalAllocatedBytes()</c> delta), so
/// work done on worker threads by a stage is included. Two caveats: nested scopes overlap exactly
/// as time does, and marks fired concurrently from inside <c>Parallel.For</c> (MoE routing) can
/// double-count bytes that fall in overlapping worker windows — treat Route's allocation number
/// as approximate, everything else as exact for the mostly-main-thread decode path.</para>
/// </summary>
public static class DecodeProfiler
{
    private static readonly int StageCount = Enum.GetValues<DecodeStage>().Length;
    private static readonly long[] s_ticks = new long[StageCount];
    private static readonly long[] s_allocBytes = new long[StageCount];
    private static readonly long[] s_counts = new long[StageCount];
    private static volatile bool s_enabled;

    /// <summary>Whether marks are recorded. Default false; the probe's <c>stages</c> mode and
    /// tests flip it around a measurement window.</summary>
    public static bool Enabled
    {
        get => s_enabled;
        set => s_enabled = value;
    }

    /// <summary>Opens a mark: the snapshot to hand back to <see cref="Mark"/>, or a no-op token
    /// when profiling is off (which <see cref="Mark"/> then ignores).</summary>
    public static DecodeMeasure Begin() =>
        s_enabled
            ? new DecodeMeasure(Stopwatch.GetTimestamp(), GC.GetTotalAllocatedBytes())
            : default;

    /// <summary>Closes the interval opened by <see cref="Begin"/> and charges its elapsed time
    /// and allocated bytes to <paramref name="stage"/>. Ignored when profiling was off.</summary>
    public static void Mark(DecodeStage stage, in DecodeMeasure m)
    {
        if (!s_enabled || m.IsNoOp) return;
        int i = (int)stage;
        Interlocked.Add(ref s_ticks[i], Stopwatch.GetTimestamp() - m.Ticks);
        Interlocked.Add(ref s_allocBytes[i], GC.GetTotalAllocatedBytes() - m.AllocatedBytes);
        Interlocked.Increment(ref s_counts[i]);
    }

    /// <summary>Drops everything recorded so far. Call after the warm-up window: JIT promotion,
    /// the first workspace growth and the prefill all land there.</summary>
    public static void Reset()
    {
        for (int i = 0; i < StageCount; i++)
        {
            Interlocked.Exchange(ref s_ticks[i], 0);
            Interlocked.Exchange(ref s_allocBytes[i], 0);
            Interlocked.Exchange(ref s_counts[i], 0);
        }
    }

    /// <summary>Milliseconds accumulated for <paramref name="stage"/> since the last <see cref="Reset"/>.</summary>
    public static double TotalMs(DecodeStage stage) =>
        Interlocked.Read(ref s_ticks[(int)stage]) * 1000.0 / Stopwatch.Frequency;

    /// <summary>How many marks of <paramref name="stage"/> were recorded since the last <see cref="Reset"/>.</summary>
    public static long Count(DecodeStage stage) => Interlocked.Read(ref s_counts[(int)stage]);

    /// <summary>Managed bytes allocated while <paramref name="stage"/> was active since the last <see cref="Reset"/>.</summary>
    public static long AllocatedBytes(DecodeStage stage) => Interlocked.Read(ref s_allocBytes[(int)stage]);

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
                count,
                Interlocked.Read(ref s_allocBytes[i])));
        }
        return [.. list.OrderByDescending(p => p.TotalMs)];
    }

    /// <summary>
    /// Renders <see cref="Snapshot"/> as a table. Time percentages are of <paramref name="wallMs"/>
    /// when the caller supplies one (so unaccounted host time shows up as a gap), otherwise of the
    /// recorded stages themselves. The MiB column is each stage's managed allocation, which is
    /// <em>not</em> part of the percentage math — bytes don't add up like wall time across nested
    /// scopes.
    /// </summary>
    public static string Format(string title, double wallMs = 0)
    {
        var rows = Snapshot();
        double denominator = wallMs > 0 ? wallMs : rows.Sum(r => r.TotalMs);
        var sb = new StringBuilder();
        sb.Append($"--- {title}  ({(rows.Count > 0 ? "recorded" : "no marks")}, ")
          .AppendLine($"{denominator:F1} ms window) ---");
        sb.Append($"{"stage",-12} {"total ms",10} {"%",7} {"calls",8} {"us/call",10} {"MiB",9}").AppendLine();
        foreach (var r in rows)
        {
            double pct = denominator > 0 ? 100.0 * r.TotalMs / denominator : 0;
            double usPerCall = r.Count > 0 ? r.TotalMs * 1000.0 / r.Count : 0;
            sb.Append($"{r.Name,-12} {r.TotalMs,10:F1} {pct,6:F1}% {r.Count,8} {usPerCall,10:F1} {r.AllocatedBytes / (1024.0 * 1024.0),9:F1}").AppendLine();
        }
        return sb.ToString();
    }
}