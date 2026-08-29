using System.Diagnostics;
using System.Text;

namespace SharpMind.GPU;

/// <summary>One phase's accumulated cost across the profiled steps.</summary>
/// <param name="Name">The label the engine passed to <see cref="GpuStepProfiler.Mark"/>.</param>
/// <param name="TotalMs">Wall-clock milliseconds summed over every mark carrying this label.</param>
/// <param name="Count">How many marks contributed, i.e. how often the phase ran — once per
/// layer per step for the block phases, once per step for the head and the host transfers.</param>
public readonly record struct GpuPhaseTiming(string Name, double TotalMs, long Count);

/// <summary>
/// Per-phase wall-clock breakdown of a GPU training step, reached as
/// <see cref="GpuBackpropEngine.Profiler"/>.
///
/// <para><b>Enabling it changes what it measures.</b> Every mark calls
/// <see cref="GpuDevice.Synchronize"/>, because kernel launches are asynchronous: without a
/// barrier a phase's cost lands on whichever later phase happens to block, and the breakdown is
/// noise. Those barriers drain the pipeline that normally overlaps launch with execution, so a
/// profiled step is slower than a real one. Take throughput from a run with profiling off and
/// the breakdown from one with it on; do not read both off the same run.</para>
///
/// <para>Disabled it is one bool test per mark — the engine leaves the calls in place rather
/// than compiling them out, so a breakdown never needs a rebuild.</para>
///
/// <para>Not thread-safe, and does not need to be: one profiler belongs to one engine and is
/// marked only from the thread driving that engine's forward and backward.</para>
/// </summary>
public sealed class GpuStepProfiler
{
    /// <summary>
    /// Whether a new profiler starts enabled: true when the <c>SM_PROF</c> environment variable
    /// is <c>1</c>. An environment switch rather than a compile-time one because SharpMind.GPU
    /// is loaded from a path as a plugin — whoever wants a breakdown usually cannot rebuild it.
    /// </summary>
    public static bool EnabledByDefault { get; } = Environment.GetEnvironmentVariable("SM_PROF") == "1";

    private readonly GpuDevice _device;
    private readonly Dictionary<string, (double Ms, long Count)> _phases = [];
    private long _last;
    private bool _inStep;

    /// <param name="device">The device whose queue each mark drains before reading the clock.</param>
    public GpuStepProfiler(GpuDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        Enabled = EnabledByDefault;
    }

    /// <summary>Whether marks are recorded. Defaults to <see cref="EnabledByDefault"/>.</summary>
    public bool Enabled { get; set; }

    /// <summary>Steps recorded since the last <see cref="Reset"/>.</summary>
    public int Steps { get; private set; }

    /// <summary>Milliseconds attributed to phases — the accounted part of those steps.</summary>
    public double TotalMs { get; private set; }

    /// <summary>Opens a step: drains the queue so the first phase is not charged for the last
    /// step's tail, and starts the interval the next <see cref="Mark"/> closes.</summary>
    public void BeginStep()
    {
        if (!Enabled) return;
        _device.Synchronize();
        _last = Stopwatch.GetTimestamp();
        _inStep = true;
        Steps++;
    }

    /// <summary>
    /// Closes the interval since the last mark and charges it to <paramref name="phase"/>.
    /// Ignored outside a <see cref="BeginStep"/>/<see cref="EndStep"/> pair, so the marks inside
    /// the shared forward do not accumulate when it is called off the training path.
    /// </summary>
    public void Mark(string phase)
    {
        if (!Enabled || !_inStep) return;
        ArgumentNullException.ThrowIfNull(phase);
        _device.Synchronize();
        long now = Stopwatch.GetTimestamp();
        double ms = (now - _last) * 1000.0 / Stopwatch.Frequency;
        _last = now;
        _phases.TryGetValue(phase, out var cur);
        _phases[phase] = (cur.Ms + ms, cur.Count + 1);
        TotalMs += ms;
    }

    /// <summary>Closes the step. Time after the final mark is not charged to any phase.</summary>
    public void EndStep() => _inStep = false;

    /// <summary>
    /// Drops everything recorded so far. Call it after the warm-up steps: JIT, the first kernel
    /// compile and the arena's first growth all land on those, and averaging them into the
    /// timed steps flatters whichever phase happened to run first.
    /// </summary>
    public void Reset()
    {
        _phases.Clear();
        Steps = 0;
        TotalMs = 0;
        _last = 0;
        _inStep = false;
    }

    /// <summary>The phases recorded so far, most expensive first.</summary>
    public IReadOnlyList<GpuPhaseTiming> Snapshot() =>
        [.. _phases.Select(p => new GpuPhaseTiming(p.Key, p.Value.Ms, p.Value.Count))
                   .OrderByDescending(p => p.TotalMs)];

    /// <summary>Renders <see cref="Snapshot"/> as a table, one phase per line.</summary>
    /// <param name="title">Names the run in the header, e.g. the batch shape.</param>
    /// <param name="indent">Prefixed to every line, for nesting under a caller's own output.</param>
    public string Format(string title, string indent = "")
    {
        int steps = Math.Max(Steps, 1);
        var sb = new StringBuilder();
        sb.Append(indent).Append($"--- {title}  ({Steps} steps, {TotalMs / steps:F1} ms/step accounted) ---").AppendLine();
        sb.Append(indent).Append($"{"phase",-22} {"ms/step",9} {"%",7}").AppendLine();
        foreach (var p in Snapshot())
            sb.Append(indent).Append($"{p.Name,-22} {p.TotalMs / steps,9:F2} {(TotalMs > 0 ? 100 * p.TotalMs / TotalMs : 0),6:F1}%").AppendLine();
        return sb.ToString();
    }
}
