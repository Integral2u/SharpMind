namespace SharpMind.GPU.Tests;

/// <summary>
/// The profiler's own bookkeeping. The timings it produces are wall clock and not worth
/// asserting on, but what it counts, what it ignores and what it clears all are — those are
/// what make a breakdown trustworthy, and all three have been wrong at some point.
/// </summary>
public class GpuStepProfilerTests
{
    private static GpuStepProfiler Enabled()
    {
        var p = new GpuStepProfiler(GpuTestDevice.Device);
        p.Enabled = true;
        return p;
    }

    [Fact]
    public void DisabledProfilerRecordsNothing()
    {
        var p = new GpuStepProfiler(GpuTestDevice.Device) { Enabled = false };
        p.BeginStep();
        p.Mark("fwd/attn");
        p.EndStep();

        Assert.Equal(0, p.Steps);
        Assert.Empty(p.Snapshot());
        Assert.Equal(0, p.TotalMs);
    }

    [Fact]
    public void MarksOutsideAStepAreIgnored()
    {
        var p = Enabled();
        p.Mark("stray/before");           // never opened a step
        p.BeginStep();
        p.Mark("fwd/attn");
        p.EndStep();
        p.Mark("stray/after");            // step already closed

        Assert.Equal(1, p.Steps);
        Assert.Equal("fwd/attn", Assert.Single(p.Snapshot()).Name);
    }

    [Fact]
    public void RepeatedPhasesAccumulateAndCount()
    {
        var p = Enabled();
        for (int step = 0; step < 2; step++)
        {
            p.BeginStep();
            for (int layer = 0; layer < 3; layer++) p.Mark("fwd/norm");
            p.Mark("fwd/lm-head");
            p.EndStep();
        }

        Assert.Equal(2, p.Steps);
        var phases = p.Snapshot();
        Assert.Equal(2, phases.Count);
        // A per-layer phase is marked once per layer per step; a per-step phase once per step.
        Assert.Equal(6, phases.Single(x => x.Name == "fwd/norm").Count);
        Assert.Equal(2, phases.Single(x => x.Name == "fwd/lm-head").Count);
        Assert.Equal(p.TotalMs, phases.Sum(x => x.TotalMs), 6);
    }

    [Fact]
    public void SnapshotIsOrderedByCost()
    {
        var p = Enabled();
        p.BeginStep();
        p.Mark("cheap");
        GpuTestDevice.Device.Synchronize();
        Thread.Sleep(15);                 // the only way to make one phase reliably dearer
        p.Mark("dear");
        p.EndStep();

        var phases = p.Snapshot();
        Assert.Equal("dear", phases[0].Name);
        Assert.True(phases[0].TotalMs > phases[1].TotalMs);
    }

    [Fact]
    public void ResetClearsEverythingIncludingTheOpenStep()
    {
        var p = Enabled();
        p.BeginStep();
        p.Mark("fwd/attn");
        p.Reset();

        Assert.Equal(0, p.Steps);
        Assert.Empty(p.Snapshot());
        Assert.Equal(0, p.TotalMs);

        // Reset closed the step it interrupted, so a mark before the next BeginStep cannot be
        // charged the whole gap since the reset. This is what a bench harness does between its
        // warm-up and timed passes.
        p.Mark("fwd/attn");
        Assert.Empty(p.Snapshot());
    }

    [Fact]
    public void FormatListsEveryPhaseUnderTheTitle()
    {
        var p = Enabled();
        p.BeginStep();
        p.Mark("fwd/attn");
        p.Mark("bwd/attn");
        p.EndStep();

        var lines = p.Format("b=2 breakdown", "  ").Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Contains("b=2 breakdown", lines[0]);
        Assert.Contains("1 steps", lines[0]);
        Assert.All(lines, l => Assert.StartsWith("  ", l));
        Assert.Contains(lines, l => l.Contains("fwd/attn"));
        Assert.Contains(lines, l => l.Contains("bwd/attn"));
    }

    [Fact]
    public void EngineExposesADisabledProfilerByDefault()
    {
        // SM_PROF is not set in the test environment, so nothing a test run does should pay for
        // the per-mark Synchronize — the whole reason the marks can stay in the shipped engine.
        Assert.False(GpuStepProfiler.EnabledByDefault);
    }
}
