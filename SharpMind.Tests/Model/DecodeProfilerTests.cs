using System;
using System.Linq;
using System.Threading.Tasks;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The per-stage decode profiler must be a strict no-op when disabled (zero cost, zero
/// allocation, no timeline changes) and exact when enabled: marks accumulate ticks and counts
/// per stage, clears on Reset, survives parallel callers via Interlocked, and the mark sites
/// actually charge the stages they claim. Everything here leaves the global profiler off.
/// </summary>
/// <remarks>
/// DecodeProfiler is process-wide, so exact per-stage counts would be polluted by any test
/// collection running model layers concurrently (their marks land in the same accumulators and
/// their teardowns could Reset mid-assert). Joining the established "Non-Parallel" collection
/// keeps these assertions deterministic without serialising the whole suite.
/// </remarks>
[Collection("Non-Parallel")]
public class DecodeProfilerTests : IDisposable
{
    public DecodeProfilerTests() { DecodeProfiler.Reset(); }

    public void Dispose()
    {
        DecodeProfiler.Enabled = false;
        DecodeProfiler.Reset();
    }

    [Fact]
    public void DisabledByDefault_BeginReturnsZero_NoMarksRecorded()
    {
        Assert.False(DecodeProfiler.Enabled);
        Assert.Equal(0, DecodeProfiler.Begin());
        DecodeProfiler.Mark(DecodeStage.Matmul, 123);
        Assert.Empty(DecodeProfiler.Snapshot());
        Assert.Equal(0, DecodeProfiler.Count(DecodeStage.Matmul));
        Assert.Equal(0, DecodeProfiler.TotalMs(DecodeStage.Matmul));
    }

    [Fact]
    public void Enabled_MarksChargeTicksAndCounts_ResetClears()
    {
        DecodeProfiler.Enabled = true;
        for (int i = 0; i < 10; i++)
            DecodeProfiler.Mark(DecodeStage.Scores, DecodeProfiler.Begin());

        Assert.Equal(10, DecodeProfiler.Count(DecodeStage.Scores));
        Assert.True(DecodeProfiler.TotalMs(DecodeStage.Scores) > 0);
        Assert.Single(DecodeProfiler.Snapshot());

        DecodeProfiler.Reset();
        Assert.Equal(0, DecodeProfiler.Count(DecodeStage.Scores));
        Assert.Equal(0, DecodeProfiler.TotalMs(DecodeStage.Scores));
        Assert.Empty(DecodeProfiler.Snapshot());
    }

    [Fact]
    public void MarksWithZeroBegin_AreIgnoredEvenWhenEnabled()
    {
        DecodeProfiler.Enabled = true;
        DecodeProfiler.Mark(DecodeStage.Matmul, 0);
        Assert.Equal(0, DecodeProfiler.Count(DecodeStage.Matmul));
    }

    [Fact]
    public void StagesAccumulateIndependently_WithoutNestingContamination()
    {
        DecodeProfiler.Enabled = true;
        long tMatmul = DecodeProfiler.Begin();
        long tScores = DecodeProfiler.Begin(); // nested
        DecodeProfiler.Mark(DecodeStage.Scores, tScores);
        DecodeProfiler.Mark(DecodeStage.Matmul, tMatmul);

        Assert.Equal(1, DecodeProfiler.Count(DecodeStage.Matmul));
        Assert.Equal(1, DecodeProfiler.Count(DecodeStage.Scores));
    }

    [Fact]
    public void SnapshotOrdersStagesByTotalMsDescending()
    {
        DecodeProfiler.Enabled = true;
        long tLmHead = DecodeProfiler.Begin();
        System.Threading.Thread.Sleep(3);
        DecodeProfiler.Mark(DecodeStage.LmHead, tLmHead);
        DecodeProfiler.Mark(DecodeStage.Scores, DecodeProfiler.Begin());

        var rows = DecodeProfiler.Snapshot();
        Assert.Equal(2, rows.Count);
        Assert.Equal(nameof(DecodeStage.LmHead), rows[0].Name);
    }

    [Fact]
    public void MarksFromParallelCallers_AreAllCounted()
    {
        DecodeProfiler.Enabled = true;
        const int N = 20000;
        Parallel.For(0, N, _ =>
            DecodeProfiler.Mark(DecodeStage.Sample, DecodeProfiler.Begin()));

        Assert.Equal(N, DecodeProfiler.Count(DecodeStage.Sample));
    }

    [Fact]
    public void RmsNormForward_ChargesNormStage()
    {
        using var norm = new RmsNormLayer(4);
        using var x = new Tensor<float>(2, 4);
        for (int i = 0; i < 8; i++) x.Data[i] = i + 1;

        DecodeProfiler.Enabled = true;
        using var y = norm.Forward(x);

        Assert.Equal(1, DecodeProfiler.Count(DecodeStage.Norm));
        Assert.True(DecodeProfiler.TotalMs(DecodeStage.Norm) > 0);
        Assert.Equal(2, y.Shape.Rows);
    }

    [Fact]
    public void Format_RendersRecordedRows_AndErrorsGracefullyWhenEmpty()
    {
        string empty = DecodeProfiler.Format("nothing");
        Assert.Contains("no marks", empty);

        DecodeProfiler.Enabled = true;
        DecodeProfiler.Mark(DecodeStage.Embed, DecodeProfiler.Begin());
        string table = DecodeProfiler.Format("decode", 1000);
        Assert.Contains("Embed", table);
        Assert.Contains("%", table);
    }
}