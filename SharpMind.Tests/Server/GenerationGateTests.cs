using System;
using System.Threading;
using System.Threading.Tasks;
using SharpMind.Server;
using Xunit;

namespace SharpMind.Tests.Server;

/// <summary>
/// The gate is what keeps an overloaded server from converting queue depth into
/// memory pressure and timeouts: a fixed number of generations run, a bounded
/// number wait, everything else is refused. These pin the accounting that makes
/// that true, since an over-release silently raises the cap.
/// </summary>
public sealed class GenerationGateTests
{
    [Fact]
    public async Task EnterAsync_UpToTheCap_DoesNotBlock()
    {
        using var gate = new GenerationGate(maxConcurrent: 2, maxQueued: 0);

        using var first = await gate.EnterAsync();
        using var second = await gate.EnterAsync();

        Assert.Equal(2, gate.Active);
        Assert.Equal(0, gate.Queued);
    }

    [Fact]
    public async Task EnterAsync_BeyondTheCap_WaitsUntilASlotIsReleased()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 1);

        var first = await gate.EnterAsync();
        var second = gate.EnterAsync();

        Assert.False(second.IsCompleted);
        Assert.Equal(1, gate.Active);
        Assert.Equal(1, gate.Queued);

        first.Dispose();

        using var lease = await second.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, gate.Active);
        Assert.Equal(0, gate.Queued);
    }

    [Fact]
    public async Task EnterAsync_WhenTheQueueIsFull_RefusesImmediately()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 1);

        var first = await gate.EnterAsync();
        var queued = gate.EnterAsync();

        var ex = await Assert.ThrowsAsync<GenerationQueueFullException>(() => gate.EnterAsync());
        Assert.Equal(1, ex.MaxConcurrent);
        Assert.Equal(1, ex.MaxQueued);

        first.Dispose();
        using var lease = await queued.WaitAsync(TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task MaxQueuedZero_RefusesAsSoonAsTheCapIsBusy()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 0);

        using var first = await gate.EnterAsync();

        await Assert.ThrowsAsync<GenerationQueueFullException>(() => gate.EnterAsync());
    }

    [Fact]
    public async Task CanceledWaiter_ReleasesItsQueuePlace()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 1);
        using var cts = new CancellationTokenSource();

        var first = await gate.EnterAsync();
        var queued = gate.EnterAsync(cts.Token);
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => queued);
        Assert.Equal(0, gate.Queued);

        // The place is free again, so a later caller queues instead of being
        // refused while the cap is still held.
        var next = gate.EnterAsync();
        first.Dispose();
        using var lease = await next.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, gate.Active);
    }

    [Fact]
    public void Lease_DisposedTwice_ReleasesOnce()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 0);

        var lease = gate.EnterAsync().GetAwaiter().GetResult();
        lease.Dispose();
        lease.Dispose();   // must not release the slot a second time

        Assert.Equal(0, gate.Active);
        Assert.True(gate.TryEnter(out var again));
        Assert.False(gate.TryEnter(out _));   // cap is still one
        again.Dispose();
    }

    [Fact]
    public async Task TryEnter_WithTheCapBusy_ReturnsFalseAndTakesNothing()
    {
        using var gate = new GenerationGate(maxConcurrent: 1, maxQueued: 8);

        using var first = await gate.EnterAsync();

        Assert.False(gate.TryEnter(out _));
        Assert.Equal(1, gate.Active);
        Assert.Equal(0, gate.Queued);
    }
}
