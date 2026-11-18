using SharpMind.Server;
using Xunit;

namespace SharpMind.Tests.Server;

/// <summary>
/// A loaded model is one Transformer shared by chat sessions and the batched
/// completion generator. Two engines forwarding at once clobber the transformer's
/// activation fields, so every engine takes this lease. These pin the contract:
/// exclusive per model, independent across models, and a waiter can give up.
/// </summary>
public sealed class ModelLeaseServiceTests
{
    [Fact]
    public async Task Lease_IsExclusivePerModel()
    {
        using var leases = new ModelLeaseService();
        var first = await leases.AcquireAsync("a.gguf");

        var second = leases.AcquireAsync("a.gguf").AsTask();
        Assert.False(second.IsCompleted, "a second engine took the model while the first held it");

        first.Dispose();
        using var acquired = await second.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(acquired);
    }

    [Fact]
    public async Task Lease_IsIndependentAcrossModels()
    {
        using var leases = new ModelLeaseService();
        using var a = await leases.AcquireAsync("a.gguf");

        using var b = await leases.AcquireAsync("b.gguf").AsTask().WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(b);
        Assert.True(leases.IsBusy("a.gguf"));
        Assert.True(leases.IsBusy("b.gguf"));
    }

    [Fact]
    public async Task Lease_CanBeAbandonedWhileWaiting()
    {
        using var leases = new ModelLeaseService();
        using var held = await leases.AcquireAsync("a.gguf");

        using var cts = new CancellationTokenSource();
        var waiting = leases.AcquireAsync("a.gguf", cts.Token).AsTask();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
    }

    [Fact]
    public async Task Lease_DoubleDisposeReleasesOnce()
    {
        using var leases = new ModelLeaseService();
        var lease = await leases.AcquireAsync("a.gguf");
        lease.Dispose();
        lease.Dispose();

        // An over-release would leave the gate at count 2 and let two engines in.
        using var again = await leases.AcquireAsync("a.gguf").AsTask().WaitAsync(TimeSpan.FromSeconds(5));
        var blocked = leases.AcquireAsync("a.gguf").AsTask();
        Assert.False(blocked.IsCompleted);
        again.Dispose();
        using var third = await blocked.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(third);
    }
}
