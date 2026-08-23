using System.Numerics;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Memory;

/// <summary>
/// Pre-allocated memory workspace interface.
/// Provides a way to "rent" slices of a large contiguous buffer as Tensors
/// without allocating new memory.
/// </summary>
public interface IWorkspace : IDisposable
{
    /// <summary>
    /// Returns a Tensor that views a slice of the workspace.
    /// </summary>
    Tensor<T> Rent<T>(ReadOnlySpan<int> shape) where T : unmanaged, INumber<T>;

    /// <summary>
    /// Resets the offset to 0, effectively "freeing" all rented tensors.
    /// </summary>
    void Reset();

    long UsedBytes { get; }
    long CapacityBytes { get; }
    float UsagePercentage { get; }
}
