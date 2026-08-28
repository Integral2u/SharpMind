using ILGPU;
using ILGPU.Runtime;

namespace SharpMind.GPU;

/// <summary>
/// Bump allocator for a step's activations: one device allocation, handed out
/// in order, reset once per step. Mirrors Core's Workspace. Rent() never touches
/// the driver, so the per-op path has no allocation cost.
/// </summary>
internal sealed class DeviceArena : IDisposable
{
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _buffer;
    private long _used;

    public DeviceArena(GpuDevice device, long capacityFloats)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacityFloats);
        _buffer = device.Accelerator.Allocate1D<float>(capacityFloats);
    }

    public long Capacity => _buffer.Length;
    public long Used => _used;

    public DeviceTensor Rent(int rows, int cols)
    {
        long n = CheckShape(rows, cols);
        long start = (_used + 31) & ~31L;   // 128-byte aligned starts
        if (start + n > Capacity)
            throw new InvalidOperationException($"DeviceArena exhausted: need {n} floats at {start}, capacity {Capacity}. Size the arena from the model and batch shape.");
        _used = start + n;
        return new DeviceTensor(_buffer.View.SubView(start, n), rows, cols);
    }

    public void Reset() => _used = 0;

    public void Dispose() => _buffer.Dispose();

    /// <summary>
    /// Validates a tensor shape at the point of creation and returns its element count.
    /// <see cref="DeviceTensor.Length"/> is an <c>int</c> (as are the ILGPU launch extents that
    /// consume it), so a shape whose product overflows must be rejected here rather than surfacing
    /// downstream as a negative length in a GEMM argument message. At V = 151936 the logits tensor
    /// crosses this at m ≈ 14k rows.
    /// </summary>
    internal static long CheckShape(int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        long n = (long)rows * cols;
        if (n > int.MaxValue)
            throw new NotSupportedException($"[{rows},{cols}] exceeds the int element index DeviceTensor and ILGPU launches use.");
        return n;
    }
}
