using ILGPU;
using ILGPU.Runtime;

namespace SharpMind.GPU;

/// <summary>
/// Owns one device allocation of raw bytes — a GGUF quantized weight or the token embedding's
/// raw Q8_0 data. Exposed directly as an <see cref="ArrayView{T}"/> because every consumer
/// (<see cref="GpuKernels"/>/quant kernels) addresses it as flat bytes, never as a shaped float
/// tensor.
/// </summary>
internal sealed class DeviceByteBuffer : IDisposable
{
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _buffer;
    public ArrayView<byte> View => _buffer.View;

    public DeviceByteBuffer(Accelerator acc, ReadOnlySpan<byte> host)
    {
        var copy = host.ToArray();
        _buffer = acc.Allocate1D(copy);
    }

    public int Length => checked((int)_buffer.Length);

    public void Dispose() => _buffer.Dispose();
}
