using ILGPU;
using ILGPU.Runtime;
using SharpMind.Core.Tensors;

namespace SharpMind.GPU;

/// <summary>Owns one device allocation (a weight, a persistent grad) and exposes it as a <see cref="DeviceTensor"/>.</summary>
internal sealed class DeviceBuffer : IDisposable
{
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _buffer;
    public DeviceTensor Tensor { get; }

    public DeviceBuffer(GpuDevice device, int rows, int cols)
    {
        _buffer = device.Accelerator.Allocate1D<float>(DeviceArena.CheckShape(rows, cols));
        Tensor = new DeviceTensor(_buffer.View, rows, cols);
    }

    public static DeviceBuffer From(GpuDevice device, Tensor<float> host)
    {
        int rows = host.Rank >= 2 ? host.ElementCount / host.Shape[^1] : 1;
        int cols = host.Shape[^1];
        var b = new DeviceBuffer(device, rows, cols);
        b.Tensor.Upload(host.Data);
        return b;
    }

    public void Dispose() => _buffer.Dispose();
}
