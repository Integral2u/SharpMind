using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace SharpMind.GPU;

/// <summary>A row-major [Rows, Cols] window onto device memory. Does not own it.</summary>
internal readonly struct DeviceTensor(ArrayView<float> view, int rows, int cols)
{
    public ArrayView<float> View { get; } = view;
    public int Rows { get; } = rows;
    public int Cols { get; } = cols;
    public int Length => Rows * Cols;

    public DeviceTensor Slice(int rowStart, int rowCount)
        => new(View.SubView((long)rowStart * Cols, (long)rowCount * Cols), rowCount, Cols);

    public DeviceTensor Reshape(int rows, int cols)
    {
        if ((long)rows * cols != Length) throw new ArgumentException($"Cannot reshape [{Rows},{Cols}] to [{rows},{cols}].");
        return new DeviceTensor(View, rows, cols);
    }

    public void Upload(ReadOnlySpan<float> host)
    {
        if (host.Length != Length) throw new ArgumentException($"Upload of {host.Length} floats into [{Rows},{Cols}].");
        View.CopyFromCPU(host);
    }

    public void Download(Span<float> host)
    {
        if (host.Length != Length) throw new ArgumentException($"Download of [{Rows},{Cols}] into {host.Length} floats.");
        View.CopyToCPU(host);
    }

    public float[] ToArray() { var a = new float[Length]; Download(a); return a; }

    public void Zero()
    {
        // ponytail: ILGPU's implicitly grouped kernels index with int, so this tops out at 2^31
        // floats (8 GB) in one view. Chunk the launch if a single tensor ever gets that big.
        if (View.Length > int.MaxValue) throw new NotSupportedException($"Zero() of {View.Length} floats exceeds ILGPU's int kernel index.");
        var acc = View.GetAccelerator();
        ZeroLaunchers.GetValue(acc, static a => a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>>(ZeroKernel))
            ((int)View.Length, View);
    }

    // ponytail: not View.MemSetToZero() — ILGPU 1.5.3's OpenCL MemSet passes a sub-view's *byte*
    // offset but bounds-checks it against the buffer's *element* count, so it throws once
    // offsetBytes >= elementCount and silently zeroes the wrong window below that. Every arena
    // tensor is a sub-view, so zeroing goes through a kernel. ILGPU caches the compiled kernel;
    // this table caches the launcher delegate per accelerator.
    private static readonly ConditionalWeakTable<Accelerator, Action<Index1D, ArrayView<float>>> ZeroLaunchers = new();

    private static void ZeroKernel(Index1D i, ArrayView<float> v) => v[i] = 0f;
}
