using SharpMind.GPU;

namespace SharpMind.Tests.GPU;

/// <summary>One device per test run. CUDA on the pod, OpenCL on the laptop, ILGPU's CPU accelerator in CI.</summary>
public static class GpuTestDevice
{
    public static GpuDevice Device => GpuDevice.Shared;

    public static float[] Random(int n, int seed, float scale = 0.1f)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    public static void AssertClose(ReadOnlySpan<float> want, ReadOnlySpan<float> got, double relTol, string what)
    {
        Xunit.Assert.Equal(want.Length, got.Length);
        double maxRef = 1e-12, maxErr = 0;
        for (int i = 0; i < want.Length; i++) { maxRef = Math.Max(maxRef, Math.Abs(want[i])); maxErr = Math.Max(maxErr, Math.Abs(want[i] - got[i])); }
        Xunit.Assert.True(maxErr / maxRef < relTol, $"{what}: max rel err {maxErr / maxRef:e2} > {relTol:e1}");
    }
}
