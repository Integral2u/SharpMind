using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SharpMind.GPU.Kernels;
using SharpMind.GPU.Native;

namespace SharpMind.GPU;

/// <summary>
/// The one accelerator the engine runs on, its stream, and — on CUDA with
/// libcublas present — a cuBLAS handle bound to that stream. All GEMMs go through
/// <see cref="Gemm"/>; everything else is an ILGPU kernel.
/// </summary>
public sealed class GpuDevice : IDisposable
{
    private static readonly Lock _sharedLock = new();
    private static GpuDevice? _shared;

    /// <summary>
    /// The process-wide device, created on first use. A failed attempt is NOT cached — the next
    /// caller retries — so a transient first-run CUDA/cuBLAS problem does not poison the process
    /// (which is what <see cref="Lazy{T}"/> with ExecutionAndPublication would do).
    /// </summary>
    public static GpuDevice Shared
    {
        get
        {
            if (_shared is not null) return _shared;
            lock (_sharedLock) return _shared ??= Create();
        }
    }

    private readonly Context _context;
    private readonly IntPtr _cublas;
    private readonly Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float> _tiled;
    private bool _disposed;
    private bool _tf32;

    internal Accelerator Accelerator { get; }
    internal GpuKernels Kernels { get; }
    public bool HasCublas => _cublas != IntPtr.Zero;

    /// <summary>True when no real GPU was found and the engine is running on ILGPU's CPU accelerator.</summary>
    public bool IsCpuFallback => Accelerator.AcceleratorType == AcceleratorType.CPU;

    /// <summary>Allow TF32 tensor-core GEMMs on cuBLAS. Set once, before the GEMMs start.</summary>
    public bool UseTf32
    {
        get => _tf32;
        set
        {
            _tf32 = value;
            // Applied here rather than per-GEMM: it is a host-side cuBLAS call and Gemm is the hot path.
            if (HasCublas)
                Cublas.Check(Cublas.SetMathMode(_cublas, value ? Cublas.MathTf32TensorOp : Cublas.MathDefault), "cublasSetMathMode");
        }
    }

    public string Description { get; }

    public static GpuDevice Create(bool preferCpu = false)
    {
        // EnableAlgorithms is REQUIRED on the CUDA backend: XMath's Exp/Log/Sqrt/Tanh
// have no intrinsic there without it ("The function 'ExpF' does not have an
// intrinsic implementation for this backend"). OpenCL happens to accept plain
// Math.* and hid this until the first real CUDA run.
        var ctx = Context.Create(b => b.Default().EnableAlgorithms());
        Accelerator? acc = null;
        try
        {
            if (!preferCpu && ctx.GetCudaDevices().Count > 0) acc = ctx.GetCudaDevice(0).CreateCudaAccelerator(ctx);
            else if (!preferCpu && ctx.GetCLDevices().Count > 0) acc = ctx.GetCLDevice(0).CreateCLAccelerator(ctx);
            else acc = ctx.GetPreferredDevice(preferCPU: true).CreateAccelerator(ctx);
            return new GpuDevice(ctx, acc);
        }
        catch
        {
            // The constructor can throw after Bind() (any nonzero cuBLAS status). Nothing owns the
            // accelerator or context until it returns, so unwind them here rather than leaking a
            // driver context for the lifetime of the process.
            acc?.Dispose();
            ctx.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Returns null with a reason instead of throwing when no real accelerator is present.
    /// The plugin path needs that question answered rather than signalled: the host turns the
    /// reason into the message the user reads, and must never fall back silently.
    ///
    /// Unlike <see cref="Create"/> this refuses ILGPU's CPU accelerator. A plugin named "cuda"
    /// quietly executing on the CPU would be slower than the CPU engine it displaced, and the
    /// user asked for a GPU.
    /// </summary>
    public static GpuDevice? TryCreate(out string? reason)
    {
        Context? ctx = null;
        try
        {
            ctx = Context.Create(b => b.Default().EnableAlgorithms());
            if (ctx.GetCudaDevices().Count == 0 && ctx.GetCLDevices().Count == 0)
            {
                reason = "no CUDA or OpenCL device is available on this machine.";
                ctx.Dispose();
                return null;
            }
        }
        catch (Exception ex)
        {
            ctx?.Dispose();
            reason = $"the GPU driver could not be initialised: {ex.GetBaseException().Message}";
            return null;
        }

        ctx.Dispose();      // Create builds its own; this one only answered the question.
        try
        {
            reason = null;
            return Create();
        }
        catch (Exception ex)
        {
            reason = $"the GPU device could not be opened: {ex.GetBaseException().Message}";
            return null;
        }
    }

    private GpuDevice(Context ctx, Accelerator acc)
    {
        _context = ctx;
        Accelerator = acc;
        Kernels = new GpuKernels(acc);
        _tiled = GemmKernels.Load(acc);
        string blas = "tiled16";
        if (acc is CudaAccelerator cuda)
        {
            NativeResolver.Install();
            try
            {
                cuda.Bind();   // make ILGPU's driver context current so the runtime-API cuBLAS adopts it
                Cublas.Check(Cublas.Create(out _cublas), "cublasCreate");
                Cublas.Check(Cublas.SetStream(_cublas, ((CudaStream)acc.DefaultStream).StreamPtr), "cublasSetStream");
                Cublas.GetVersion(_cublas, out int v);
                blas = $"cuBLAS {v / 10000}.{v % 10000 / 100}";
            }
            catch (DllNotFoundException) { _cublas = IntPtr.Zero; blas = "tiled16 (libcublas not found)"; }
            catch
            {
                // A nonzero status from SetStream/GetVersion leaves a live handle nobody can reach,
                // since the constructor never returns. Release it; Create unwinds acc + ctx.
                if (_cublas != IntPtr.Zero) { Cublas.Destroy(_cublas); _cublas = IntPtr.Zero; }
                throw;
            }
        }
        Description = $"[{acc.AcceleratorType}] {acc.Name}, {acc.MemorySize >> 20} MB, {blas}";
    }

    /// <summary>Row-major C[m×n] = A·B + beta·C with A[i,k]=a[i·saI+k·saK], B[k,j]=b[k·sbK+j·sbJ].</summary>
    internal void Gemm(DeviceTensor c, DeviceTensor a, DeviceTensor b, int m, int n, int k, int saI, int saK, int sbK, int sbJ, float beta = 0f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        // cuBLAS reads one stride of each operand as its leading dimension and requires the other to
        // be 1; the tiled kernel would compute any stride pair happily. Without this the two backends
        // silently disagree — green on OpenCL, garbage on the pod.
        if (saI < 1 || saK < 1 || (saI == 1) == (saK == 1))
            throw new ArgumentException($"A strides (saI={saI}, saK={saK}) must be positive with exactly one equal to 1.");
        if (sbK < 1 || sbJ < 1 || (sbK == 1) == (sbJ == 1))
            throw new ArgumentException($"B strides (sbK={sbK}, sbJ={sbJ}) must be positive with exactly one equal to 1.");
        if (c.Length < (long)m * n) throw new ArgumentException($"C holds {c.Length} floats, GEMM needs {m}×{n}.");
        // Highest element each operand is addressed at, +1. An undersized operand reads out of
        // bounds — on the cuBLAS path in Release that is memory corruption, not an exception.
        long needA = (long)(m - 1) * saI + (long)(k - 1) * saK + 1;
        long needB = (long)(k - 1) * sbK + (long)(n - 1) * sbJ + 1;
        if (a.Length < needA) throw new ArgumentException($"A holds {a.Length} floats, GEMM needs {needA} at strides (saI={saI}, saK={saK}).");
        if (b.Length < needB) throw new ArgumentException($"B holds {b.Length} floats, GEMM needs {needB} at strides (sbK={sbK}, sbJ={sbJ}).");
        if (HasCublas)
        {
            Cublas.GemmRowMajor(_cublas, Ptr(c), Ptr(a), Ptr(b), m, n, k, saI, saK, sbK, sbJ, beta);
        }
        else
        {
            _tiled(GemmKernels.Config(m, n), c.View, a.View, b.View, m, n, k, saI, saK, sbK, sbJ, beta);
        }
    }

    private static unsafe IntPtr Ptr(DeviceTensor t) => (IntPtr)t.View.LoadEffectiveAddressAsPtr();

    public void Synchronize() => Accelerator.Synchronize();

    internal DeviceIntBuffer UploadInts(int[] host) => new(Accelerator, host);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_cublas != IntPtr.Zero) Cublas.Destroy(_cublas);
        Accelerator.Dispose();
        _context.Dispose();
    }
}

/// <summary>Owns a device-side int array — token ids for <see cref="GpuKernels.EmbedGather"/>.</summary>
internal sealed class DeviceIntBuffer : IDisposable
{
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _buffer;
    public ArrayView<int> View => _buffer.View;
    internal DeviceIntBuffer(Accelerator acc, int[] host) { _buffer = acc.Allocate1D(host); }
    public void Dispose() => _buffer.Dispose();
}
