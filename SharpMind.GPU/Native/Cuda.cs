using System.Runtime.InteropServices;

namespace SharpMind.GPU.Native;

// ponytail: only the cuBLAS entry points the engine actually calls, nothing more.
// dotLLM's pattern (P/Invoke the system libraries, no custom native lib) — names only, no code copied.
internal static class Cublas
{
    private const string Lib = "cublas";

    public const int OpN = 0, OpT = 1;
    public const int MathDefault = 0, MathTf32TensorOp = 3;

    [DllImport(Lib, EntryPoint = "cublasCreate_v2")] public static extern int Create(out IntPtr handle);
    [DllImport(Lib, EntryPoint = "cublasDestroy_v2")] public static extern int Destroy(IntPtr handle);
    [DllImport(Lib, EntryPoint = "cublasSetStream_v2")] public static extern int SetStream(IntPtr handle, IntPtr stream);
    [DllImport(Lib, EntryPoint = "cublasSetMathMode")] public static extern int SetMathMode(IntPtr handle, int mode);
    [DllImport(Lib, EntryPoint = "cublasGetVersion_v2")] public static extern int GetVersion(IntPtr handle, out int version);

    [DllImport(Lib, EntryPoint = "cublasSgemm_v2")]
    public static extern int Sgemm(IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr a, int lda, IntPtr b, int ldb, ref float beta, IntPtr c, int ldc);

    public static void Check(int rc, string what)
    {
        if (rc != 0) throw new InvalidOperationException($"{what}: cuBLAS status {rc}");
    }

    /// <summary>
    /// Row-major C[m×n] = A·B + beta·C where A is addressed as A[i*saI + k*saK] and B as B[k*sbK + j*sbJ].
    /// Exactly one of (saI, saK) is 1 and exactly one of (sbK, sbJ) is 1. Translated to the
    /// column-major call Cᵀ = Bᵀ·Aᵀ so no data is ever transposed.
    /// </summary>
    public static void GemmRowMajor(IntPtr h, IntPtr c, IntPtr a, IntPtr b, int m, int n, int k,
        int saI, int saK, int sbK, int sbJ, float beta = 0f)
    {
        int transA = saK == 1 ? OpN : OpT, lda = saK == 1 ? saI : saK;
        int transB = sbJ == 1 ? OpN : OpT, ldb = sbJ == 1 ? sbK : sbJ;
        float alpha = 1f;
        Check(Sgemm(h, transB, transA, n, m, k, ref alpha, b, ldb, a, lda, ref beta, c, n), "cublasSgemm");
    }
}

internal static class NativeResolver
{
    private static readonly string[] CudaCandidates = ["nvcuda.dll", "libcuda.so.1", "libcuda.so"];
    private static readonly string[] CublasCandidates =
    [
        "cublas64_13.dll", "cublas64_12.dll",
        "libcublas.so.13", "libcublas.so.12", "libcublas.so",
        "/usr/local/cuda/lib64/libcublas.so.13", "/usr/local/cuda/lib64/libcublas.so.12", "/usr/local/cuda/lib64/libcublas.so",
    ];

    private static int _installed;

    /// <summary>Idempotent: SetDllImportResolver throws if one is already registered for the assembly.</summary>
    public static void Install()
    {
        if (Interlocked.Exchange(ref _installed, 1) != 0) return;
        NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, (name, _, _) => name switch
        {
            "cuda" => TryLoad(CudaCandidates),
            "cublas" => TryLoad(CublasCandidates),
            _ => IntPtr.Zero,
        });
    }

    private static IntPtr TryLoad(string[] candidates)
    {
        foreach (var c in candidates)
            if (NativeLibrary.TryLoad(c, out var h)) return h;
        return IntPtr.Zero;
    }
}
