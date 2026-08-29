using ILGPU;
using ILGPU.Runtime;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// Fallback GEMM for accelerators without cuBLAS (OpenCL, CPU). 16×16 shared-memory
/// tiles. Measured 1.85 TF on a 3090 vs 18-20 TF for cuBLAS — correct and present,
/// not fast; see tools/GpuSpike/results.
/// </summary>
internal static class GemmKernels
{
    public const int Tile = 16;

    public static Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float, int> Load(Accelerator acc)
        => acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float, int>(Tiled16);

    public static KernelConfig Config(int m, int n)
        => new(new Index2D((n + Tile - 1) / Tile, (m + Tile - 1) / Tile), new Index2D(Tile, Tile));

    // C[i,j] = Σ_t A[i·saI + t·saK] · B[t·sbK + j·sbJ]  (+ beta·C), C's row stride being ldc
    private static void Tiled16(ArrayView<float> c, ArrayView<float> a, ArrayView<float> b,
        int m, int n, int k, int saI, int saK, int sbK, int sbJ, float beta, int ldc)
    {
        var tileA = SharedMemory.Allocate2DDenseX<float>(new Index2D(Tile, Tile)); // [k, row]
        var tileB = SharedMemory.Allocate2DDenseX<float>(new Index2D(Tile, Tile)); // [col, k]
        int tx = Group.IdxX, ty = Group.IdxY;
        int col = Grid.IdxX * Tile + tx;
        int row = Grid.IdxY * Tile + ty;
        float s = 0f;
        for (int t0 = 0; t0 < k; t0 += Tile)
        {
            int ka = t0 + tx, kb = t0 + ty;
            // long addressing: the lm_head's strides (151936) overflow int once row passes ~14k.
            tileA[tx, ty] = (row < m && ka < k) ? a[(long)row * saI + (long)ka * saK] : 0f;
            tileB[tx, ty] = (kb < k && col < n) ? b[(long)kb * sbK + (long)col * sbJ] : 0f;
            Group.Barrier();
            for (int t = 0; t < Tile; t++) s += tileA[t, ty] * tileB[tx, t];
            Group.Barrier();
        }
        if (row < m && col < n)
        {
            long idx = (long)row * ldc + col;
            c[idx] = beta == 0f ? s : s + beta * c[idx];
        }
    }
}
