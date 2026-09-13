using ILGPU;
using ILGPU.Runtime;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// Fallback GEMM for accelerators without cuBLAS (OpenCL, CPU). Shared-memory tiles
/// up to 16×16, launched at the largest square tile the accelerator's group size
/// admits — <see cref="PickTile"/>. Measured 1.85 TF on a 3090 vs 18-20 TF for
/// cuBLAS — correct and present, not fast; see tools/GpuSpike/results.
/// </summary>
internal static class GemmKernels
{
    /// <summary>Largest tile; also the shared-memory capacity (a compile-time size).</summary>
    public const int Tile = 16;

    /// <summary>
    /// The largest square tile whose thread count fits <paramref name="maxThreadsPerGroup"/>:
    /// 16 (256 threads) wherever the accelerator allows ≥ 256, falling to 8, 4, 2, 1 for the
    /// tightest group sizes. ILGPU's CPU accelerator reports its (often small) working set here,
    /// so a hardcoded 16×16 group would throw "Not supported total group size" on it.
    /// </summary>
    public static int PickTile(int maxThreadsPerGroup)
    {
        int tile = Tile;
        while (tile > 1 && tile * tile > maxThreadsPerGroup)
            tile /= 2;
        return tile;
    }

    public static Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float, int, int> Load(Accelerator acc)
        => acc.LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int, int, int, int, float, int, int>(Tiled);

    public static KernelConfig Config(int m, int n, int tile)
        => new(new Index2D((n + tile - 1) / tile, (m + tile - 1) / tile), new Index2D(tile, tile));

    // C[i,j] = Σ_t A[i·saI + t·saK] · B[t·sbK + j·sbJ]  (+ beta·C), C's row stride being ldc
    private static void Tiled(ArrayView<float> c, ArrayView<float> a, ArrayView<float> b,
        int m, int n, int k, int saI, int saK, int sbK, int sbJ, float beta, int ldc, int tile)
    {
        // Allocated at capacity Tile (compile-time); only cells below `tile` are touched, and only
        // by the tile×tile group's own threads, so any tile ≤ 16 computes the same product.
        var tileA = SharedMemory.Allocate2DDenseX<float>(new Index2D(Tile, Tile)); // [k, row]
        var tileB = SharedMemory.Allocate2DDenseX<float>(new Index2D(Tile, Tile)); // [col, k]
        int tx = Group.IdxX, ty = Group.IdxY;
        int col = Grid.IdxX * tile + tx;
        int row = Grid.IdxY * tile + ty;
        float s = 0f;
        for (int t0 = 0; t0 < k; t0 += tile)
        {
            int ka = t0 + tx, kb = t0 + ty;
            // long addressing: the lm_head's strides (151936) overflow int once row passes ~14k.
            tileA[tx, ty] = (row < m && ka < k) ? a[(long)row * saI + (long)ka * saK] : 0f;
            tileB[tx, ty] = (kb < k && col < n) ? b[(long)kb * sbK + (long)col * sbJ] : 0f;
            Group.Barrier();
            for (int t = 0; t < tile; t++) s += tileA[t, ty] * tileB[tx, t];
            Group.Barrier();
        }
        if (row < m && col < n)
        {
            long idx = (long)row * ldc + col;
            c[idx] = beta == 0f ? s : s + beta * c[idx];
        }
    }
}
