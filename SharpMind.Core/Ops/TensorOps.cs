using JigSawDotNet;
using SharpMind.Core.Activations;
using SharpMind.Core.Tensors;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Ops;

// ─────────────────────────────────────────────────────────────────────────────
// TensorOps
//
// Abstract class assembled by JigSawDotNet. The inner matmul kernel is selected
// once at factory time — no per-call Avx2.IsSupported check.
//
// Public static convenience methods (MatMul, Add, Scale…) remain accessible
// without instantiation — they delegate to the DefaultInstance which is set
// by TensorOpsFactory at startup.
// ─────────────────────────────────────────────────────────────────────────────
public abstract class TensorOps
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Activations)}.{nameof(ActivationKernels)}";

    // ── Singleton set by TensorOpsFactory ─────────────────────────────────
    private static volatile TensorOps? _default;

    /// <summary>
    /// The default assembled instance used by static convenience methods.
    /// Set once by <see cref="TensorOpsFactory.SetDefault"/>.
    /// </summary>
    public static TensorOps Default => _default
        ?? throw new InvalidOperationException(
            $"{nameof(TensorOps)}.{nameof(Default)} has not been initialised. " +
            $"Call {nameof(TensorOpsFactory)}.{nameof(TensorOpsFactory.SetDefault)} at application startup.");

    internal static void SetDefault(TensorOps instance) => _default = instance;

    // ═══════════════════════════════════════════════════════════════════════
    // Abstract kernel — PuzzleCornerPiece selects AVX2 or scalar path
    // Signature takes raw pointers so the kernel owns the inner loop entirely;
    // the public wrapper handles allocation and B-transpose.
    // ═══════════════════════════════════════════════════════════════════════

    [PuzzleCornerPiece(SharpMindConfig.KeyMatMul, true, null,
        SharpMindConfig.ValFma, $"{NS}.{nameof(ActivationKernels.MatMulInnerFMA)}",
        SharpMindConfig.ValAvx2, $"{NS}.{nameof(ActivationKernels.MatMulInnerAVX2)}",
        SharpMindConfig.ValScalar, $"{NS}.{nameof(ActivationKernels.MatMulInnerScalar)}")]
    public abstract unsafe void MatMulInner(float* a, float* bt, float* c, int M, int K, int N);

    // ═══════════════════════════════════════════════════════════════════════
    // Instance matmul — transposes B, allocates result, calls kernel
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>Computes C = A @ B and returns a new tensor.</summary>
    public Tensor<float> MatMul(Tensor<float> a, Tensor<float> b)
    {
        TensorShape.AssertMatMulCompatible(a.Shape, b.Shape);
        int M = a.Shape.Rows, K = a.Shape.Cols, N = b.Shape.Cols;
        var c = new Tensor<float>(M, N);
        using var bt = TransposeInternal(b);
        unsafe { MatMulInner(a.DataPtr, bt.DataPtr, c.DataPtr, M, K, N); }
        return c;
    }

    /// <summary>
    /// Computes C = A @ B where <paramref name="bt"/> is already stored as B-transposed.
    /// </summary>
    /// <remarks>
    /// This is a memory-critical fast path: the inner kernel consumes B in transposed layout
    /// (<c>[N, K]</c>) so we can avoid allocating a temporary transpose tensor.
    /// Shapes: A=[M,K], BT=[N,K] → C=[M,N].
    /// </remarks>
    public Tensor<float> MatMulWithBT(Tensor<float> a, Tensor<float> bt)
    {
        AssertMatMulWithBTCompatible(a.Shape, bt.Shape);
        int M = a.Shape.Rows, K = a.Shape.Cols, N = bt.Shape.Rows;
        var c = new Tensor<float>(M, N);
        unsafe { MatMulInner(a.DataPtr, bt.DataPtr, c.DataPtr, M, K, N); }
        return c;
    }

    /// <summary>
    /// Computes C = A @ B into a pre-allocated output tensor, with B provided in transposed layout.
    /// </summary>
    public void MatMulWithBTInto(Tensor<float> a, Tensor<float> bt, Tensor<float> c)
    {
        AssertMatMulWithBTCompatible(a.Shape, bt.Shape);
        if (c.Rank != 2 || c.Shape.Rows != a.Shape.Rows || c.Shape.Cols != bt.Shape.Rows)
            throw new ArgumentException($"Output shape must be [{a.Shape.Rows}, {bt.Shape.Rows}], got {c.Shape}.");
        int M = a.Shape.Rows, K = a.Shape.Cols, N = bt.Shape.Rows;
        unsafe { MatMulInner(a.DataPtr, bt.DataPtr, c.DataPtr, M, K, N); }
    }
    /// <summary>Computes C = A @ B into a pre-allocated output tensor.</summary>
    public void MatMulInto(Tensor<float> a, Tensor<float> b, Tensor<float> c)
    {
        TensorShape.AssertMatMulCompatible(a.Shape, b.Shape);
        int M = a.Shape.Rows, K = a.Shape.Cols, N = b.Shape.Cols;
        using var bt = TransposeInternal(b);
        unsafe { MatMulInner(a.DataPtr, bt.DataPtr, c.DataPtr, M, K, N); }
    }
    // ═══════════════════════════════════════════════════════════════════════
    // Batched MatMul — [*, M, K] × [*, K, N] → [*, M, N]
    // Handles 3D and 4D (batch × heads × seq × dim).
    // All batch dims are flattened — the inner 2D kernel is reused.
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Batched matrix multiply. Supports rank 3 [B,M,K]×[B,K,N] and
    /// rank 4 [B,H,M,K]×[B,H,K,N] — the pattern used by multi-head attention.
    /// All leading batch dims must match.
    /// </summary>
    public Tensor<float> BatchedMatMul(Tensor<float> a, Tensor<float> b)
    {
        AssertBatchedMatMulCompatible(a.Shape, b.Shape);

        int rank = a.Shape.Rank;
        int M = a.Shape[rank - 2];
        int K = a.Shape[rank - 1];
        int N = b.Shape[rank - 1];

        // Batch = product of all dims before the last two
        int batch = a.Shape.ElementCount / (M * K);

        int[] outDims = (int[])a.Shape.Dims.ToArray();//.Clone();
        outDims[rank - 1] = N;
        var result = new Tensor<float>(new TensorShape(outDims));

        using var bt = TransposeLast2D(b, M: K, N: N, batch: batch);

        int aStride = M * K;
        int bStride = K * N;
        int cStride = M * N;

        unsafe
        {
            float* pA = a.DataPtr;
            float* pBT = bt.DataPtr;
            float* pC = result.DataPtr;

            for (int i = 0; i < batch; i++)
                MatMulInner(pA + i * aStride, pBT + i * bStride, pC + i * cStride, M, K, N);
        }

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Static convenience methods — delegate to Default instance
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>C = A @ B using the default assembled instance.</summary>
    public static Tensor<float> MatMul(Tensor<float> a, Tensor<float> b, TensorOps? ops = null)
        => (ops ?? Default).MatMul(a, b);

    public static Tensor<float> BatchedMatMul(Tensor<float> a, Tensor<float> b, TensorOps? ops = null)
        => (ops ?? Default).BatchedMatMul(a, b);

    // ── SAIM operator types (JIT-inlinable, no delegate indirection) ────

    private interface IBinaryOp<T> where T : unmanaged, INumber<T>
    {
        static abstract Vector<T> Invoke(Vector<T> a, Vector<T> b);
        static abstract T InvokeScalar(T a, T b);
    }

    private readonly struct AddOp<T> : IBinaryOp<T> where T : unmanaged, INumber<T>
    {
        public static Vector<T> Invoke(Vector<T> a, Vector<T> b) => a + b;
        public static T InvokeScalar(T a, T b) => a + b;
    }

    private readonly struct SubtractOp<T> : IBinaryOp<T> where T : unmanaged, INumber<T>
    {
        public static Vector<T> Invoke(Vector<T> a, Vector<T> b) => a - b;
        public static T InvokeScalar(T a, T b) => a - b;
    }

    private readonly struct MultiplyOp<T> : IBinaryOp<T> where T : unmanaged, INumber<T>
    {
        public static Vector<T> Invoke(Vector<T> a, Vector<T> b) => a * b;
        public static T InvokeScalar(T a, T b) => a * b;
    }

    private readonly struct DivideOp<T> : IBinaryOp<T> where T : unmanaged, INumber<T>
    {
        public static Vector<T> Invoke(Vector<T> a, Vector<T> b) => a / b;
        public static T InvokeScalar(T a, T b) => a / b;
    }

    // ── Generic elementwise ops (no JigSaw — Vector<T> handles dispatch) ──

    /// <summary>Returns a + b elementwise.</summary>
    public static Tensor<T> Add<T>(Tensor<T> a, Tensor<T> b)
       where T : unmanaged, INumber<T>
    {
        TensorShape.AssertSameShape(a.Shape, b.Shape);
        var r = new Tensor<T>(a.Shape);
        BinaryOp<T, AddOp<T>>(a.Data, b.Data, r.Data);
        return r;
    }

    public static Tensor<T> Subtract<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, INumber<T>
    {
        TensorShape.AssertSameShape(a.Shape, b.Shape);
        var r = new Tensor<T>(a.Shape);
        BinaryOp<T, SubtractOp<T>>(a.Data, b.Data, r.Data);
        return r;
    }

    public static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, INumber<T>
    {
        TensorShape.AssertSameShape(a.Shape, b.Shape);
        var r = new Tensor<T>(a.Shape);
        BinaryOp<T, MultiplyOp<T>>(a.Data, b.Data, r.Data);
        return r;
    }

    public static Tensor<T> Divide<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, INumber<T>
    {
        TensorShape.AssertSameShape(a.Shape, b.Shape);
        var r = new Tensor<T>(a.Shape);
        BinaryOp<T, DivideOp<T>>(a.Data, b.Data, r.Data);
        return r;
    }

    /// <summary>Returns a * scalar elementwise.</summary>
    public static Tensor<T> Scale<T>(Tensor<T> a, T scalar)
        where T : unmanaged, INumber<T>
    {
        var r = new Tensor<T>(a.Shape);
        ScaleVectorized(a.Data, r.Data, scalar);
        return r;
    }

    /// <summary>In-place a[i] += b[i].</summary>
    public static void AddInPlace<T>(Tensor<T> a, Tensor<T> b)
        where T : unmanaged, INumber<T>
    {
        TensorShape.AssertSameShape(a.Shape, b.Shape);
        BinaryOp<T, AddOp<T>>(a.Data, b.Data, a.Data);
    }

    public static void ScaleInPlace<T>(Tensor<T> a, T scalar)
        where T : unmanaged, INumber<T>
    {
        ScaleVectorized(a.Data, a.Data, scalar);
    }
    /// <summary>Clamps all elements to [min, max].</summary>
    public static Tensor<T> Clamp<T>(Tensor<T> a, T min, T max)
        where T : unmanaged, INumber<T>
    {
        var r = new Tensor<T>(a.Shape);
        var src = a.Data;
        var dst = r.Data;
        int vecLen = Vector<T>.Count;
        var vMin = new Vector<T>(min);
        var vMax = new Vector<T>(max);
        int i = 0;
        for (; i <= src.Length - vecLen; i += vecLen)
            Vector.Min(vMax, Vector.Max(vMin, new Vector<T>(src[i..]))).CopyTo(dst[i..]);
        for (; i < src.Length; i++)
            dst[i] = T.Clamp(src[i], min, max);
        return r;
    }

    /// <summary>Elementwise square root.</summary>
    public static unsafe Tensor<float> Sqrt(Tensor<float> a)
    {
        var r = new Tensor<float>(a.Shape);
        var src = a.Data;
        var dst = r.Data;
        int i = 0;
        if (Avx.IsSupported)
        {
            fixed (float* pSrc = src, pDst = dst)
            {
                for (; i <= src.Length - 8; i += 8)
                    Avx.Sqrt(Vector256.LoadUnsafe(ref pSrc[i])).StoreUnsafe(ref pDst[i]);
            }
        }
        for (; i < src.Length; i++)
            dst[i] = MathF.Sqrt(src[i]);
        return r;
    }

    /// <summary>Elementwise absolute value.</summary>
    public static Tensor<T> Abs<T>(Tensor<T> a)
        where T : unmanaged, INumber<T>
    {
        var r = new Tensor<T>(a.Shape);
        var src = a.Data;
        var dst = r.Data;
        int vecLen = Vector<T>.Count;
        int i = 0;
        for (; i <= src.Length - vecLen; i += vecLen)
            Vector.Abs(new Vector<T>(src[i..])).CopyTo(dst[i..]);
        for (; i < src.Length; i++)
            dst[i] = T.Abs(src[i]);
        return r;
    }
    /// <summary>
    /// Fills positions where <paramref name="mask"/> is true with <paramref name="value"/>.
    /// Used for causal attention masking — fills future positions with -inf before softmax.
    /// </summary>
    public static void MaskedFill(Tensor<float> a, ReadOnlySpan<bool> mask, float value)
    {
        if (mask.Length != a.ElementCount)
            throw new ArgumentException(
                $"Mask length {mask.Length} must match element count {a.ElementCount}.");
        var data = a.Data;
        for (int i = 0; i < data.Length; i++)
            if (mask[i]) data[i] = value;
    }

    /// <summary>Sum of all elements.</summary>
    public static T Sum<T>(Tensor<T> a)
        where T : unmanaged, INumber<T>
    {
        var data   = a.Data;
        int vecLen = Vector<T>.Count;
        var acc    = Vector<T>.Zero;
        int i      = 0;
        for (; i <= data.Length - vecLen; i += vecLen)
            acc += new Vector<T>(data[i..]);
        T sum = T.Zero;
        for (int lane = 0; lane < vecLen; lane++) sum += acc[lane];
        for (; i < data.Length; i++) sum += data[i];
        return sum;
    }

    /// <summary>Mean of all elements.</summary>
    public static T Mean<T>(Tensor<T> a)
        where T : unmanaged, System.Numerics.INumber<T>
        => Sum(a) / T.CreateChecked(a.ElementCount);
    /// <summary>Variance of all elements (population variance).</summary>
    public static float Variance(Tensor<float> a)
    {
        float mu = float.CreateChecked(Mean(a));
        var src = a.Data;
        float ss = 0f;
        for (int i = 0; i < src.Length; i++) { float d = src[i] - mu; ss += d * d; }
        return ss / src.Length;
    }
    /// <summary>Index of the maximum element.</summary>
    public static int ArgMax<T>(Tensor<T> a)
        where T : unmanaged, INumber<T>
    {
        var   data = a.Data;
        T     max  = data[0];
        int   idx  = 0;
        for (int i = 1; i < data.Length; i++)
            if (data[i] > max) { max = data[i]; idx = i; }
        return idx;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Returns the flat indices of the top <paramref name="k"/> elements (sorted descending).
    /// Used for Top-K sampling and MoE expert routing.
    /// </summary>
    public static int[] ArgTopK(Tensor<float> a, int k)
    {
        if (k <= 0 || k > a.ElementCount)
            throw new ArgumentOutOfRangeException(nameof(k), $"k={k} must be in [1, {a.ElementCount}].");

        int n = a.ElementCount;
        ReadOnlySpan<float> data = a.Data;

        if (k >= n)
        {
            var indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            float[] dataArr = a.Data.ToArray();
            Array.Sort(indices, (x, y) => dataArr[y].CompareTo(dataArr[x]));
            var result = new int[k];
            Array.Copy(indices, result, k);
            return result;
        }

        if (k <= 64)
        {
            var pq = new PriorityQueue<int, float>();
            for (int i = 0; i < n; i++)
            {
                float val = data[i];
                if (pq.Count < k)
                {
                    pq.Enqueue(i, val);
                }
                else if (pq.TryPeek(out _, out float minPriority) && val > minPriority)
                {
                    pq.Dequeue();
                    pq.Enqueue(i, val);
                }
            }
            var result = new int[k];
            for (int i = k - 1; i >= 0; i--) result[i] = pq.Dequeue();
            return result;
        }

        return ArgTopKIntroselectArray(data, k);
    }

    private static int[] ArgTopKIntroselectArray(ReadOnlySpan<float> data, int k)
    {
        int n = data.Length;
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        int left = 0, right = n - 1;
        int target = k - 1;

        while (left < right)
        {
            int pivot = PartitionArray(data, indices, left, right);
            if (pivot == target) break;
            if (pivot > target) right = pivot - 1;
            else left = pivot + 1;
        }

        var result = new int[k];
        for (int i = 0; i < k; i++) result[i] = indices[i];
        float[] dataArr = data.ToArray();
        Array.Sort(result, (x, y) => dataArr[y].CompareTo(dataArr[x]));
        return result;
    }

    private static int PartitionArray(ReadOnlySpan<float> data, int[] indices, int left, int right)
    {
        float pivot = data[indices[left]];
        int i = left;
        for (int j = left + 1; j <= right; j++)
        {
            if (data[indices[j]] > pivot)
            {
                i++;
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
        (indices[i], indices[left]) = (indices[left], indices[i]);
        return i;
    }

    /// <summary>Transposes a 2D matrix [R, C] → [C, R]. Allocates a new tensor.</summary>
    public static Tensor<float> Transpose(Tensor<float> src)
    {
        if (src.Rank != 2)
            throw new ArgumentException($"Transpose requires rank-2 tensor, got rank {src.Rank}.");
        return TransposeInternal(src);
    }

    public static void TransposeInPlace(Tensor<float> src)
    {
        if (src.Rank != 2)
            throw new ArgumentException($"Transpose requires rank-2 tensor, got rank {src.Rank}.");
        int R = src.Shape.Rows, C = src.Shape.Cols;
        if (R != C) throw new ArgumentException($"In-place transpose requires square matrix [{R},{C}].");
        var data = src.Data;
        for (int r = 0; r < R; r++)
        {
            for (int c = r + 1; c < C; c++)
            {
                int i = r * C + c;
                int j = c * R + r;
                (data[i], data[j]) = (data[j], data[i]);
            }
        }
    }
    // ═══════════════════════════════════════════════════════════════════════
    // Private helpers
    // ═══════════════════════════════════════════════════════════════════════

    private static unsafe Tensor<float> TransposeInternal(Tensor<float> src)
    {
        int R = src.Shape.Rows, C = src.Shape.Cols;
        var dst = new Tensor<float>(C, R);
        float* pS = src.DataPtr, pD = dst.DataPtr;

        if (R * C < 4096)
        {
            for (int r = 0; r < R; r++)
                for (int c = 0; c < C; c++)
                    pD[(long)c * R + r] = pS[(long)r * C + c];
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, R, r =>
            {
                for (int c = 0; c < C; c++)
                    pD[(long)c * R + r] = pS[(long)r * C + c];
            });
        }
        return dst;
    }
    /// <summary>
    /// Transposes the last two dimensions of a batched tensor.
    /// Used to prepare B for batched matmul — turns [*, K, N] into [*, N, K].
    /// </summary>
    private static unsafe Tensor<float> TransposeLast2D(Tensor<float> src, int M, int N, int batch)
    {
        var dst = new Tensor<float>(src.Shape.Reshape(batch, N, M));
        float* pS = src.DataPtr, pD = dst.DataPtr;
        for (int b = 0; b < batch; b++)
        {
            float* sSlice = pS + (long)b * M * N;
            float* dSlice = pD + (long)b * N * M;
            for (int r = 0; r < M; r++)
                for (int c = 0; c < N; c++)
                    dSlice[(long)c * M + r] = sSlice[(long)r * N + c];
        }
        return dst;
    }

    private static void AssertBatchedMatMulCompatible(TensorShape a, TensorShape b)
    {
        if (a.Rank < 3 || b.Rank < 3)
            throw new ArgumentException(
                $"BatchedMatMul requires rank ≥ 3, got {a} and {b}.");
        if (a.Rank != b.Rank)
            throw new ArgumentException(
                $"BatchedMatMul requires equal ranks, got {a} and {b}.");
        int rank = a.Rank;
        if (a[rank - 1] != b[rank - 2])
            throw new ArgumentException(
                $"BatchedMatMul inner dim mismatch: {a} × {b} " +
                $"(K={a[rank - 1]} ≠ {b[rank - 2]}).");
        for (int i = 0; i < rank - 2; i++)
            if (a[i] != b[i])
                throw new ArgumentException(
                    $"BatchedMatMul batch dim {i} mismatch: {a[i]} ≠ {b[i]}.");
    }

    private static void AssertMatMulWithBTCompatible(TensorShape a, TensorShape bt)
    {
        if (a.Rank != 2 || bt.Rank != 2)
            throw new ArgumentException($"MatMulWithBT requires rank-2 tensors, got {a} and {bt}.");
        int M = a.Rows;
        int K = a.Cols;
        int N = bt.Rows;
        int K2 = bt.Cols;
        if (K != K2)
            throw new ArgumentException($"MatMulWithBT inner dim mismatch: A is [{M},{K}] but BT is [{N},{K2}].");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BinaryOp<T, TOp>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> dst)
        where T : unmanaged, INumber<T>
        where TOp : struct, IBinaryOp<T>
    {
        int v = Vector<T>.Count, i = 0;
        for (; i <= dst.Length - v; i += v)
            TOp.Invoke(new Vector<T>(a[i..]), new Vector<T>(b[i..])).CopyTo(dst[i..]);
        for (; i < dst.Length; i++)
            dst[i] = TOp.InvokeScalar(a[i], b[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleVectorized<T>(ReadOnlySpan<T> src, Span<T> dst, T scalar)
        where T : unmanaged, INumber<T>
    {
        var vs = new Vector<T>(scalar);
        int v = Vector<T>.Count, i = 0;
        for (; i <= dst.Length - v; i += v)
            (new Vector<T>(src[i..]) * vs).CopyTo(dst[i..]);
        for (; i < dst.Length; i++)
            dst[i] = src[i] * scalar;
    }

}
