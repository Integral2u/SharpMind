using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Memory;

namespace SharpMind.Core.Tensors;

/// <summary>
/// A multi-dimensional, SIMD-aligned array of <typeparamref name="T"/>.
/// </summary>
/// <remarks>
/// <para>
/// Memory is backed by a reference-counted <see cref="NativeBuffer{T}"/>,
/// which means <see cref="Reshape"/> and row slices return lightweight
/// <em>views</em> that share the same allocation — no copying.
/// </para>
/// <para>
/// <typeparamref name="T"/> must be both <c>unmanaged</c> (for unsafe /
/// SIMD access) and implement <see cref="INumber{T}"/> (for generic math).
/// Typical use-cases: <c>float</c>, <c>Half</c>, <c>double</c>, <c>int</c>.
/// </para>
/// <para>
/// Always call <see cref="Dispose"/> (or use a <c>using</c> statement) when
/// done. Views hold their own reference and are safe to dispose independently.
/// </para>
/// </remarks>
public sealed unsafe class Tensor<T> : IDisposable
    where T : unmanaged, INumber<T>
{
    // fields

    private readonly NativeBuffer<T>? _buffer;
    private readonly int _offset; // element offset into buffer (for views)
    private readonly bool _ownsMemory;
    private readonly T* _rawPtr;


    // constructors
    
    /// <summary>Allocates a new zero-initialised tensor of the given shape.</summary>
    public Tensor(TensorShape shape) : this(shape, NativeBufferPool<T>.Rent(shape.ElementCount), 0, true) { }

    /// <inheritdoc cref="Tensor(TensorShape)"/>
    public Tensor(params int[] dims) : this(new TensorShape(dims)) { }
    public Tensor(int d0) : this(new TensorShape(d0)) { }
    public Tensor(int d0, int d1) : this(new TensorShape(d0, d1)) { }
    public Tensor(int d0, int d1, int d2) : this(new TensorShape(d0, d1, d2)) { }
    public Tensor(int d0, int d1, int d2, int d3) : this(new TensorShape(d0, d1, d2, d3)) { }

    /// <summary>
    /// Creates a tensor from a raw pointer.
    /// </summary>
    internal Tensor(T* ptr, TensorShape shape, bool ownsMemory = false)
    {
        Shape = shape;
        _ownsMemory = ownsMemory;
        _offset = 0;
        _rawPtr = ptr;
        _buffer = null;
    }

    /// <summary>
    /// Creates a tensor backed by an existing buffer.
    /// When <paramref name="ownsMemory"/> is true (owner created via Rent),
    /// AddRef is NOT called because Rent already set refcount=1.
    /// When false (view), AddRef is called to pin the buffer independently.
    /// </summary>
    internal Tensor(TensorShape shape, NativeBuffer<T> buffer, int offset = 0, bool ownsMemory = false)
    {
        Shape   = shape;
        _buffer = buffer;
        if (!ownsMemory && _buffer != null)
            _buffer.AddRef();
        _offset = offset;
        _ownsMemory = ownsMemory;
        _rawPtr = _buffer != null ? _buffer.Ptr : null;
    }

    // properties

    public TensorShape Shape        { get; }
    public int         Rank         => Shape.Rank;
    public int         ElementCount => Shape.ElementCount;    
    /// <summary>
    /// Raw pointer to element 0 of this tensor (may be an offset view).
    /// Valid only while the tensor is alive.
    /// </summary>
    public T* DataPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _rawPtr;
    }

    /// <summary>Span over all elements in flat (row-major) order.</summary>
    public Span<T> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get 
        {
            if (_buffer != null) return _buffer.AsSpan(_offset, ElementCount);
            return new Span<T>(_rawPtr + _offset, ElementCount);
        }
    }


    // indexers

    /// <summary>Flat element access.</summary>
    public ref T this[int flatIndex]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref DataPtr[flatIndex];
    }

    /// <summary>2-D element access (row, col).</summary>
    public ref T this[int row, int col]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref DataPtr[Shape.GetOffset(row, col)];
    }

    /// <summary>N-D element access via index array.</summary>
    public ref T this[params int[] indices]
    {
        get => ref DataPtr[Shape.GetOffset(indices)];
    }

    // factory methods

    /// <summary>Allocates a zero-filled tensor.</summary>
    public static Tensor<T> Zeros(params int[] dims) => new(dims);
    public static Tensor<T> Zeros(int d0) => new(d0);
    public static Tensor<T> Zeros(int d0, int d1) => new(d0, d1);
    public static Tensor<T> Zeros(int d0, int d1, int d2) => new(d0, d1, d2);
    public static Tensor<T> Zeros(int d0, int d1, int d2, int d3) => new(d0, d1, d2, d3);

    /// <summary>Allocates a tensor filled with ones.</summary>
    public static Tensor<T> Ones(params int[] dims)
    {
        var t = new Tensor<T>(dims);
        t.Data.Fill(T.One);
        return t;
    }

    /// <summary>Wraps existing data in a tensor (copies into native memory).</summary>
    public static Tensor<T> From(ReadOnlySpan<T> data, params int[] dims)
    {
        var t = new Tensor<T>(dims);
        if (data.Length != t.ElementCount)
            throw new ArgumentException(
                $"Data length {data.Length} != shape element count {t.ElementCount}.");
        data.CopyTo(t.Data);
        return t;
    }

    /// <summary>Creates an identity matrix of size <paramref name="n"/>.</summary>
    public static Tensor<T> Eye(int n)
    {
        var t = new Tensor<T>(n, n);
        for (int i = 0; i < n; i++)
            t[i, i] = T.One;
        return t;
    }

    // mutation

    public void Fill(T value) => Data.Fill(value);

    public void CopyFrom(ReadOnlySpan<T> src)
    {
        if (src.Length != ElementCount)
            throw new ArgumentException("Source length must match element count.");
        src.CopyTo(Data);
    }

    public void CopyTo(Span<T> dst) => Data.CopyTo(dst);

    private Tensor<T> CreateView(TensorShape shape, int offset, bool _)
    {
        // Views never own memory — the parent manages the buffer lifecycle.
        // This prevents premature Return-to-pool when a view is disposed.
        if (_buffer != null)
            return new Tensor<T>(shape, _buffer, offset, false);
        
        return new Tensor<T>(_rawPtr + offset, shape, false);
    }

    /// <summary>
    /// Returns a new tensor with the same data but a different shape.
    /// Zero-copy: both tensors share the same buffer.
    /// </summary>
    public Tensor<T> Reshape(params int[] newDims) => CreateView(Shape.Reshape(newDims), _offset, _ownsMemory);
    public Tensor<T> Reshape(int d0) => CreateView(Shape.Reshape(d0), _offset, _ownsMemory);
    public Tensor<T> Reshape(int d0, int d1) => CreateView(Shape.Reshape(d0, d1), _offset, _ownsMemory);
    public Tensor<T> Reshape(int d0, int d1, int d2) => CreateView(Shape.Reshape(d0, d1, d2), _offset, _ownsMemory);
    public Tensor<T> Reshape(int d0, int d1, int d2, int d3) => CreateView(Shape.Reshape(d0, d1, d2, d3), _offset, _ownsMemory);


    /// <summary>Row as a <see cref="Span{T}"/></summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> RowSpan(int i) => Data.Slice(i * Shape.Cols, Shape.Cols);
    
    /// <summary>
    /// Returns a view into dimensions [dim1, dim2, ...] starting at the given indices.
    /// For example, given [B,S,H,D], Slice(0,startPos,0,0) returns [S,H,D].
    /// </summary>
    public Tensor<T> Slice(params int[] startIndices)
    {
        if (startIndices.Length >= Rank)
            throw new ArgumentException($"Slice requires at most {Rank} indices.");
        
        int offset = 0;
        for (int i = 0; i < startIndices.Length; i++)
        {
            if ((uint)startIndices[i] >= (uint)Shape.Dims[i])
                throw new ArgumentOutOfRangeException(nameof(startIndices),
                    $"Index {startIndices[i]} is out of range for dimension {i} (size {Shape.Dims[i]}).");
            offset += startIndices[i] * Shape.Strides[i];
        }
        
        int[] newDimsArray = new int[Shape.Rank - startIndices.Length];
        for (int i = 0; i < newDimsArray.Length; i++)
        {
            newDimsArray[i] = Shape.Dims[startIndices.Length + i];
        }
        
        return CreateView(new TensorShape(newDimsArray), _offset + offset, _ownsMemory);
    }

    // diagnostics

    public override string ToString() =>
        $"Tensor<{typeof(T).Name}> {Shape} [{ElementCount} elements]";

    /// <summary>
    /// Formats the tensor contents for debugging (small tensors only).
    /// </summary>
    public string ToDebugString(int maxElements = 32)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"Tensor<{typeof(T).Name}> {Shape}: [");
        int n = Math.Min(ElementCount, maxElements);
        for (int i = 0; i < n; i++)
        {
            sb.Append(this[i]);
            if (i < n - 1) sb.Append(", ");
        }
        if (ElementCount > maxElements) sb.Append(", ...");
        sb.Append(']');
        return sb.ToString();
    }

    // ── Elementwise operator helpers ──────────────────────────────────────

    private interface IBinaryOp<U> where U : unmanaged, INumber<U>
    {
        static abstract Vector<U> Invoke(Vector<U> a, Vector<U> b);
        static abstract U InvokeScalar(U a, U b);
    }

    private readonly struct AddOp<U> : IBinaryOp<U> where U : unmanaged, INumber<U>
    {
        public static Vector<U> Invoke(Vector<U> a, Vector<U> b) => a + b;
        public static U InvokeScalar(U a, U b) => a + b;
    }

    private readonly struct SubtractOp<U> : IBinaryOp<U> where U : unmanaged, INumber<U>
    {
        public static Vector<U> Invoke(Vector<U> a, Vector<U> b) => a - b;
        public static U InvokeScalar(U a, U b) => a - b;
    }

    private readonly struct MultiplyOp<U> : IBinaryOp<U> where U : unmanaged, INumber<U>
    {
        public static Vector<U> Invoke(Vector<U> a, Vector<U> b) => a * b;
        public static U InvokeScalar(U a, U b) => a * b;
    }

    private readonly struct DivideOp<U> : IBinaryOp<U> where U : unmanaged, INumber<U>
    {
        public static Vector<U> Invoke(Vector<U> a, Vector<U> b) => a / b;
        public static U InvokeScalar(U a, U b) => a / b;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BinaryOp<U, TOp>(ReadOnlySpan<U> a, ReadOnlySpan<U> b, Span<U> dst)
        where U : unmanaged, INumber<U>
        where TOp : struct, IBinaryOp<U>
    {
        int v = Vector<U>.Count, i = 0;
        for (; i <= dst.Length - v; i += v)
            TOp.Invoke(new Vector<U>(a[i..]), new Vector<U>(b[i..])).CopyTo(dst[i..]);
        for (; i < dst.Length; i++)
            dst[i] = TOp.InvokeScalar(a[i], b[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ScaleVectorized<U>(ReadOnlySpan<U> src, Span<U> dst, U scalar)
        where U : unmanaged, INumber<U>
    {
        var vs = new Vector<U>(scalar);
        int v = Vector<U>.Count, i = 0;
        for (; i <= dst.Length - v; i += v)
            (new Vector<U>(src[i..]) * vs).CopyTo(dst[i..]);
        for (; i < dst.Length; i++)
            dst[i] = src[i] * scalar;
    }

    private static Tensor<float> TransposeInternal(Tensor<float> src)
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
            Parallel.For(0, R, r =>
            {
                for (int c = 0; c < C; c++)
                    pD[(long)c * R + r] = pS[(long)r * C + c];
            });
        }
        return dst;
    }

    private static Tensor<float> TransposeLast2D(Tensor<float> src, int M, int N, int batch)
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

    // ── Instance elementwise methods ──────────────────────────────────────

    public Tensor<T> Add(Tensor<T> other)
    {
        TensorShape.AssertSameShape(Shape, other.Shape);
        var r = new Tensor<T>(Shape);
        BinaryOp<T, AddOp<T>>(Data, other.Data, r.Data);
        return r;
    }

    public void AddInPlace(Tensor<T> other)
    {
        TensorShape.AssertSameShape(Shape, other.Shape);
        BinaryOp<T, AddOp<T>>(Data, other.Data, Data);
    }

    public Tensor<T> Subtract(Tensor<T> other)
    {
        TensorShape.AssertSameShape(Shape, other.Shape);
        var r = new Tensor<T>(Shape);
        BinaryOp<T, SubtractOp<T>>(Data, other.Data, r.Data);
        return r;
    }

    public Tensor<T> Multiply(Tensor<T> other)
    {
        TensorShape.AssertSameShape(Shape, other.Shape);
        var r = new Tensor<T>(Shape);
        BinaryOp<T, MultiplyOp<T>>(Data, other.Data, r.Data);
        return r;
    }

    public Tensor<T> Divide(Tensor<T> other)
    {
        TensorShape.AssertSameShape(Shape, other.Shape);
        var r = new Tensor<T>(Shape);
        BinaryOp<T, DivideOp<T>>(Data, other.Data, r.Data);
        return r;
    }

    public Tensor<T> Scale(T scalar)
    {
        var r = new Tensor<T>(Shape);
        ScaleVectorized(Data, r.Data, scalar);
        return r;
    }

    public void ScaleInPlace(T scalar)
    {
        ScaleVectorized(Data, Data, scalar);
    }

    public Tensor<T> Clamp(T min, T max)
    {
        var r = new Tensor<T>(Shape);
        var src = Data;
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

    public Tensor<float> Sqrt()
    {
        if (typeof(T) != typeof(float))
            throw new InvalidOperationException("Sqrt is only supported for Tensor<float>.");
        var tf = Unsafe.As<Tensor<float>>(this);
        var src = tf.Data;
        var r = new Tensor<float>(Shape);
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

    public Tensor<T> Abs()
    {
        var r = new Tensor<T>(Shape);
        var src = Data;
        var dst = r.Data;
        int vecLen = Vector<T>.Count;
        int i = 0;
        for (; i <= src.Length - vecLen; i += vecLen)
            Vector.Abs(new Vector<T>(src[i..])).CopyTo(dst[i..]);
        for (; i < src.Length; i++)
            dst[i] = T.Abs(src[i]);
        return r;
    }

    public void MaskedFill(ReadOnlySpan<bool> mask, T value)
    {
        if (mask.Length != ElementCount)
            throw new ArgumentException($"Mask length {mask.Length} must match element count {ElementCount}.");
        var data = Data;
        for (int i = 0; i < data.Length; i++)
            if (mask[i]) data[i] = value;
    }

    public T Sum()
    {
        var data = Data;
        int vecLen = Vector<T>.Count;
        var acc = Vector<T>.Zero;
        int i = 0;
        for (; i <= data.Length - vecLen; i += vecLen)
            acc += new Vector<T>(data[i..]);
        T sum = T.Zero;
        for (int lane = 0; lane < vecLen; lane++) sum += acc[lane];
        for (; i < data.Length; i++) sum += data[i];
        return sum;
    }

    public T Mean() => Sum() / T.CreateChecked(ElementCount);

    public float Variance()
    {
        float mu = float.CreateChecked(Mean());
        var src = Data;
        float ss = 0f;
        for (int i = 0; i < src.Length; i++) { float d = float.CreateChecked(src[i]) - mu; ss += d * d; }
        return ss / src.Length;
    }

    public int ArgMax()
    {
        var data = Data;
        T max = data[0];
        int idx = 0;
        for (int i = 1; i < data.Length; i++)
            if (data[i].CompareTo(max) > 0) { max = data[i]; idx = i; }
        return idx;
    }

    public Tensor<float> Transpose()
    {
        if (typeof(T) != typeof(float))
            throw new InvalidOperationException("Transpose is only supported for Tensor<float>.");
        if (Rank != 2)
            throw new ArgumentException($"Transpose requires rank-2 tensor, got rank {Rank}.");
        return TransposeInternal(Unsafe.As<Tensor<float>>(this));
    }

    public void TransposeInPlace()
    {
        if (typeof(T) != typeof(float))
            throw new InvalidOperationException("Transpose is only supported for Tensor<float>.");
        if (Rank != 2)
            throw new ArgumentException($"Transpose requires rank-2 tensor, got rank {Rank}.");
        var tf = Unsafe.As<Tensor<float>>(this);
        int R = tf.Shape.Rows, C = tf.Shape.Cols;
        if (R != C) throw new ArgumentException($"In-place transpose requires square matrix [{R},{C}].");
        var data = tf.Data;
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

    // disposal

    /// <summary>
    /// Releases this tensor's reference to the backing buffer.
    /// The native memory is freed only when all views are disposed.
    /// </summary>
    public void Dispose()
    {
        _buffer?.Dispose();
    }
}
