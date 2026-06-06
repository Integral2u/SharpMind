using System.Numerics;
using System.Runtime.CompilerServices;
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
    // ── fields ─────────────────────────────────────────────────────────────

    private readonly NativeBuffer<T>? _buffer;
    private readonly int _offset; // element offset into buffer (for views)
    private readonly bool _ownsMemory;
    private readonly T* _rawPtr;


    // ── constructors ───────────────────────────────────────────────────────
    
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
    /// Creates a view into an existing buffer (increments ref-count).
    /// The view is independently disposable.
    /// </summary>
    internal Tensor(TensorShape shape, NativeBuffer<T> buffer, int offset = 0, bool ownsMemory = false)
    {
        Shape   = shape;
        _buffer = buffer;
        if (ownsMemory || _buffer != null)
            _buffer?.AddRef();
        _offset = offset;
        _ownsMemory = ownsMemory;
        _rawPtr = _buffer != null ? _buffer.Ptr + offset : null;
    }

    // ── properties ─────────────────────────────────────────────────────────

    public TensorShape Shape        { get; }
    public int         Rank         => Shape.Rank;
    public int         ElementCount => Shape.ElementCount;    
    /// <summary>
    /// Raw pointer to element 0 of this tensor (may be an offset view).
    /// Valid only while the tensor is alive.
    /// </summary>
    public unsafe T* DataPtr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _rawPtr + _offset;
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

    // ... [rest of the file]


    // ── indexers ───────────────────────────────────────────────────────────

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

    // ── factory methods ────────────────────────────────────────────────────

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

    // ── mutation ───────────────────────────────────────────────────────────

    public void Fill(T value) => Data.Fill(value);

    public void CopyFrom(ReadOnlySpan<T> src)
    {
        if (src.Length != ElementCount)
            throw new ArgumentException("Source length must match element count.");
        src.CopyTo(Data);
    }

    public void CopyTo(Span<T> dst) => Data.CopyTo(dst);

    private Tensor<T> CreateView(TensorShape shape, int offset, bool ownsMemory)
    {
        if (_buffer != null)
            return new Tensor<T>(shape, _buffer, offset, ownsMemory);
        
        return new Tensor<T>(_rawPtr + offset, shape, ownsMemory);
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


    /// <summary>
    /// Returns a 1-D view of row <paramref name="i"/> for a 2-D tensor (zero-copy).
    /// </summary>
    public Tensor<T> RowView(int i)
    {
        if (Rank != 2) throw new InvalidOperationException("RowView requires a 2-D tensor.");
        int cols = Shape.Cols;
        return CreateView(new TensorShape(cols), _offset + i * cols, _ownsMemory);
    }


    /// <summary>Row as a <see cref="Span{T}"/> — slightly cheaper than <see cref="RowView"/>.</summary>
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
            offset += startIndices[i] * Shape.Strides[i];
        }
        
        int[] newDimsArray = new int[Shape.Rank - startIndices.Length];
        for (int i = 0; i < newDimsArray.Length; i++)
        {
            newDimsArray[i] = Shape.Dims[startIndices.Length + i];
        }
        
        return CreateView(new TensorShape(newDimsArray), _offset + offset, _ownsMemory);
    }

    // ── diagnostics ────────────────────────────────────────────────────────

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

    // ── disposal ───────────────────────────────────────────────────────────

    /// <summary>
    /// Releases this tensor's reference to the backing buffer.
    /// The native memory is freed only when all views are disposed.
    /// </summary>
    public void Dispose()
    {
        if (_ownsMemory && _buffer != null)
            _buffer.Dispose();
    }
}
