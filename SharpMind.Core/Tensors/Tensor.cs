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

    private readonly NativeBuffer<T> _buffer;
    private readonly int _offset; // element offset into buffer (for views)

    // ── constructors ───────────────────────────────────────────────────────

    /// <summary>Allocates a new zero-initialised tensor of the given shape.</summary>
    public Tensor(TensorShape shape)
    {
        Shape   = shape;
        _buffer = NativeBufferPool<T>.Rent(shape.ElementCount);
        _offset = 0;
    }

    /// <inheritdoc cref="Tensor(TensorShape)"/>
    public Tensor(params int[] dims) : this(new TensorShape(dims)) { }

    /// <summary>
    /// Creates a view into an existing buffer (increments ref-count).
    /// The view is independently disposable.
    /// </summary>
    internal Tensor(TensorShape shape, NativeBuffer<T> buffer, int offset = 0)
    {
        Shape   = shape;
        _buffer = buffer;
        _buffer.AddRef();
        _offset = offset;
    }

    // ── properties ─────────────────────────────────────────────────────────

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
        get => _buffer.Ptr + _offset;
    }

    /// <summary>Span over all elements in flat (row-major) order.</summary>
    public Span<T> Data
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _buffer.AsSpan(_offset, ElementCount);
    }

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

    // ── view operations (zero-copy) ────────────────────────────────────────

    /// <summary>
    /// Returns a new tensor with the same data but a different shape.
    /// Zero-copy: both tensors share the same buffer.
    /// </summary>
    public Tensor<T> Reshape(params int[] newDims)
    {
        var newShape = Shape.Reshape(newDims);
        return new Tensor<T>(newShape, _buffer, _offset);
    }

    /// <summary>
    /// Returns a 1-D view of row <paramref name="i"/> for a 2-D tensor (zero-copy).
    /// </summary>
    public Tensor<T> RowView(int i)
    {
        if (Rank != 2) throw new InvalidOperationException("RowView requires a 2-D tensor.");
        int cols = Shape.Cols;
        return new Tensor<T>(new TensorShape(cols), _buffer, _offset + i * cols);
    }

    /// <summary>Row as a <see cref="Span{T}"/> — slightly cheaper than <see cref="RowView"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> RowSpan(int i) => Data.Slice(i * Shape.Cols, Shape.Cols);

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
        _buffer.Dispose();
    }
}
