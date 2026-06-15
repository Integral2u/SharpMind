using SharpMind.Core.Ops;
using System.Runtime.CompilerServices;
using System.Text;

namespace SharpMind.Core.Tensors;

/// <summary>
/// Immutable descriptor of a tensor's shape and strides. Row-major (C) order.
/// </summary>
/// <remarks>
/// Strides are element counts, not byte offsets. For a shape (3, 4, 5):
///   strides = (20, 5, 1) so element [i,j,k] is at flat index i*20 + j*5 + k.
/// </remarks>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    private readonly int[] _dims;
    private readonly int[] _strides;

    // construction

    /// <summary>Core init with span — no <c>params</c> heap allocation at call site.</summary>
    public TensorShape(ReadOnlySpan<int> dims)
    {
        if (dims.Length == 0)
            throw new ArgumentException($"A {nameof(TensorShape)} must have at least one dimension.", nameof(dims));
        foreach (int d in dims)
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(dims), $"Dimension must be > 0, got {d}.");
        _dims    = dims.ToArray();
        _strides = ComputeStrides(_dims);
        ElementCount = ComputeElementCount(_dims);
    }

    public TensorShape(params int[] dims)
        : this((ReadOnlySpan<int>)(dims ?? throw new ArgumentNullException(nameof(dims)))) { }

    public TensorShape(int d0) : this([d0]) { }
    public TensorShape(int d0, int d1) : this([d0, d1]) { }
    public TensorShape(int d0, int d1, int d2) : this([d0, d1, d2]) { }
    public TensorShape(int d0, int d1, int d2, int d3) : this([d0, d1, d2, d3]) { }

    // properties

    public int Rank         => _dims.Length;
    public int ElementCount { get; }

    /// <summary>Dimension sizes (read-only view).</summary>
    public ReadOnlySpan<int> Dims    => _dims;

    /// <summary>Element strides in row-major order (read-only view).</summary>
    public ReadOnlySpan<int> Strides => _strides;

    /// <summary>Supports negative indexing: shape[-1] is the last dim.</summary>
    public int this[int dim]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _dims[dim < 0 ? _dims.Length + dim : dim];
    }
    public int Length => _dims.Length; // Required for implicit index support
    // Convenience aliases for 2-D tensors
    public int Rows => _dims[^2];
    public int Cols => _dims[^1];

    public bool IsScalar => Rank == 1 && _dims[0] == 1;
    public bool IsVector  => Rank == 1;
    public bool IsMatrix  => Rank == 2;

    // flat offset

    /// <summary>Converts a multi-index to a flat buffer offset.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetOffset(ReadOnlySpan<int> indices)
    {
        if (indices.Length != Rank)
            throw new ArgumentException($"Expected {Rank} indices, got {indices.Length}.");
        int offset = 0;
        for (int i = 0; i < Rank; i++)
            offset += indices[i] * _strides[i];
        return offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetOffset(int row, int col) => row * _strides[^2] + col;

    // shape algebra

    /// <summary>
    /// Returns a new shape with the same element count but different dims.
    /// Pass -1 for exactly one dim to infer it automatically.
    /// </summary>
    public TensorShape Reshape(params int[] newDims) => Reshape((ReadOnlySpan<int>)newDims);

    public TensorShape Reshape(ReadOnlySpan<int> newDims)
    {
        Span<int> dims = newDims.Length <= 4 ? stackalloc int[newDims.Length] : new int[newDims.Length];
        newDims.CopyTo(dims);

        int inferred = -1;
        long known = 1;
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] == -1)
            {
                if (inferred >= 0)
                    throw new ArgumentException("Only one dimension can be inferred (-1).");
                inferred = i;
            }
            else known *= dims[i];
        }

        if (inferred >= 0)
            dims[inferred] = (int)(ElementCount / known);

        int count = ComputeElementCount(dims);
        if (count != ElementCount)
            throw new ArgumentException(
                $"Cannot reshape {this} ({ElementCount} elements) into ({string.Join(", ", dims.ToArray())}) ({count} elements).");
        return new TensorShape(dims.ToArray());
    }

    public TensorShape Reshape(int d0) => Reshape([d0]);
    public TensorShape Reshape(int d0, int d1) => Reshape([d0, d1]);
    public TensorShape Reshape(int d0, int d1, int d2) => Reshape([d0, d1, d2]);
    public TensorShape Reshape(int d0, int d1, int d2, int d3) => Reshape([d0, d1, d2, d3]);

    /// <summary>Adds a size-1 dimension at <paramref name="axis"/>.</summary>
    public TensorShape Unsqueeze(int axis)
    {
        if (axis < 0) axis += Rank + 1;
        var newDims = new int[Rank + 1];
        _dims.AsSpan(0, axis).CopyTo(newDims);
        newDims[axis] = 1;
        _dims.AsSpan(axis).CopyTo(newDims.AsSpan(axis + 1));
        return new TensorShape(newDims);
    }

    /// <summary>Removes a size-1 dimension at <paramref name="axis"/>.</summary>
    public TensorShape Squeeze(int axis)
    {
        if (axis < 0) axis += Rank;
        if (_dims[axis] != 1)
            throw new InvalidOperationException($"Cannot squeeze dim {axis} of size {_dims[axis]}.");
        var newDims = new int[Rank - 1];
        _dims.AsSpan(0, axis).CopyTo(newDims);
        _dims.AsSpan(axis + 1).CopyTo(newDims.AsSpan(axis));
        return new TensorShape(newDims);
    }

    // validation helpers

    public static void AssertSameShape(TensorShape a, TensorShape b)
    {
        if (!a.Equals(b))
            throw new ArgumentException($"{(new System.Diagnostics.StackTrace()?.GetFrame(1)?.GetMethod()?.Name??"Operation")} requires identical shapes, got {a} and {b}.");
    }

    public static void AssertMatMulCompatible(TensorShape a, TensorShape b)
    {
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException($"{nameof(TensorOps.MatMul)} requires 2-D tensors, got {a} and {b}.");
        if (a.Cols != b.Rows)
            throw new ArgumentException(
                $"{nameof(TensorOps.MatMul)} shape mismatch: {a} · {b} (inner dims {a.Cols} ≠ {b.Rows}).");
    }

    // equality

    public bool Equals(TensorShape other) =>
        _dims is not null && other._dims is not null &&
        _dims.AsSpan().SequenceEqual(other._dims);

    public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);
    public override int GetHashCode()         => HashCode.Combine(Rank, ElementCount);

    public static bool operator ==(TensorShape a, TensorShape b) => a.Equals(b);
    public static bool operator !=(TensorShape a, TensorShape b) => !a.Equals(b);

    // display

    public override string ToString()
    {
        var sb = new StringBuilder("(");
        for (int i = 0; i < _dims.Length; i++)
        {
            sb.Append(_dims[i]);
            if (i < _dims.Length - 1) sb.Append(", ");
        }
        sb.Append(')');
        return sb.ToString();
    }

    // private helpers

    private static int[] ComputeStrides(int[] dims)
    {
        var s = new int[dims.Length];
        s[^1] = 1;
        for (int i = dims.Length - 2; i >= 0; i--)
            s[i] = s[i + 1] * dims[i + 1];
        return s;
    }

    private static int ComputeElementCount(ReadOnlySpan<int> dims)
    {
        int n = 1;
        foreach (int d in dims) n *= d;
        return n;
    }

    private static int ComputeElementCount(int[] dims)
        => ComputeElementCount((ReadOnlySpan<int>)dims);
}
