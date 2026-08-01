using System.Runtime.CompilerServices;
using System.Text;
using SharpMind.Core.Diagnostics;

namespace SharpMind.Core.Tensors;

/// <summary>
/// Immutable descriptor of a tensor's shape and strides. Row-major (C) order.
/// </summary>
/// <remarks>
/// For ranks 1–4 the dimension and stride data is stored inline (no heap allocation).
/// For ranks &gt; 4, two <c>int[]</c> arrays are allocated on the heap as before.
/// Strides are element counts, not byte offsets. For a shape (3, 4, 5):
///   strides = (20, 5, 1) so element [i,j,k] is at flat index i*20 + j*5 + 1.
/// </remarks>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    private readonly int _rank;
    private readonly int _elementCount;

    // Inline storage for ranks 1–4 (zero heap allocation).
    private readonly int _d0, _d1, _d2, _d3;
    private readonly int _s0, _s1, _s2, _s3;

    // Heap overflow for ranks > 4 (null when inline path is used).
    private readonly int[]? _dimsOverflow;
    private readonly int[]? _stridesOverflow;

    // ── construction ──────────────────────────────────────────────

    /// <summary>Core init with span — no <c>params</c> heap allocation at call site.</summary>
    public TensorShape(ReadOnlySpan<int> dims)
    {
        if (dims.Length == 0)
            throw new ArgumentException($"A {nameof(TensorShape)} must have at least one dimension.", nameof(dims));
        foreach (int d in dims)
            if (d <= 0) throw new ArgumentOutOfRangeException(nameof(dims), $"Dimension must be > 0, got {d}.");

        _rank = dims.Length;
        _elementCount = ComputeElementCount(dims);

        if (dims.Length <= 4)
        {
            // Inline path — no heap allocation.
            if (dims.Length >= 1) { _d0 = dims[0]; _s0 = dims.Length == 1 ? 1 : ComputeStride(dims, 0); }
            if (dims.Length >= 2) { _d1 = dims[1]; _s1 = dims.Length == 2 ? 1 : ComputeStride(dims, 1); }
            if (dims.Length >= 3) { _d2 = dims[2]; _s2 = dims.Length == 3 ? 1 : ComputeStride(dims, 2); }
            if (dims.Length >= 4) { _d3 = dims[3]; _s3 = 1; }
            _dimsOverflow = null;
            _stridesOverflow = null;
        }
        else
        {
            // Overflow path — heap arrays for rank > 4.
            _dimsOverflow = dims.ToArray();
            _stridesOverflow = ComputeStridesArray(_dimsOverflow);
        }
    }

    public TensorShape(params int[] dims)
        : this((ReadOnlySpan<int>)(dims ?? throw new ArgumentNullException(nameof(dims)))) { }

    public TensorShape(int d0)
    {
        if (d0 <= 0) throw new ArgumentOutOfRangeException(nameof(d0), $"Dimension must be > 0, got {d0}.");
        _rank = 1; _elementCount = d0;
        _d0 = d0; _s0 = 1;
    }

    public TensorShape(int d0, int d1)
    {
        if (d0 <= 0) throw new ArgumentOutOfRangeException(nameof(d0), $"Dimension must be > 0, got {d0}.");
        if (d1 <= 0) throw new ArgumentOutOfRangeException(nameof(d1), $"Dimension must be > 0, got {d1}.");
        long count = (long)d0 * d1;
        if (count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(d0), $"Element count {count} overflows int.");
        _rank = 2; _elementCount = (int)count;
        _d0 = d0; _d1 = d1;
        _s0 = d1; _s1 = 1;
    }

    public TensorShape(int d0, int d1, int d2)
    {
        if (d0 <= 0) throw new ArgumentOutOfRangeException(nameof(d0), $"Dimension must be > 0, got {d0}.");
        if (d1 <= 0) throw new ArgumentOutOfRangeException(nameof(d1), $"Dimension must be > 0, got {d1}.");
        if (d2 <= 0) throw new ArgumentOutOfRangeException(nameof(d2), $"Dimension must be > 0, got {d2}.");
        long count = (long)d0 * d1 * d2;
        if (count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(d0), $"Element count {count} overflows int.");
        _rank = 3; _elementCount = (int)count;
        _d0 = d0; _d1 = d1; _d2 = d2;
        long s0 = (long)d1 * d2; _s0 = (int)s0; _s1 = d2; _s2 = 1;
    }

    public TensorShape(int d0, int d1, int d2, int d3)
    {
        if (d0 <= 0) throw new ArgumentOutOfRangeException(nameof(d0), $"Dimension must be > 0, got {d0}.");
        if (d1 <= 0) throw new ArgumentOutOfRangeException(nameof(d1), $"Dimension must be > 0, got {d1}.");
        if (d2 <= 0) throw new ArgumentOutOfRangeException(nameof(d2), $"Dimension must be > 0, got {d2}.");
        if (d3 <= 0) throw new ArgumentOutOfRangeException(nameof(d3), $"Dimension must be > 0, got {d3}.");
        long count = (long)d0 * d1 * d2 * d3;
        if (count > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(d0), $"Element count {count} overflows int.");
        _rank = 4; _elementCount = (int)count;
        _d0 = d0; _d1 = d1; _d2 = d2; _d3 = d3;
        long s0 = (long)d1 * d2 * d3; _s0 = (int)s0;
        long s1 = (long)d2 * d3; _s1 = (int)s1;
        _s2 = d3; _s3 = 1;
    }

    // ── properties ────────────────────────────────────────────────

    public int Rank => _rank;
    public int ElementCount => _elementCount;

    /// <summary>
    /// Dimension sizes (read-only view). Allocates a small array on demand for inline ranks (1–4).
    /// Prefer using the indexer <c>shape[i]</c> in hot paths to avoid this allocation.
    /// </summary>
    public ReadOnlySpan<int> Dims => _dimsOverflow ?? CreateInlineDimsArray();

    /// <summary>
    /// Element strides in row-major order (read-only view). Allocates on demand for inline ranks.
    /// Prefer using <see cref="GetOffset(int, int)"/> in hot paths.
    /// </summary>
    public ReadOnlySpan<int> Strides => _stridesOverflow ?? CreateInlineStridesArray();

    /// <summary>Supports negative indexing: shape[-1] is the last dim.</summary>
    public int this[int dim]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            int i = dim < 0 ? _rank + dim : dim;
            return i switch
            {
                0 => _d0,
                1 => _d1,
                2 => _d2,
                3 => _d3,
                _ => _dimsOverflow![i]
            };
        }
    }

    public int Length => _rank;

    public int Rows
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _rank >= 2 ? this[^2] : throw new InvalidOperationException("Shape must have at least 2 dimensions.");
    }

    public int Cols
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => this[^1];
    }

    public bool IsScalar => _rank == 1 && _d0 == 1;
    public bool IsVector => _rank == 1;
    public bool IsMatrix => _rank == 2;

    // ── flat offset ───────────────────────────────────────────────

    /// <summary>Converts a multi-index to a flat buffer offset.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetOffset(ReadOnlySpan<int> indices)
    {
        if (indices.Length != _rank)
            throw new ArgumentException($"Expected {_rank} indices, got {indices.Length}.");
        int offset = 0;
        if (_stridesOverflow is not null)
        {
            for (int i = 0; i < _rank; i++)
            {
                int dim = _dimsOverflow![i];
                SanityChecks.IndexInRange(indices[i], dim, $"indices[{i}]");
                offset += indices[i] * _stridesOverflow[i];
            }
        }
        else
        {
            for (int i = 0; i < _rank; i++)
            {
                int dim = this[i];
                SanityChecks.IndexInRange(indices[i], dim, $"indices[{i}]");
                offset += indices[i] * GetStrideInline(i);
            }
        }
        return offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetOffset(int row, int col)
    {
        if (_stridesOverflow is not null)
        {
            int dimR = _dimsOverflow![^2];
            int dimC = _dimsOverflow[^1];
            SanityChecks.IndexInRange(row, dimR, nameof(row));
            SanityChecks.IndexInRange(col, dimC, nameof(col));
            return row * _stridesOverflow[^2] + col;
        }
        int dR = this[_rank - 2];
        int dC = this[_rank - 1];
        SanityChecks.IndexInRange(row, dR, nameof(row));
        SanityChecks.IndexInRange(col, dC, nameof(col));
        return row * GetStrideInline(_rank - 2) + col;
    }

    // ── shape algebra ─────────────────────────────────────────────

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
            dims[inferred] = (int)(_elementCount / known);

        int count = ComputeElementCount(dims);
        if (count != _elementCount)
            throw new ArgumentException(
                $"Cannot reshape {this} ({_elementCount} elements) into ({string.Join(", ", dims.ToArray())}) ({count} elements).");
        return new TensorShape(dims);
    }

    public TensorShape Reshape(int d0)
    {
        if (d0 == -1) d0 = _elementCount;
        if (d0 != _elementCount)
            throw new ArgumentException($"Cannot reshape {this} ({_elementCount} elements) into ({d0}).");
        return new TensorShape(d0);
    }

    public TensorShape Reshape(int d0, int d1) => Reshape((ReadOnlySpan<int>)[d0, d1]);
    public TensorShape Reshape(int d0, int d1, int d2) => Reshape((ReadOnlySpan<int>)[d0, d1, d2]);
    public TensorShape Reshape(int d0, int d1, int d2, int d3) => Reshape((ReadOnlySpan<int>)[d0, d1, d2, d3]);

    public TensorShape Unsqueeze(int axis)
    {
        if (axis < 0) axis += _rank + 1;

        Span<int> newDims = stackalloc int[_rank + 1];
        for (int i = 0; i < axis; i++)
            newDims[i] = this[i];
        newDims[axis] = 1;
        for (int i = axis; i < _rank; i++)
            newDims[i + 1] = this[i];
        return new TensorShape(newDims);
    }

    public TensorShape Squeeze(int axis)
    {
        if (axis < 0) axis += _rank;
        int dim = this[axis];
        if (dim != 1)
            throw new InvalidOperationException($"Cannot squeeze dim {axis} of size {dim}.");

        Span<int> newDims = stackalloc int[_rank - 1];
        for (int i = 0; i < axis; i++)
            newDims[i] = this[i];
        for (int i = axis + 1; i < _rank; i++)
            newDims[i - 1] = this[i];
        return new TensorShape(newDims);
    }

    // ── validation helpers ────────────────────────────────────────

    public static void AssertSameShape(TensorShape a, TensorShape b)
    {
        if (!a.Equals(b))
            throw new ArgumentException($"{(new System.Diagnostics.StackTrace()?.GetFrame(1)?.GetMethod()?.Name ?? "Operation")} requires identical shapes, got {a} and {b}.");
    }

    public static void AssertMatMulCompatible(TensorShape a, TensorShape b)
    {
        if (a._rank != 2 || b._rank != 2)
            throw new ArgumentException($"MatMul requires 2-D tensors, got {a} and {b}.");
        if (a.Cols != b.Rows)
            throw new ArgumentException(
                $"MatMul shape mismatch: {a} · {b} (inner dims {a.Cols} ≠ {b.Rows}).");
    }

    // ── equality ──────────────────────────────────────────────────

    public bool Equals(TensorShape other)
    {
        if (_rank != other._rank) return false;
        if (_dimsOverflow is not null && other._dimsOverflow is not null)
            return _dimsOverflow.AsSpan().SequenceEqual(other._dimsOverflow);
        // Inline comparison — no allocation.
        return _rank switch
        {
            1 => _d0 == other._d0,
            2 => _d0 == other._d0 && _d1 == other._d1,
            3 => _d0 == other._d0 && _d1 == other._d1 && _d2 == other._d2,
            4 => _d0 == other._d0 && _d1 == other._d1 && _d2 == other._d2 && _d3 == other._d3,
            _ => false
        };
    }

    public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);
    public override int GetHashCode() => HashCode.Combine(_rank, _elementCount);

    public static bool operator ==(TensorShape a, TensorShape b) => a.Equals(b);
    public static bool operator !=(TensorShape a, TensorShape b) => !a.Equals(b);

    // ── display ───────────────────────────────────────────────────

    public override string ToString()
    {
        var sb = new StringBuilder("(");
        for (int i = 0; i < _rank; i++)
        {
            sb.Append(this[i]);
            if (i < _rank - 1) sb.Append(", ");
        }
        sb.Append(')');
        return sb.ToString();
    }

    // ── private helpers ───────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetStrideInline(int i) => i switch
    {
        0 => _s0,
        1 => _s1,
        2 => _s2,
        3 => _s3,
        _ => 0
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeStride(ReadOnlySpan<int> dims, int i)
    {
        long s = 1;
        for (int j = i + 1; j < dims.Length; j++)
            s *= dims[j];
        if (s > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(dims), $"Stride {s} overflows int.");
        return (int)s;
    }

    private int[] CreateInlineDimsArray() => _rank switch
    {
        1 => [_d0],
        2 => [_d0, _d1],
        3 => [_d0, _d1, _d2],
        4 => [_d0, _d1, _d2, _d3],
        _ => []
    };

    private int[] CreateInlineStridesArray() => _rank switch
    {
        1 => [_s0],
        2 => [_s0, _s1],
        3 => [_s0, _s1, _s2],
        4 => [_s0, _s1, _s2, _s3],
        _ => []
    };

    private static int[] ComputeStridesArray(int[] dims)
    {
        var s = new int[dims.Length];
        s[^1] = 1;
        for (int i = dims.Length - 2; i >= 0; i--)
        {
            long val = (long)s[i + 1] * dims[i + 1];
            if (val > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(dims), $"Stride {val} overflows int.");
            s[i] = (int)val;
        }
        return s;
    }

    private static int ComputeElementCount(ReadOnlySpan<int> dims)
    {
        long n = 1;
        foreach (int d in dims) n *= d;
        if (n > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(dims), $"Element count {n} overflows int.");
        return (int)n;
    }
}