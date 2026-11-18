using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// A cache whose rows carry their own absolute position. Attention reads it to
/// place rotary embeddings per row when a batch mixes sequences at different
/// positions; without it a single <c>positionOffset</c> applies to every row.
/// </summary>
public interface IRowPositionedKvCache : IKVCache
{
    /// <summary>Absolute position of the next token in <paramref name="row"/>.</summary>
    int PositionForRow(int row);
}

/// <summary>
/// Presents a handful of rows of a larger batched cache as a small cache, one
/// view row per parent row. A continuous batcher owns <c>maxBatch</c> rows but
/// runs forwards for however many sequences are active, and prefills one
/// sequence at a time; this view is how a <c>[1, L]</c> or <c>[3, 1]</c> forward
/// reaches the right parent rows without moving any key/value data.
///
/// The view owns no storage and cannot rewind or reset a subset of rows, so
/// those operations throw. Its rows are also assumed to keep
/// position == length (the batcher never trims), which is what
/// <see cref="PositionForRow"/> reports.
/// </summary>
public sealed class KVCacheRowView(IKVCache parent) : IRowPositionedKvCache
{
    private readonly IKVCache _parent = parent;
    private int[] _rows = [];
    private int _count;

    /// <summary>Maps view rows 0..<paramref name="rows"/>.Length-1 onto parent rows.</summary>
    public void SetRows(ReadOnlySpan<int> rows)
    {
        if (_rows.Length < rows.Length) _rows = new int[rows.Length];
        rows.CopyTo(_rows);
        _count = rows.Length;
    }

    public int Count => _count;
    public IKVCache Parent => _parent;

    public int RowCount => _count;
    public int MaxSeqLen => _parent.MaxSeqLen;
    public bool IsContiguous => _parent.IsContiguous;
    public bool SupportsRaggedRows => true;

    public int Length
    {
        get
        {
            int max = 0;
            for (int i = 0; i < _count; i++)
            {
                int length = _parent.LengthForRow(_rows[i]);
                if (length > max) max = length;
            }
            return max;
        }
    }

    public bool IsFull => _count > 0 && Length >= MaxSeqLen;

    public int LengthForRow(int row) => _parent.LengthForRow(_rows[row]);

    /// <summary>Rows are dense from position 0 and are never trimmed while in a view.</summary>
    public int PositionForRow(int row) => LengthForRow(row);

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
        => _parent.GetKeyPtr(_rows[batchIdx], position, kvHead);

    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead)
        => _parent.GetValuePtr(_rows[batchIdx], position, kvHead);

    public bool IsQuantized => _parent.IsQuantized;
    public QuantDType QuantKind => _parent.QuantKind;

    public unsafe byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead)
        => _parent.GetQuantizedKeyPtr(_rows[batchIdx], position, kvHead);

    public unsafe byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead)
        => _parent.GetQuantizedValuePtr(_rows[batchIdx], position, kvHead);

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];
        int rowLength = seqLen * numKvHeads * headDim;
        if (batch > _count)
            throw new InvalidOperationException(
                $"Row view maps {_count} row(s) but the update carries {batch}.");

        for (int b = 0; b < batch; b++)
        {
            long start = (long)b * rowLength;
            _parent.UpdateRow(_rows[b],
                k.Data.Slice((int)start, rowLength),
                v.Data.Slice((int)start, rowLength),
                numKvHeads, headDim);
        }
    }

    public void UpdateRow(int row, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int numKvHeads, int headDim)
        => _parent.UpdateRow(_rows[row], k, v, numKvHeads, headDim);

    public void Reset() => throw OwnsNoStorage(nameof(Reset));
    public void TrimToLast(int keep) => throw OwnsNoStorage(nameof(TrimToLast));
    public void Truncate(int length) => throw OwnsNoStorage(nameof(Truncate));
    public void MoveRow(int from, int to) => throw OwnsNoStorage(nameof(MoveRow));
    public object? Snapshot() => throw OwnsNoStorage(nameof(Snapshot));
    public void Restore(object? snapshot) => throw OwnsNoStorage(nameof(Restore));

    private NotSupportedException OwnsNoStorage(string operation) =>
        new($"A {nameof(KVCacheRowView)} owns no storage; call {operation} on the underlying cache.");

    public void Dispose() { /* the parent owns the storage */ }
}
