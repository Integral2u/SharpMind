using System.Runtime.InteropServices;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public interface IKVCache : IDisposable
{
    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim);
    public void Reset();
    public void TrimToLast(int keep);
    public void Truncate(int length);
    /// <summary>
    /// Longest cached row. Single-sequence caches use one row, so this is the
    /// cache length; a batched cache reports its fullest row.
    /// </summary>
    public int Length { get; }
    public int MaxSeqLen { get; }

    /// <summary>
    /// Rows this cache stores. One for every single-sequence cache; a batched
    /// cache reports its batch capacity.
    /// </summary>
    public int RowCount => 1;

    /// <summary>
    /// Cached tokens in one row. Rows of a batched cache advance independently
    /// (sequences sit at different positions), so attention must ask per row
    /// instead of using <see cref="Length"/>. Single-row caches return
    /// <see cref="Length"/>.
    /// </summary>
    public int LengthForRow(int row) => Length;

    /// <summary>
    /// True when <see cref="LengthForRow(int)"/> and <see cref="UpdateRow"/> are
    /// implemented, i.e. rows may hold different lengths. A batcher must not
    /// assume ragged rows on a cache that reports false.
    /// </summary>
    public bool SupportsRaggedRows => false;

    /// <summary>
    /// Appends one row's keys and values, whose layout is a slice of the
    /// <see cref="Update"/> tensors: <c>[seqLen, numKvHeads, headDim]</c>. The
    /// row is placed at its own current length, so ragged batches keep every
    /// row correctly addressed. Implemented by the float and latent caches.
    /// </summary>
    public void UpdateRow(int row, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int numKvHeads, int headDim)
        => throw new NotSupportedException($"{GetType().Name} does not support per-row updates.");

    /// <summary>
    /// Drops trailing entries from one row while leaving the other rows alone. A
    /// continuous batcher needs this when one sequence rejects speculative drafts:
    /// its row must rewind to the verified prefix without touching its neighbours.
    /// </summary>
    public void TruncateRow(int row, int length)
        => throw new NotSupportedException($"{GetType().Name} does not support per-row truncation.");

    /// <summary>
    /// Moves one row's cached state to another row and clears the source. A
    /// continuous batcher uses this to keep its active rows dense when a
    /// sequence leaves.
    /// </summary>
    public void MoveRow(int from, int to)
        => throw new NotSupportedException($"{GetType().Name} does not support moving rows.");

    /// <summary>
    /// Empties one row, leaving its storage (and every other row) alone. Reaping
    /// a sequence whose row is the last active one has nobody to move over it,
    /// so the row must be reset explicitly before it is handed to a new
    /// sequence — otherwise that sequence appends after stale positions.
    /// </summary>
    public void ClearRow(int row)
        => throw new NotSupportedException($"{GetType().Name} does not support clearing rows.");
    public bool IsFull { get; }
    public bool IsContiguous { get; }
    /// <summary>
    /// True when <see cref="Truncate"/> can rebuild the state for a kept prefix.
    /// Attention KV caches can drop trailing rows; recurrent caches (e.g.
    /// <see cref="ShortConvCache"/>) cannot rewind their state and must be reset
    /// or restored from a snapshot instead.
    /// </summary>
    public bool SupportsTruncate => true;
    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead);
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead);
    public object? Snapshot();
    public void Restore(object? snapshot);

    /// <summary>
    /// Serializes the cache state to a compact binary format suitable for
    /// persistence (session save). Returns null when the cache is empty
    /// (position 0). Tags the format so <see cref="RestoreBytes"/> can
    /// reconstruct the correct tuple for each cache type.
    /// </summary>
    public byte[]? SnapshotBytes()
    {
        var obj = Snapshot();
        if (obj is null) return null;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);
        switch (obj)
        {
            case (int pos, float[] k, float[] v):
                w.Write((byte)1);
                w.Write(pos);
                WriteFloats(w, k);
                WriteFloats(w, v);
                break;
            case (int pos, float[] data):
                w.Write((byte)2);
                w.Write(pos);
                WriteFloats(w, data);
                break;
            case (int pos, byte[] k, byte[] v):
                // Q8_0/Q4_0 block rows and int8 rows can be the same size (headDim 64: 2 x 34 = 4 + 64),
                // so the tag carries the format and RestoreBytes refuses the other one.
                w.Write(QuantKind == QuantDType.I8 ? (byte)4 : (byte)3);
                w.Write(pos);
                w.Write(k.Length);
                w.Write(k);
                w.Write(v.Length);
                w.Write(v);
                break;
            default:
                throw new InvalidOperationException($"Unknown KVCache snapshot type: {obj.GetType()}");
        }
        return ms.ToArray();

        static void WriteFloats(BinaryWriter bw, float[] arr)
        {
            bw.Write(arr.Length);
            var bytes = MemoryMarshal.AsBytes(arr.AsSpan());
            bw.Write(bytes);
        }
    }

    /// <summary>
    /// Restores cache state previously produced by <see cref="SnapshotBytes"/>.
    /// No-op when <paramref name="data"/> is null or empty.
    /// </summary>
    public void RestoreBytes(byte[]? data)
    {
        if (data is null || data.Length == 0) return;
        using var ms = new MemoryStream(data);
        using var r = new BinaryReader(ms);
        byte tag = r.ReadByte();
        int pos = r.ReadInt32();
        switch (tag)
        {
            case 1: // KVCache: (int, float[], float[])
            {
                var k = ReadFloats(r);
                var v = ReadFloats(r);
                Restore((pos, k, v));
                break;
            }
            case 2: // PagedKVCache: (int, float[])
            {
                var d = ReadFloats(r);
                Restore((pos, d));
                break;
            }
            case 3: // QuantizedKVCache: (int, byte[], byte[])
            case 4: // Int8KVCache: (int, byte[], byte[])
            {
                if ((tag == 4) != (QuantKind == QuantDType.I8))
                    throw new InvalidDataException($"KVCache snapshot tag {tag} does not match this {QuantKind} cache.");
                byte[] k = r.ReadBytes(r.ReadInt32());
                byte[] v = r.ReadBytes(r.ReadInt32());
                Restore((pos, k, v));
                break;
            }
            default:
                throw new InvalidOperationException($"Unknown KVCache snapshot tag: {tag}");
        }

        static float[] ReadFloats(BinaryReader br)
        {
            int len = br.ReadInt32();
            var bytes = br.ReadBytes(len * sizeof(float));
            var arr = new float[len];
            bytes.AsSpan().CopyTo(MemoryMarshal.AsBytes(arr.AsSpan()));
            return arr;
        }
    }

    /// <summary>The quantization format used by quantized caches (default Q8_0).</summary>
    public QuantDType QuantKind => QuantDType.Q8_0;
    /// <summary>True if the cache stores quantized data.</summary>
    public bool IsQuantized => false;
    /// <summary>Returns a pointer to quantized key data at (batchIdx, position, kvHead). Only valid when IsQuantized is true.</summary>
    public unsafe byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead) => null;
    /// <summary>Returns a pointer to quantized value data at (batchIdx, position, kvHead). Only valid when IsQuantized is true.</summary>
    public unsafe byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead) => null;
}
