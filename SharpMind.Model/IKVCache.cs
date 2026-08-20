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
    public int Length { get; }
    public int MaxSeqLen { get; }
    public bool IsFull { get; }
    public bool IsContiguous { get; }
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
                w.Write((byte)3);
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
            {
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
