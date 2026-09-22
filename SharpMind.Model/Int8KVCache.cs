using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// KV cache that stores every K and V row as int8, symmetric: scale = max|x| / 127, values rounded
/// and clamped to ±127. A value row is one float scale followed by headDim values. A key row carries
/// one scale per <see cref="KeyBlock"/> values, all scales first and then the headDim values: keys
/// have outlier channels, and one scale per row spends the whole int8 range on them (measured
/// +4.3% perplexity on Qwen2-0.5B against -0.2% with a scale per 16; values cost nothing per row).
/// The rows of one head are contiguous, which is the layout the int8 attention kernel reads through
/// <see cref="GetQuantizedKeyPtr"/> / <see cref="GetQuantizedValuePtr"/>.
/// </summary>
public sealed unsafe class Int8KVCache : IKVCache
{
    /// <summary>Values per key scale.</summary>
    public const int KeyBlock = 16;

    internal static int KeyBlocks(int headDim) => (headDim + KeyBlock - 1) / KeyBlock;
    internal static int KeyRowBytes(int headDim) => KeyBlocks(headDim) * sizeof(float) + headDim;
    internal static int ValueRowBytes(int headDim) => sizeof(float) + headDim;

    private readonly byte[] _keys;
    private readonly byte[] _values;
    private readonly byte* _keyBase;   // both arrays live on the pinned object heap
    private readonly byte* _valueBase;
    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _keyRowBytes;
    private readonly int _valueRowBytes;

    public Int8KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
    {
        _batchSize = batchSize;
        _numKvHeads = numKvHeads;
        _keyRowBytes = KeyRowBytes(headDim);
        _valueRowBytes = ValueRowBytes(headDim);
        MaxSeqLen = maxSeqLen;

        long keyTotal = (long)batchSize * numKvHeads * maxSeqLen * _keyRowBytes;
        if (keyTotal > Array.MaxLength)
            throw new ArgumentOutOfRangeException(nameof(maxSeqLen), $"Int8KVCache of {keyTotal} bytes exceeds a single array.");
        _keys = GC.AllocateArray<byte>((int)keyTotal, pinned: true);
        _values = GC.AllocateArray<byte>((int)((long)batchSize * numKvHeads * maxSeqLen * _valueRowBytes), pinned: true);
        _keyBase = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_keys));
        _valueBase = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_values));
    }

    public int MaxSeqLen { get; }
    public int Length { get; private set; }
    public bool IsFull => Length >= MaxSeqLen;
    public bool IsContiguous => true;
    public bool IsQuantized => true;
    public QuantDType QuantKind => QuantDType.I8;

    public byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead) => _keyBase + RowOffset(batchIdx, position, kvHead, _keyRowBytes);
    public byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead) => _valueBase + RowOffset(batchIdx, position, kvHead, _valueRowBytes);
    public float* GetKeyPtr(int batchIdx, int position, int kvHead) => null;
    public float* GetValuePtr(int batchIdx, int position, int kvHead) => null;

    private long RowOffset(int batchIdx, int position, int kvHead, int rowBytes) =>
        (((long)batchIdx * _numKvHeads + kvHead) * MaxSeqLen + position) * rowBytes;

    public void Reset() => Length = 0;

    public void Truncate(int length) => Length = Math.Min(length, Length);

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];
        if (Length + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"Int8KVCache overflow: position {Length} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numKvHeads; h++)
                {
                    long source = (((long)b * seqLen + s) * numKvHeads + h) * headDim;
                    QuantizeKeyRow(k.DataPtr + source, GetQuantizedKeyPtr(b, Length + s, h), headDim);
                    QuantizeRow(v.DataPtr + source, GetQuantizedValuePtr(b, Length + s, h), headDim);
                }

        Length += seqLen;
    }

    /// <summary>Writes one value row as [scale][n x int8].</summary>
    internal static void QuantizeRow(float* source, byte* destination, int n) =>
        *(float*)destination = QuantizeBlock(source, (sbyte*)(destination + sizeof(float)), n);

    /// <summary>Writes one key row as [scale per KeyBlock values][n x int8]; the last block may be short.</summary>
    internal static void QuantizeKeyRow(float* source, byte* destination, int n)
    {
        int blocks = KeyBlocks(n);
        float* scales = (float*)destination;
        sbyte* q = (sbyte*)(destination + blocks * sizeof(float));
        for (int b = 0; b < blocks; b++)
        {
            int start = b * KeyBlock;
            scales[b] = QuantizeBlock(source + start, q + start, Math.Min(KeyBlock, n - start));
        }
    }

    /// <summary>scale = max|x| / 127, values rounded and clamped to ±127; returns the scale.</summary>
    private static float QuantizeBlock(float* source, sbyte* q, int n)
    {
        float amax = 0f;
        for (int i = 0; i < n; i++) amax = MathF.Max(amax, MathF.Abs(source[i]));
        float scale = amax / 127f;
        if (scale == 0f)
        {
            new Span<byte>(q, n).Clear();
            return 0f;
        }
        float inverse = 1f / scale;
        for (int i = 0; i < n; i++)
            q[i] = (sbyte)Math.Clamp((int)MathF.Round(source[i] * inverse), -127, 127);
        return scale;
    }

    public void TrimToLast(int keep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(keep);
        if (keep >= Length) return;

        int drop = Length - keep;
        for (int b = 0; b < _batchSize; b++)
            for (int h = 0; h < _numKvHeads; h++)
            {
                // Buffer.MemoryCopy handles the overlap of the retained window with its new place.
                byte* keyHead = GetQuantizedKeyPtr(b, 0, h);
                byte* valueHead = GetQuantizedValuePtr(b, 0, h);
                Buffer.MemoryCopy(keyHead + (long)drop * _keyRowBytes, keyHead, (long)MaxSeqLen * _keyRowBytes, (long)keep * _keyRowBytes);
                Buffer.MemoryCopy(valueHead + (long)drop * _valueRowBytes, valueHead, (long)MaxSeqLen * _valueRowBytes, (long)keep * _valueRowBytes);
            }
        Length = keep;
    }

    public object? Snapshot()
    {
        if (Length == 0) return null;
        return (Length, CopyOut(_keys, _keyRowBytes), CopyOut(_values, _valueRowBytes));
    }

    private byte[] CopyOut(byte[] store, int rowBytes)
    {
        int perHead = checked(Length * rowBytes);
        var rows = new byte[checked(_batchSize * _numKvHeads * perHead)];
        int destination = 0;
        for (int b = 0; b < _batchSize; b++)
            for (int h = 0; h < _numKvHeads; h++)
            {
                store.AsSpan((int)RowOffset(b, 0, h, rowBytes), perHead).CopyTo(rows.AsSpan(destination));
                destination += perHead;
            }
        return rows;
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null) return;
        var (position, keys, values) = ((int, byte[], byte[]))snapshot;
        if ((uint)position > (uint)MaxSeqLen)
            throw new InvalidDataException($"Int8KVCache snapshot position {position} exceeds capacity {MaxSeqLen}.");
        int heads = _batchSize * _numKvHeads;
        if (keys.Length != checked(heads * position * _keyRowBytes) || values.Length != checked(heads * position * _valueRowBytes))
            throw new InvalidDataException("Int8KVCache snapshot dimensions do not match this cache.");

        CopyIn(keys, _keys, _keyRowBytes, position);
        CopyIn(values, _values, _valueRowBytes, position);
        Length = position;
    }

    private void CopyIn(byte[] rows, byte[] store, int rowBytes, int position)
    {
        int perHead = position * rowBytes;
        int source = 0;
        for (int b = 0; b < _batchSize; b++)
            for (int h = 0; h < _numKvHeads; h++)
            {
                rows.AsSpan(source, perHead).CopyTo(store.AsSpan((int)RowOffset(b, 0, h, rowBytes)));
                source += perHead;
            }
    }

    public void Dispose() { } // pinned managed arrays: nothing native to release
}
