using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public sealed class QuantizedKVCache : IKVCache
{
    private const int QK = 32;
    private const int BLOCK_BYTES = 34;

    private readonly byte[] _qKeys;
    private readonly byte[] _qValues;
    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _nBlocks;
    private readonly int _qStride;
    private readonly int _headStride;

    public QuantizedKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
    {
        _batchSize = batchSize;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _nBlocks = (headDim + QK - 1) / QK;
        _qStride = _nBlocks * BLOCK_BYTES;
        _headStride = maxSeqLen * _qStride;
        MaxSeqLen = maxSeqLen;

        long totalSize = (long)batchSize * numKvHeads * maxSeqLen * _qStride;
        _qKeys = new byte[totalSize];
        _qValues = new byte[totalSize];
    }

    public int CurrentPosition { get; private set; } = 0;
    public int MaxSeqLen { get; }
    public int AllocatedCapacity => MaxSeqLen;
    public bool IsContiguous => true;
    public bool IsQuantized => true;
    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public unsafe byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead)
    {
        fixed (byte* p = _qKeys)
        {
            return p + (long)batchIdx * (_numKvHeads * _headStride)
                     + (long)kvHead * _headStride
                     + (long)position * _qStride;
        }
    }

    public unsafe byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead)
    {
        fixed (byte* p = _qValues)
        {
            return p + (long)batchIdx * (_numKvHeads * _headStride)
                     + (long)kvHead * _headStride
                     + (long)position * _qStride;
        }
    }

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead) => null;
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) => null;

    public void Reset() => CurrentPosition = 0;

    public void Truncate(int length) => CurrentPosition = Math.Min(length, CurrentPosition);

    public unsafe void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];

        if (CurrentPosition + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"QuantizedKVCache overflow: position {CurrentPosition} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numKvHeads; h++)
                {
                    float* srcK = k.DataPtr
                        + (long)b * (seqLen * numKvHeads * headDim)
                        + (long)s * (numKvHeads * headDim)
                        + (long)h * headDim;

                    byte* dstK = GetQuantizedKeyPtr(b, CurrentPosition + s, h);
                    QuantizeRowQ8_0(srcK, dstK, headDim);

                    float* srcV = v.DataPtr
                        + (long)b * (seqLen * numKvHeads * headDim)
                        + (long)s * (numKvHeads * headDim)
                        + (long)h * headDim;

                    byte* dstV = GetQuantizedValuePtr(b, CurrentPosition + s, h);
                    QuantizeRowQ8_0(srcV, dstV, headDim);
                }
            }
        }

        CurrentPosition += seqLen;
    }

    public void TrimToLast(int keep)
    {
        if (keep <= 0)
        {
            Reset();
            return;
        }
        if (keep >= CurrentPosition) return;

        int offset = CurrentPosition - keep;

        unsafe
        {
            for (int b = 0; b < _batchSize; b++)
            {
                for (int h = 0; h < _numKvHeads; h++)
                {
                    byte* headBase = GetQuantizedKeyPtr(b, 0, h) - (long)b * (_numKvHeads * _headStride) - (long)h * _headStride;

                    byte* srcK = headBase + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride + (long)offset * _qStride;
                    byte* dstK = headBase + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride;
                    Buffer.MemoryCopy(srcK, dstK, (long)keep * _qStride, (long)keep * _qStride);

                    byte* srcV = headBase + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride + (long)offset * _qStride;
                    byte* dstV = headBase + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride;
                    Buffer.MemoryCopy(srcV, dstV, (long)keep * _qStride, (long)keep * _qStride);
                }
            }
        }

        CurrentPosition = keep;
    }

    public object? Snapshot()
    {
        if (CurrentPosition == 0) return null;
        int headBytes = CurrentPosition * _qStride;
        int totalBytes = _batchSize * _numKvHeads * headBytes;
        var k = new byte[totalBytes];
        var v = new byte[totalBytes];
        Buffer.BlockCopy(_qKeys, 0, k, 0, totalBytes);
        Buffer.BlockCopy(_qValues, 0, v, 0, totalBytes);
        return (CurrentPosition, k, v);
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null) return;
        var (pos, k, v) = ((int, byte[], byte[]))snapshot;
        Buffer.BlockCopy(k, 0, _qKeys, 0, k.Length);
        Buffer.BlockCopy(v, 0, _qValues, 0, v.Length);
        CurrentPosition = pos;
    }

    public void Dispose()
    {
        // byte[] is managed — nothing to release
    }

    private static unsafe void QuantizeRowQ8_0(float* src, byte* dst, int n)
    {
        int nBlocks = (n + QK - 1) / QK;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockEnd = Math.Min(QK, n - b * QK);
            float* pSrc = src + b * QK;
            byte* pDst = dst + b * BLOCK_BYTES;

            float amax = 0f;
            for (int i = 0; i < blockEnd; i++)
            {
                float abs = Math.Abs(pSrc[i]);
                if (abs > amax) amax = abs;
            }

            float d = amax / 127f;
            if (amax == 0f) d = 1f;

            *(ushort*)pDst = FloatToHalf_Scalar(d);
            sbyte* qVals = (sbyte*)(pDst + 2);

            for (int i = 0; i < blockEnd; i++)
            {
                int q = (int)MathF.Round(pSrc[i] / d);
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                qVals[i] = (sbyte)q;
            }
            // Zero out tail (if n not multiple of QK)
            for (int i = blockEnd; i < QK; i++)
                qVals[i] = 0;
        }
    }

    private static unsafe ushort FloatToHalf_Scalar(float f)
    {
        uint bits = *(uint*)&f;
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = bits & 0x007FFFFF;

        if (exp <= 0)
        {
            if (exp < -10) return (ushort)sign;
            mant = (mant | 0x00800000) >> (1 - exp);
            return (ushort)(sign | (mant >> 13));
        }

        if (exp >= 31)
        {
            if (exp > 31) return (ushort)(sign | 0x7C00 | (mant >> 13));
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 0x200u : 0));
        }

        return (ushort)(sign | ((uint)exp << 10) | (mant >> 13));
    }
}
