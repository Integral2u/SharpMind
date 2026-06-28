using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;

namespace SharpMind.Model;

public sealed class QuantizedKVCache : IKVCache
{
    private const int QK = 32;
    private const int Q8_BLOCK = 34;
    private const int Q4_BLOCK = 18;

    private readonly byte[] _qKeys;
    private readonly byte[] _qValues;
    private readonly System.Runtime.InteropServices.GCHandle _keyHandle;
    private readonly System.Runtime.InteropServices.GCHandle _valHandle;
    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _nBlocks;
    private readonly int _blockBytes;
    private readonly int _qStride;
    private readonly int _headStride;

    public GgufDtype QuantKind { get; }

    public QuantizedKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim, GgufDtype quantKind = GgufDtype.Q8_0)
    {
        _batchSize = batchSize;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        QuantKind = quantKind;
        _nBlocks = (headDim + QK - 1) / QK;
        _blockBytes = quantKind switch
        {
            GgufDtype.Q4_0 => Q4_BLOCK,
            _ => Q8_BLOCK
        };
        _qStride = _nBlocks * _blockBytes;
        _headStride = maxSeqLen * _qStride;
        MaxSeqLen = maxSeqLen;

        long totalSize = (long)batchSize * numKvHeads * maxSeqLen * _qStride;
        _qKeys = new byte[totalSize];
        _qValues = new byte[totalSize];
        _keyHandle = System.Runtime.InteropServices.GCHandle.Alloc(_qKeys, System.Runtime.InteropServices.GCHandleType.Pinned);
        _valHandle = System.Runtime.InteropServices.GCHandle.Alloc(_qValues, System.Runtime.InteropServices.GCHandleType.Pinned);
    }

    public int CurrentPosition { get; private set; } = 0;
    public int MaxSeqLen { get; }
    public bool IsContiguous => true;
    public bool IsQuantized => true;
    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public unsafe byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead)
    {
        return (byte*)_keyHandle.AddrOfPinnedObject()
             + (long)batchIdx * (_numKvHeads * _headStride)
             + (long)kvHead * _headStride
             + (long)position * _qStride;
    }

    public unsafe byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead)
    {
        return (byte*)_valHandle.AddrOfPinnedObject()
             + (long)batchIdx * (_numKvHeads * _headStride)
             + (long)kvHead * _headStride
             + (long)position * _qStride;
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
                    if (QuantKind == GgufDtype.Q4_0)
                        QuantizeRowQ4_0(srcK, dstK, headDim);
                    else
                        QuantizeRowQ8_0(srcK, dstK, headDim);

                    float* srcV = v.DataPtr
                        + (long)b * (seqLen * numKvHeads * headDim)
                        + (long)s * (numKvHeads * headDim)
                        + (long)h * headDim;

                    byte* dstV = GetQuantizedValuePtr(b, CurrentPosition + s, h);
                    if (QuantKind == GgufDtype.Q4_0)
                        QuantizeRowQ4_0(srcV, dstV, headDim);
                    else
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
                    byte* headBaseK = GetQuantizedKeyPtr(b, 0, h) - (long)b * (_numKvHeads * _headStride) - (long)h * _headStride;
                byte* headBaseV = GetQuantizedValuePtr(b, 0, h) - (long)b * (_numKvHeads * _headStride) - (long)h * _headStride;

                    byte* srcK = headBaseK + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride + (long)offset * _qStride;
                    byte* dstK = headBaseK + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride;
                    Buffer.MemoryCopy(srcK, dstK, (long)keep * _qStride, (long)keep * _qStride);

                    byte* srcV = headBaseV + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride + (long)offset * _qStride;
                    byte* dstV = headBaseV + (long)b * (_numKvHeads * _headStride) + (long)h * _headStride;
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
        if (_keyHandle.IsAllocated) _keyHandle.Free();
        if (_valHandle.IsAllocated) _valHandle.Free();
    }

    private static unsafe void QuantizeRowQ8_0(float* src, byte* dst, int n)
    {
        int nBlocks = (n + QK - 1) / QK;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockEnd = Math.Min(QK, n - b * QK);
            float* pSrc = src + b * QK;
            byte* pDst = dst + b * Q8_BLOCK;

            float amax = 0f;
            for (int i = 0; i < blockEnd; i++)
            {
                float abs = Math.Abs(pSrc[i]);
                if (abs > amax) amax = abs;
            }

            float d = amax / 127f;
            if (amax == 0f) d = 1f;

            *(ushort*)pDst = QuantizationKernels.FloatToHalf_F16C(d);
            sbyte* qVals = (sbyte*)(pDst + 2);

            for (int i = 0; i < blockEnd; i++)
            {
                int q = (int)MathF.Round(pSrc[i] / d);
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                qVals[i] = (sbyte)q;
            }
            for (int i = blockEnd; i < QK; i++)
                qVals[i] = 0;
        }
    }

    private static unsafe void QuantizeRowQ4_0(float* src, byte* dst, int n)
    {
        int nBlocks = (n + QK - 1) / QK;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockEnd = Math.Min(QK, n - b * QK);
            float* pSrc = src + b * QK;
            byte* pDst = dst + b * Q4_BLOCK;

            float amax = 0f;
            for (int i = 0; i < blockEnd; i++)
            {
                float abs = Math.Abs(pSrc[i]);
                if (abs > amax) amax = abs;
            }

            float d = amax / 8f;
            if (amax == 0f) d = 1f;

            *(ushort*)pDst = QuantizationKernels.FloatToHalf_F16C(d);
            byte* qNibbles = pDst + 2;

            for (int i = 0; i < blockEnd; i++)
            {
                int q = (int)MathF.Round(pSrc[i] / d) + 8;
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                if ((i & 1) == 0)
                    qNibbles[i / 2] = (byte)(q & 0x0F);
                else
                    qNibbles[i / 2] = (byte)((qNibbles[i / 2] & 0x0F) | ((byte)q << 4));
            }
            for (int i = blockEnd; i < QK; i++)
            {
                if ((i & 1) == 0)
                    qNibbles[i / 2] &= 0xF0;
                else
                    qNibbles[i / 2] &= 0x0F;
            }
        }
    }
}
