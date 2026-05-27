using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Format;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Model.Layers;

public sealed class LinearLayer : IDisposable
{
    private readonly Tensor<float> _weight;
    private Tensor<float>? _weightBT;
    private readonly Tensor<float>? _bias;
    private bool _disposed;

    // Raw GGUF quantized data for quantized matmul (null means use float32 path).
    public byte[]? RawQuantizedData { get; set; }
    public GgufDtype? QuantDtype { get; set; }
    public bool UseQuantizedForward { get; set; }

    public LinearLayer(string name, int inFeatures, int outFeatures, bool bias = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = new Tensor<float>(inFeatures, outFeatures);
        _bias = bias ? new Tensor<float>(outFeatures) : null;
    }

    public LinearLayer(int inFeatures, int outFeatures, bool bias = false)
        : this($"Linear.{Guid.NewGuid():N}", inFeatures, outFeatures, bias)
    {
    }

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public bool HasBias => _bias is not null;
    public string Name { get; }
    public Tensor<float> Weight => _weight;
    public Tensor<float>? Bias => _bias;

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter($"{Name}.weight", _weight);
        if (_bias is not null)
            yield return new Parameter($"{Name}.bias", _bias);
    }

    public Tensor<float> Forward(Tensor<float> input, TensorOps ops)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        Tensor<float> output;
        if (UseQuantizedForward && RawQuantizedData != null && QuantDtype.HasValue)
        {
            output = QuantizedForward(flat, ops);
        }
        else
        {
            _weightBT ??= TensorOps.Transpose(_weight);
            output = ops.MatMulWithBT(flat, _weightBT);
        }

        if (_bias is not null)
            TensorOps.AddInPlace(output, BroadcastBias(batchSize));
        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return reshaped;
        }
        return output;
    }

    private Tensor<float> QuantizedForward(Tensor<float> input, TensorOps ops)
    {
        var dtype = QuantDtype!.Value;
        var rawData = RawQuantizedData!;
        int m = input.ElementCount / InFeatures;
        var result = new Tensor<float>(m, OutFeatures);
        int inF = InFeatures, outF = OutFeatures;

        for (int row = 0; row < m; row++)
        {
            unsafe
            {
                float* pIn = input.DataPtr + (long)row * inF;
                float* pOut = result.DataPtr + (long)row * outF;
                fixed (byte* pRaw = rawData)
                {
                    IntPtr pInPtr = (IntPtr)pIn;
                    IntPtr pOutPtr = (IntPtr)pOut;
                    IntPtr pRawPtr = (IntPtr)pRaw;
                    System.Threading.Tasks.Parallel.For(0, outF, col =>
                    {
                        float* pInL = (float*)pInPtr;
                        float* pOutL = (float*)pOutPtr;
                        byte* pRawL = (byte*)pRawPtr;
                        pOutL[col] = VecDotQxK(pInL, pRawL, col, inF, dtype);
                    });
                }
            }
        }
        return result;
    }

    private static unsafe float VecDotQxK(float* input, byte* rawWeights, int col, int inFeatures, GgufDtype dtype)
    {
        return dtype switch
        {
            GgufDtype.Q3_K => VecDotQ3K(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_K => VecDotQ4K(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_K => VecDotQ5K(input, rawWeights, col, inFeatures),
            GgufDtype.Q6_K => VecDotQ6K(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_0 => VecDotQ4_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q4_1 => VecDotQ4_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_0 => VecDotQ5_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q5_1 => VecDotQ5_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_0 => VecDotQ8_0(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_1 => VecDotQ8_1(input, rawWeights, col, inFeatures),
            GgufDtype.Q2_K => VecDotQ2K(input, rawWeights, col, inFeatures),
            GgufDtype.Q8_K => VecDotQ8K(input, rawWeights, col, inFeatures),
            _ => 0f
        };
    }

    private static bool IsSupportedQuantDtype(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q3_K => true,
        GgufDtype.Q4_K => true,
        GgufDtype.Q5_K => true,
        GgufDtype.Q6_K => true,
        GgufDtype.Q4_0 => true,
        GgufDtype.Q4_1 => true,
        GgufDtype.Q5_0 => true,
        GgufDtype.Q5_1 => true,
        GgufDtype.Q8_0 => true,
        GgufDtype.Q8_1 => true,
        GgufDtype.Q2_K => true,
        GgufDtype.Q8_K => true,
        _ => false    // TEMP: disabled K-quants to isolate VecDot vs dequant bug
    };

    public bool SetRawWeight(byte[] rawData, GgufDtype dtype)
    {
        RawQuantizedData = rawData;
        QuantDtype = dtype;
        UseQuantizedForward = IsSupportedQuantDtype(dtype);
        return UseQuantizedForward;
    }

    // ── GGUF quantized dot product kernels (matching llama.cpp vec_dot) ──

    private const int QK_K = 256;

    internal static unsafe float VecDotQ3K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;  // d[2]+hmask[32]+qs[64]+scales[12]
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dAll = HalfToFloat(*(ushort*)block);  // d @ offset 0
            byte* hmask = block + 2;                     // hmask @ offset 2 (32 bytes)
            byte* qs = block + 34;                       // qs @ offset 34 (64 bytes)

            for (int j = 0; j < 12; j++) scaleBuf[j] = block[98 + j]; // scales @ offset 98

            uint* aux = (uint*)scaleBuf;
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
            aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
            aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
            aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
            sbyte* sc8 = (sbyte*)scaleBuf;

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            for (int i = 0; i < blockEnd; i++)
            {
                // qs transposed: byte = (i/128)*32 + i%32, shift = ((i%128)/32)*2
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[i % 32] >> (i / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                int sub = i / 16;
                float val = dAll * (sc8[sub] - 32) * actual;
                sum += input[b * QK_K + i] * val;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;  // d[2]+dmin[2]+scales[K_SCALE_SIZE]+qs[128]
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dSuper = HalfToFloat(*(ushort*)block);
            float minSuper = HalfToFloat(*(ushort*)(block + 2));
            byte* scales = block + 4;   // K_SCALE_SIZE bytes (indices 0..11 used)
            byte* qs = block + 16;      // 128 bytes

            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale(idx + 0, scales);
                byte m0 = GetScaleMinK4_Min(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale(idx + 1, scales);
                byte m1 = GetScaleMinK4_Min(idx + 1, scales);
                float d1 = dSuper * sc0;
                float m1v = minSuper * m0;
                float d2 = dSuper * sc1;
                float m2v = minSuper * m1;

                int qIdx = (j / 64) * 32;
                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);
                for (int l = 0; l < halfRem; l++)
                {
                    int pos = b * QK_K + j + l;
                    sum += input[pos] * (d1 * (qs[qIdx + l] & 0x0F) - m1v);
                }
                for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                {
                    int pos = b * QK_K + j + 32 + l;
                    sum += input[pos] * (d2 * (qs[qIdx + l] >> 4) - m2v);
                }
                idx += 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ5K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;  // d[2]+dmin[2]+scales[K_SCALE_SIZE]+qh[32]+qs[128]
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            float min = HalfToFloat(*(ushort*)(block + 2));
            byte* scales = block + 4;   // K_SCALE_SIZE bytes
            byte* qh = block + 16;      // 32 bytes
            byte* qs = block + 48;      // 128 bytes

            int idx = 0;
            int qIdx = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale(idx + 0, scales);
                byte m0 = GetScaleMinK4_Min(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale(idx + 1, scales);
                byte m1 = GetScaleMinK4_Min(idx + 1, scales);
                float d1 = d * sc0; float m1v = min * m0;
                float d2 = d * sc1; float m2v = min * m1;

                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);
                for (int l = 0; l < halfRem; l++)
                {
                    int pos = b * QK_K + j + l;
                    int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                    sum += input[pos] * (d1 * val - m1v);
                }
                for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                {
                    int pos = b * QK_K + j + 32 + l;
                    int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                    sum += input[pos] * (d2 * val - m2v);
                }
                qIdx += 32;
                idx += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ6K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* ql = block;                          // 128 bytes
            byte* qh = block + 128;                     // 64 bytes
            sbyte* scales = (sbyte*)(block + 192);     // 16 bytes
            float d = HalfToFloat(*(ushort*)(block + 208));

            int valid = Math.Min(QK_K, inFeatures - b * QK_K);

            // Current ggml Q6_K: process 128 values at a time
            for (int nOff = 0; nOff < valid; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, valid - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = b * QK_K + nOff + l;
                    int i2 = b * QK_K + nOff + l + 32;

                    if (i2 >= b * QK_K + valid)
                    {
                        if (i1 < b * QK_K + valid)
                            sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = b * QK_K + nOff + l + 64;
                    int i4 = b * QK_K + nOff + l + 96;

                    sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                    sum += input[i2] * (d * psc[is_ + 2] * (q2v - 32));
                    sum += input[i3] * (d * psc[is_ + 4] * (q3v - 32));
                    sum += input[i4] * (d * psc[is_ + 6] * (q4v - 32));
                }
            }
        }
        return (float)sum;
    }

    private static unsafe float HSum256(Vector256<float> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        var s = Sse.Add(lo, hi);
        var s2 = Sse3.HorizontalAdd(s, s);
        return s2.GetElement(0) + s2.GetElement(1);
    }

    internal static unsafe float VecDotQ8_0(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            int i = 0;
            if (Avx2.IsSupported && blockEnd >= 8)
            {
                var vacc = Vector256<float>.Zero;
                var vd = Vector256.Create(d);
                for (; i <= blockEnd - 8; i += 8)
                {
                    var vi = Vector256.LoadUnsafe(ref pIn[i]);
                    var vw = Avx.ConvertToVector256Single(
                        Avx2.ConvertToVector256Int32(values + i));
                    var vs = Avx.Multiply(vw, vd);
                    vacc = Fma.IsSupported
                        ? Fma.MultiplyAdd(vi, vs, vacc)
                        : Avx.Add(vacc, Avx.Multiply(vi, vs));
                }
                sum += HSum256(vacc);
            }
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4_0(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4_1(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            float m = HalfToFloat(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ5_0(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int j = i / 2;
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (j % 2 == 0) ? (qs[j] & 0x0F) : (qs[j] >> 4);
                int q = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ5_1(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            float m = HalfToFloat(*(ushort*)(block + 2));
            uint qh = *(uint*)(block + 4);
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int j = i % 16;
                int xh = i < 16
                    ? ((int)((qh >> (j + 0)) & 1) << 4)
                    : ((int)((qh >> (j + 12)) & 1) << 4);
                int q = ((qs[j / 2] >> (4 * (j % 2))) & 0x0F) | xh;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_1(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat(*(ushort*)block);
            // s field (offset 2) is d * sum(qs), used for K-quant dot product optimization
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK + i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ2K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        const int QK_K = 256;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dSuper = HalfToFloat(*(ushort*)block);
            float minSuper = HalfToFloat(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 20;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    float dl = dSuper * (scales[isc] & 0x0F);
                    float ml = minSuper * (scales[isc] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (qs[qOff + l] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + l] * (dl * v - ml);
                    }

                    dl = dSuper * (scales[isc + 1] & 0x0F);
                    ml = minSuper * (scales[isc + 1] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + 16 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (qs[qOff + l + 16] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + 16 + l] * (dl * v - ml);
                    }
                    shift += 2;
                }
                qOff += 32;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8K(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        const int QK_K = 256;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = *(float*)block;
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK_K + i] * (qs[i] * d);
        }
        return (float)sum;
    }

    private static unsafe byte GetScaleMinK4_Scale(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j] & 0x3F);
        return (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    }

    private static unsafe byte GetScaleMinK4_Min(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j + 4] & 0x3F);
        return (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }

    private static float HalfToFloat(ushort half)
    {
        int sign = (half >> 15) & 0x1;
        int exp = (half >> 10) & 0x1F;
        int mant = half & 0x3FF;
        if (exp == 0)
            return (sign == 0 ? 1f : -1f) * (mant / 1024f) * MathF.Pow(2f, -14f);
        if (exp == 31)
            return mant == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
        return (sign == 0 ? 1f : -1f) * MathF.Pow(2f, exp - 15) * (1f + mant / 1024f);
    }

    public (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input, TensorOps ops)
    {
        ThrowIfDisposed();
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];
        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;
        var output = ops.MatMul(flat, _weight);
        if (_bias is not null)
            TensorOps.AddInPlace(output, BroadcastBias(batchSize));
        var state = new LinearLayerState(input, flat, needReshape, _weight);
        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return (reshaped, state);
        }
        return (output, state);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state, TensorOps ops)
    {
        int batchSize = state.NeedReshape
            ? gradOutput.ElementCount / OutFeatures
            : gradOutput.Shape[^2];
        var flatGradOut = state.NeedReshape
            ? gradOutput.Reshape(batchSize, OutFeatures)
            : gradOutput;
        
        // gradInput = gradOutput @ weight
        var gradInputFlat = ops.MatMul(flatGradOut, TensorOps.Transpose(_weight));
        
        // gradWeight += input^T @ gradOutput
        using var inputT = TensorOps.Transpose(state.Input);
        using var dw = ops.MatMul(inputT, flatGradOut);
        var wg = state.WeightGrad;
        for (int i = 0; i < dw.ElementCount; i++)
            wg.Data[i] += dw.Data[i];
        dw.Dispose();
        inputT.Dispose();

        // gradBias
        if (_bias is not null)
        {
            state.BiasGrad ??= Tensor<float>.Zeros(OutFeatures);
            for (int i = 0; i < batchSize; i++)
            {
                ReadOnlySpan<float> row = flatGradOut.RowSpan(i);
                for (int j = 0; j < OutFeatures; j++)
                    state.BiasGrad.Data[j] += row[j];
            }
        }
        
        flatGradOut.Dispose();
        
        if (state.NeedReshape)
        {
            int[] inDims = [.. state.InputDims[..^1], InFeatures];
            var reshaped = gradInputFlat.Reshape(inDims);
            gradInputFlat.Dispose();
            return reshaped;
        }
        return gradInputFlat;
    }

    public void LoadWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(_weight.Data);
        InvalidateCache();
    }

    public void LoadWeightTransposed(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");

        // GGUF: [Out, In] -> SharpMind: [In, Out]
        int inF = InFeatures;
        int outF = OutFeatures;
        for (int o = 0; o < outF; o++)
            for (int i = 0; i < inF; i++)
                _weight.Data[i * outF + o] = data[o * inF + i];
        InvalidateCache();
    }

    private void InvalidateCache()
    {
        _weightBT?.Dispose();
        _weightBT = null;
    }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (_bias is null) throw new InvalidOperationException("No bias.");
        if (data.Length != _bias.ElementCount)
            throw new ArgumentException($"Expected {_bias.ElementCount} bias values, got {data.Length}.");
        data.CopyTo(_bias.Data);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weight.Dispose();
        _weightBT?.Dispose();
        _bias?.Dispose();
    }

    private Tensor<float> BroadcastBias(int batchSize)
    {
        var broadcast = new Tensor<float>(batchSize, OutFeatures);
        for (int i = 0; i < batchSize; i++)
            _bias!.Data.CopyTo(broadcast.RowSpan(i));
        return broadcast;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LinearLayer));
}

public sealed class LinearLayerState
{
    public Tensor<float> Input { get; }
    public int[] InputDims { get; }
    public bool NeedReshape { get; }
    public Tensor<float> WeightGrad { get; }
    public Tensor<float>? BiasGrad { get; set; }

    public LinearLayerState(Tensor<float> originalInput, Tensor<float> flatInput, bool needReshape, Tensor<float> weight)
    {
        Input = flatInput;
        InputDims = originalInput.Shape.Dims.ToArray();
        NeedReshape = needReshape;
        var dims = weight.Shape.Dims.ToArray();
        WeightGrad = Tensor<float>.Zeros(dims);
    }
}