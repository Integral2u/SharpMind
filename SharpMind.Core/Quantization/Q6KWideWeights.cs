using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Q6_K weights repacked for an integer matmul whose tile is eight output rows
/// wide. The 6-bit values are widened to bytes during the repack, which keeps the
/// kernel close to the Q8_0 wide path: one 256-bit load holds four columns of
/// eight rows, <c>vpmaddubsw</c> sums pairs against the int8 activation, and
/// <c>vpmaddwd</c> collapses the pairs.
///
/// Q6_K stores 16-value sub-blocks with a signed int8 scale and a per-block
/// <c>d</c>; each sub-block evaluates as
///
///   scale * (Σ(q6*q) - 32*Σq)
///
/// where q6 is the unsigned 0..63 weight and q the signed activation. The
/// repack is about 1.5x the raw Q6_K tensor because the values become bytes.
/// </summary>
public sealed unsafe class Q6KWideWeights
{
    public const int RowsPerGroup = 8;
    private const int QBlock = 16;
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 210;
    private const int SubBlocks = SuperBlock / QBlock;
    private const int QuartetsPerSubBlock = QBlock / 4;
    private const int ScaleBytes = RowsPerGroup * sizeof(float);
    private const int QuantBytes = QuartetsPerSubBlock * RowsPerGroup * 4;
    private const int SubBlockBytes = ScaleBytes + QuantBytes;
    private const int BlockBytes = SubBlocks * SubBlockBytes;
    private const long MinParallelWork = 65_536;
    private const int StackScratchBytes = 8 * 1024;
    private const float MinQuantizableMax = 1e-30f;

    public static bool Enabled { get; set; } = true;
    public static int ParallelChunkOverride { get; set; }

    private readonly byte[] _data;
    private readonly byte* _p;
    private readonly int _blocks;
    private readonly int _groups;

    public byte[] Source { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public static bool IsSupported(int inFeatures, int outFeatures) =>
        Avx2.IsSupported && Fma.IsSupported &&
        inFeatures >= SuperBlock && inFeatures % SuperBlock == 0 &&
        outFeatures >= RowsPerGroup;

    public static Q6KWideWeights? TryCreate(byte[]? raw, int inFeatures, int outFeatures)
    {
        if (raw is null || !IsSupported(inFeatures, outFeatures)) return null;

        int blocks = inFeatures / SuperBlock;
        long expected = (long)outFeatures * blocks * RawBlockBytes;
        if (expected != raw.Length) return null;
        if ((long)(outFeatures / RowsPerGroup) * blocks * BlockBytes > Array.MaxLength) return null;
        return new Q6KWideWeights(raw, inFeatures, outFeatures);
    }

    public sealed class Cache
    {
        private sealed class Slot(byte[] source, Q6KWideWeights? wide)
        {
            public readonly byte[] Source = source;
            public readonly Q6KWideWeights? Wide = wide;
        }

        private readonly Lock _lock = new();
        private Slot? _slot;

        public Q6KWideWeights? Get(byte[] raw, int inFeatures, int outFeatures)
        {
            var slot = _slot;
            if (slot is not null && ReferenceEquals(slot.Source, raw)) return slot.Wide;
            lock (_lock)
            {
                slot = _slot;
                if (slot is null || !ReferenceEquals(slot.Source, raw))
                    _slot = slot = new Slot(raw, TryCreate(raw, inFeatures, outFeatures));
                return slot.Wide;
            }
        }

        public void Clear()
        {
            lock (_lock) _slot = null;
        }

        internal bool HasSlot
        {
            get { lock (_lock) return _slot is not null; }
        }
    }

    private Q6KWideWeights(byte[] raw, int inFeatures, int outFeatures)
    {
        Source = raw;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _blocks = inFeatures / SuperBlock;
        _groups = outFeatures / RowsPerGroup;
        _data = GC.AllocateUninitializedArray<byte>(_groups * _blocks * BlockBytes, pinned: true);
        _p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_data));

        int blocks = _blocks;
        fixed (byte* pRaw = raw)
        {
            long src = (long)pRaw;
            long dst = (long)_p;
            Parallel.For(0, _groups, g =>
            {
                for (int b = 0; b < blocks; b++)
                {
                    byte* blockDst = (byte*)dst + ((long)g * blocks + b) * BlockBytes;
                    RepackBlock((byte*)src, g, b, blocks, blockDst);
                }
            });
        }
    }

    private static void RepackBlock(byte* raw, int group, int block, int blocks, byte* dst)
    {
        for (int half = 0; half < 2; half++)
        {
            for (int row = 0; row < RowsPerGroup; row++)
            {
                byte* src = raw + ((long)(group * RowsPerGroup + row) * blocks + block) * RawBlockBytes;
                byte* ql = src + half * 64;
                byte* qh = src + 128 + half * 32;
                sbyte* scales = (sbyte*)(src + 192 + half * 8);
                float d = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(src + 208));

                for (int l = 0; l < 32; l++)
                {
                    int subGroup = l / 16;
                    int inSub = l % 16;
                    int quartet = inSub / 4;
                    int lane = inSub % 4;

                    int q1 = (ql[l] & 0x0F) | ((qh[l] & 0x03) << 4);
                    int q2 = (ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4);
                    int q3 = ((ql[l] >> 4) & 0x0F) | (((qh[l] >> 4) & 0x03) << 4);
                    int q4 = ((ql[l + 32] >> 4) & 0x0F) | (((qh[l] >> 6) & 0x03) << 4);

                    WriteValue(dst, half * 8 + subGroup, row, quartet, lane, q1);
                    WriteValue(dst, half * 8 + subGroup + 2, row, quartet, lane, q2);
                    WriteValue(dst, half * 8 + subGroup + 4, row, quartet, lane, q3);
                    WriteValue(dst, half * 8 + subGroup + 6, row, quartet, lane, q4);

                    if (lane == 0)
                    {
                        SetScale(dst, half * 8 + subGroup, row, d, scales[subGroup]);
                        SetScale(dst, half * 8 + subGroup + 2, row, d, scales[subGroup + 2]);
                        SetScale(dst, half * 8 + subGroup + 4, row, d, scales[subGroup + 4]);
                        SetScale(dst, half * 8 + subGroup + 6, row, d, scales[subGroup + 6]);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteValue(byte* dst, int sub, int row, int quartet, int lane, int value)
    {
        byte* quants = dst + sub * SubBlockBytes + ScaleBytes;
        quants[quartet * RowsPerGroup * 4 + row * 4 + lane] = (byte)value;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetScale(byte* dst, int sub, int row, float d, int scale)
    {
        float* scales = (float*)(dst + sub * SubBlockBytes);
        scales[row] = d * scale;
    }

    public void MatMul(float* input, float* output, int m)
    {
        int qBytes = checked(m * InFeatures);
        int blocks16 = checked(m * (InFeatures / QBlock));
        int scratchBytes = checked(qBytes + blocks16 * (sizeof(float) + sizeof(int)));

        if (scratchBytes <= StackScratchBytes)
        {
            byte* scratch = stackalloc byte[scratchBytes];
            MatMulCore(input, output, m,
                (sbyte*)scratch,
                (float*)(scratch + qBytes),
                (int*)(scratch + qBytes + blocks16 * sizeof(float)));
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(scratchBytes);
        try
        {
            fixed (byte* scratch = rented)
                MatMulCore(input, output, m,
                    (sbyte*)scratch,
                    (float*)(scratch + qBytes),
                    (int*)(scratch + qBytes + blocks16 * sizeof(float)));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private void MatMulCore(float* input, float* output, int m, sbyte* aq, float* ad, int* aqSum)
    {
        int k = InFeatures, n = OutFeatures, blocksPerRow = k / QBlock;

        if (m >= 16)
        {
            long pin = (long)input, pq = (long)aq, pd = (long)ad, ps = (long)aqSum;
            Parallel.For(0, m, p =>
                QuantizeRow((float*)pin + (long)p * k, k, (sbyte*)pq + (long)p * k,
                    (float*)pd + (long)p * blocksPerRow, (int*)ps + (long)p * blocksPerRow));
        }
        else
        {
            for (int p = 0; p < m; p++)
                QuantizeRow(input + (long)p * k, k, aq + (long)p * k,
                    ad + (long)p * blocksPerRow, aqSum + (long)p * blocksPerRow);
        }

        int chunks = ParallelChunkOverride > 0
            ? Math.Min(_groups, ParallelChunkOverride)
            : (long)m * n * k < MinParallelWork
                ? 1
                : Math.Min(_groups, Environment.ProcessorCount * 2);

        if (chunks <= 1)
        {
            RunGroups(0, _groups, output, aq, ad, aqSum, m);
        }
        else
        {
            int perChunk = (_groups + chunks - 1) / chunks;
            chunks = (_groups + perChunk - 1) / perChunk;
            long po = (long)output, pq = (long)aq, pd = (long)ad, ps = (long)aqSum;
            Parallel.For(0, chunks, c =>
                RunGroups(c * perChunk, Math.Min(_groups, (c + 1) * perChunk),
                    (float*)po, (sbyte*)pq, (float*)pd, (int*)ps, m));
        }

        int tailStart = _groups * RowsPerGroup;
        if (tailStart < n)
        {
            fixed (byte* pRaw = Source)
            {
                for (int row = tailStart; row < n; row++)
                    for (int p = 0; p < m; p++)
                        output[(long)p * n + row] =
                            QuantizationKernels.VecDotQ6K_FMA(input + (long)p * k, pRaw, row, k);
            }
        }
    }

    private void RunGroups(int from, int to, float* output, sbyte* aq, float* ad, int* aqSum, int m)
    {
        int n = OutFeatures, k = InFeatures, blocks = _blocks;
        long groupBytes = (long)blocks * BlockBytes;
        for (int g = from; g < to; g++)
        {
            byte* w = _p + g * groupBytes;
            for (int p = 0; p < m; p++)
            {
                sbyte* q = aq + (long)p * k;
                float* dq = ad + (long)p * (k / QBlock);
                int* sums = aqSum + (long)p * (k / QBlock);
                GroupDot(w, blocks, q, dq, sums, output + (long)p * n + g * RowsPerGroup);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GroupDot(byte* w, int blocks, sbyte* q, float* dq, int* sums, float* out8)
    {
        var acc = Vector256<float>.Zero;
        var ones = Vector256.Create((short)1);

        for (int b = 0; b < blocks; b++, w += BlockBytes, q += SuperBlock, dq += SubBlocks, sums += SubBlocks)
        {
            for (int sub = 0; sub < SubBlocks; sub++)
            {
                byte* subW = w + sub * SubBlockBytes;
                float* scales = (float*)subW;
                byte* quants = subW + ScaleBytes;
                sbyte* subQ = q + sub * QBlock;
                int qSum = sums[sub];
                float d = dq[sub];

                var dot = Vector256<int>.Zero;
                for (int quartet = 0; quartet < QuartetsPerSubBlock; quartet++)
                {
                    var v = Avx.LoadVector256(quants + quartet * RowsPerGroup * 4);
                    var vq = Vector256.Create(
                        Unsafe.ReadUnaligned<int>(subQ + quartet * 4)).AsByte().AsSByte();
                    dot = Avx2.Add(dot,
                        Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(v, vq), ones));
                }

                var scale = Avx.Multiply(Avx.LoadVector256(scales), Vector256.Create(d));
                var bias = Avx.Multiply(scale, Vector256.Create(32f * qSum));
                acc = Fma.MultiplyAdd(scale, Avx.ConvertToVector256Single(dot), acc);
                acc = Avx.Subtract(acc, bias);
            }
        }

        Avx.Store(out8, acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void QuantizeRow(float* x, int inFeatures, sbyte* q, float* d, int* sums)
    {
        var absMask = Vector256.Create(0x7FFFFFFF).AsSingle();
        int blocks32 = inFeatures / 32;
        for (int b = 0; b < blocks32; b++)
        {
            int at = b * 32;
            var v0 = Avx.LoadVector256(x + at);
            var v1 = Avx.LoadVector256(x + at + 8);
            var v2 = Avx.LoadVector256(x + at + 16);
            var v3 = Avx.LoadVector256(x + at + 24);
            var mx = Avx.Max(Avx.Max(Avx.And(v0, absMask), Avx.And(v1, absMask)),
                             Avx.Max(Avx.And(v2, absMask), Avx.And(v3, absMask)));
            var m4 = Sse.Max(mx.GetLower(), mx.GetUpper());
            m4 = Sse.Max(m4, Sse.MoveHighToLow(m4, m4));
            float amax = MathF.Max(m4.ToScalar(), m4.GetElement(1));
            float scale = amax / 127f;
            d[b * 2] = scale;
            d[b * 2 + 1] = scale;
            var id = Vector256.Create(amax >= MinQuantizableMax ? 127f / amax : 0f);
            StoreInt8(q + at, v0, v1, id);
            StoreInt8(q + at + 16, v2, v3, id);

            int sum0 = 0, sum1 = 0;
            for (int i = 0; i < 16; i++) sum0 += q[at + i];
            for (int i = 16; i < 32; i++) sum1 += q[at + i];
            sums[b * 2] = sum0;
            sums[b * 2 + 1] = sum1;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreInt8(sbyte* q, Vector256<float> a, Vector256<float> b, Vector256<float> id)
    {
        var ia = Avx.ConvertToVector256Int32(Avx.Multiply(a, id));
        var ib = Avx.ConvertToVector256Int32(Avx.Multiply(b, id));
        var bytes = Sse2.PackSignedSaturate(
            Sse2.PackSignedSaturate(ia.GetLower(), ia.GetUpper()),
            Sse2.PackSignedSaturate(ib.GetLower(), ib.GetUpper()));
        Sse2.Store(q, bytes);
    }
}
