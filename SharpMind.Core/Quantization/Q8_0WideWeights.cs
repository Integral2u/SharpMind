using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Q8_0 weights repacked for an integer matmul whose tile is eight output rows wide.
///
/// Every 32-column block of eight consecutive rows is stored as 288 bytes: the eight row scales
/// as float32, then the block's eight quartets (four columns each) with the eight rows' four
/// bytes side by side. One 256-bit load therefore holds four columns of eight rows. Broadcasting
/// the activation quartet to all eight lanes, abs/sign + maddubs + madd yields the eight rows'
/// dot products over those four columns: one sign pair per 32 products, where a sign transfer
/// per (row, column) pair grows with the tile and could not widen past 4x1. Activations are
/// quantized to int8 per 32 columns, the scales multiply once per block, and the eight-lane float
/// accumulator is the output — no horizontal sum.
///
/// Work is split by pairs of row groups; each pair sweeps every input row while its sixteen
/// weight rows stay cache-hot, so each activation block is fetched once for both groups.
///
/// Technique from pulsarforge (MIT, github.com/siris9476/pulsarforge: repack_q80_group8_wide,
/// vecdot_q80_wide_group).
/// </summary>
public sealed unsafe class Q8_0WideWeights
{
    public const int RowsPerGroup = 8;
    private const int QBlock = 32;
    private const int RawBlockBytes = 34;
    private const int BlockBytes = RowsPerGroup * sizeof(float) + QBlock * RowsPerGroup;
    private const long MinParallelWork = 65_536;

    /// <summary>Below this block maximum the block quantizes to zero: 127 / max|x| must stay finite.</summary>
    private const float MinQuantizableMax = 1e-30f;

    /// <summary>
    /// Routes Q8_0 matmuls through this path where a layer allows it. False restores the float
    /// kernels, for A/B measurement.
    /// </summary>
    public static bool Enabled { get; set; } = true;

    private readonly byte[] _data;
    private readonly byte* _p;
    private readonly int _blocks;
    private readonly int _groups;

    /// <summary>The raw Q8_0 tensor this was built from; also serves rows past the last full group.</summary>
    public byte[] Source { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public static bool IsSupported(int inFeatures, int outFeatures) =>
        Avx2.IsSupported && Fma.IsSupported &&
        inFeatures >= QBlock && inFeatures % QBlock == 0 && outFeatures >= RowsPerGroup;

    /// <summary>Repacks <paramref name="raw"/> ([outFeatures, inFeatures] Q8_0), or returns null where unsupported.</summary>
    public static Q8_0WideWeights? TryCreate(byte[]? raw, int inFeatures, int outFeatures)
    {
        if (raw is null || !IsSupported(inFeatures, outFeatures)) return null;
        if ((long)outFeatures * (inFeatures / QBlock) * RawBlockBytes != raw.Length) return null;
        if ((long)(outFeatures / RowsPerGroup) * (inFeatures / QBlock) * BlockBytes > Array.MaxLength) return null;
        return new Q8_0WideWeights(raw, inFeatures, outFeatures);
    }

    /// <summary>
    /// The repacked form of one raw tensor, built on first use and rebuilt when the raw array is
    /// replaced. First use rather than load keeps the copy off paths that never run this matmul
    /// (the GPU plugin uploads the raw bytes); the lock stops concurrent first calls from each
    /// building a copy — a vocabulary-sized head is hundreds of MB. The raw array must not be
    /// mutated in place once handed over: the copy would go stale.
    /// </summary>
    public sealed class Cache
    {
        private sealed class Slot(byte[] source, Q8_0WideWeights? wide)
        {
            public readonly byte[] Source = source;
            public readonly Q8_0WideWeights? Wide = wide;
        }

        private readonly Lock _lock = new();
        private Slot? _slot;

        public Q8_0WideWeights? Get(byte[] raw, int inFeatures, int outFeatures)
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

        /// <summary>
        /// Drops the cached repack. The slot roots the raw source array — for a quantized
        /// layer that array is its whole weight payload — as well as the copy, so a streaming
        /// layer freeing its bytes must clear this too, or the "free" keeps the tensor alive
        /// for the lifetime of the layer (and the repack with it).
        /// </summary>
        public void Clear()
        {
            lock (_lock) _slot = null;
        }

        internal bool HasSlot
        {
            get { lock (_lock) return _slot is not null; }
        }
    }

    private Q8_0WideWeights(byte[] raw, int inFeatures, int outFeatures)
    {
        Source = raw;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _blocks = inFeatures / QBlock;
        _groups = outFeatures / RowsPerGroup;
        _data = GC.AllocateUninitializedArray<byte>(_groups * _blocks * BlockBytes, pinned: true);
        _p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(_data));

        int blocks = _blocks;
        long dst = (long)_p;
        fixed (byte* pRaw = raw)
        {
            long src = (long)pRaw;
            Parallel.For(0, _groups, g =>
            {
                byte* d = (byte*)dst + (long)g * blocks * BlockBytes;
                for (int b = 0; b < blocks; b++, d += BlockBytes)
                {
                    for (int i = 0; i < RowsPerGroup; i++)
                    {
                        byte* blk = (byte*)src + ((long)(g * RowsPerGroup + i) * blocks + b) * RawBlockBytes;
                        ((float*)d)[i] = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(blk));
                        for (int t = 0; t < RowsPerGroup; t++)
                            Unsafe.CopyBlockUnaligned(d + 32 + t * 32 + i * 4, blk + 2 + t * 4, 4);
                    }
                }
            });
        }
    }

    /// <summary>output[p, r] = row r of the weights · input[p], for p in [0, m).</summary>
    public void MatMul(float* input, float* output, int m)
    {
        int qBytes = checked(m * InFeatures);
        byte[] scratch = ArrayPool<byte>.Shared.Rent(checked(qBytes + m * _blocks * sizeof(float)));
        try
        {
            fixed (byte* pScratch = scratch)
                MatMulCore(input, output, m, (sbyte*)pScratch, (float*)(pScratch + qBytes));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(scratch);
        }
    }

    private void MatMulCore(float* input, float* output, int m, sbyte* aq, float* ad)
    {
        int k = InFeatures, n = OutFeatures, blocks = _blocks;

        if (m >= 16)
        {
            long pin = (long)input, pq = (long)aq, pd = (long)ad;
            Parallel.For(0, m, p =>
                QuantizeRow((float*)pin + (long)p * k, blocks, (sbyte*)pq + (long)p * k, (float*)pd + (long)p * blocks));
        }
        else
        {
            for (int p = 0; p < m; p++)
                QuantizeRow(input + (long)p * k, blocks, aq + (long)p * k, ad + (long)p * blocks);
        }

        int pairs = (_groups + 1) / 2;
        int chunks = (long)m * n * k < MinParallelWork ? 1 : Math.Min(pairs, Environment.ProcessorCount * 2);
        int perChunk = (pairs + chunks - 1) / chunks;
        chunks = (pairs + perChunk - 1) / perChunk;
        if (chunks <= 1)
        {
            RunPairs(0, pairs, output, aq, ad, m);
        }
        else
        {
            long po = (long)output, pq = (long)aq, pd = (long)ad;
            Parallel.For(0, chunks, c =>
                RunPairs(c * perChunk, Math.Min(pairs, (c + 1) * perChunk), (float*)po, (sbyte*)pq, (float*)pd, m));
        }

        int tailStart = _groups * RowsPerGroup;
        if (tailStart < n)
        {
            fixed (byte* pRaw = Source)
            {
                for (int row = tailStart; row < n; row++)
                    for (int p = 0; p < m; p++)
                        output[(long)p * n + row] = QuantizationKernels.VecDotQ8_0_FMA(input + (long)p * k, pRaw, row, k);
            }
        }
    }

    private void RunPairs(int from, int to, float* output, sbyte* aq, float* ad, int m)
    {
        int n = OutFeatures, k = InFeatures, blocks = _blocks;
        long groupBytes = (long)blocks * BlockBytes;
        for (int pair = from; pair < to; pair++)
        {
            int g0 = pair * 2, g1 = g0 + 1;
            byte* w0 = _p + g0 * groupBytes;
            bool both = g1 < _groups;
            for (int p = 0; p < m; p++)
            {
                sbyte* q = aq + (long)p * k;
                float* d = ad + (long)p * blocks;
                float* o = output + (long)p * n;
                GroupDot(w0, blocks, q, d, o + g0 * RowsPerGroup);
                if (both)
                    GroupDot(w0 + groupBytes, blocks, q, d, o + g1 * RowsPerGroup);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GroupDot(byte* w, int blocks, sbyte* q, float* d, float* out8)
    {
        var acc = Vector256<float>.Zero;
        var ones = Vector256.Create((short)1);
        for (int b = 0; b < blocks; b++, w += BlockBytes, q += QBlock)
        {
            var dot = Vector256<int>.Zero;
            sbyte* wq = (sbyte*)w + 32;
            for (int t = 0; t < QBlock; t += 4)
            {
                var vw = Avx.LoadVector256(wq + t * 8);
                var vx = Vector256.Create(Unsafe.ReadUnaligned<int>(q + t)).AsSByte();
                var pairsum = Avx2.MultiplyAddAdjacent(Avx2.Abs(vw), Avx2.Sign(vx, vw));
                dot = Avx2.Add(dot, Avx2.MultiplyAddAdjacent(pairsum, ones));
            }
            var scale = Avx.Multiply(Avx.LoadVector256((float*)w), Vector256.Create(d[b]));
            acc = Fma.MultiplyAdd(scale, Avx.ConvertToVector256Single(dot), acc);
        }
        Avx.Store(out8, acc);
    }

    /// <summary>
    /// Symmetric int8 per 32 values: d = max|x| / 127, q = round-half-even(x · 127 / max|x|).
    /// A block whose maximum is below <see cref="MinQuantizableMax"/> quantizes to zero, so q never
    /// leaves [-127, 127] (an overflowing 1/d would saturate to -128, which sign transfer cannot negate).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void QuantizeRow(float* x, int blocks, sbyte* q, float* d)
    {
        var absMask = Vector256.Create(0x7FFFFFFF).AsSingle();
        for (int b = 0; b < blocks; b++, x += QBlock, q += QBlock)
        {
            var v0 = Avx.LoadVector256(x);
            var v1 = Avx.LoadVector256(x + 8);
            var v2 = Avx.LoadVector256(x + 16);
            var v3 = Avx.LoadVector256(x + 24);
            var mx = Avx.Max(Avx.Max(Avx.And(v0, absMask), Avx.And(v1, absMask)),
                             Avx.Max(Avx.And(v2, absMask), Avx.And(v3, absMask)));
            var m4 = Sse.Max(mx.GetLower(), mx.GetUpper());
            m4 = Sse.Max(m4, Sse.MoveHighToLow(m4, m4));
            float amax = MathF.Max(m4.ToScalar(), m4.GetElement(1));
            d[b] = amax / 127f;
            var id = Vector256.Create(amax >= MinQuantizableMax ? 127f / amax : 0f);
            StoreInt8(q, v0, v1, id);
            StoreInt8(q + 16, v2, v3, id);
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
