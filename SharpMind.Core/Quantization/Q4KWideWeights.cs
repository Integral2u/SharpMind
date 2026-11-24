using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Q4_K weights repacked for an integer matmul whose tile is eight output rows wide.
///
/// The layout mirrors <see cref="Q8_0WideWeights"/>: eight consecutive rows are
/// grouped so one activation quartet feeds eight row accumulators. Q4_K adds two
/// per-32-value terms that Q8_0 does not have, so each row stores the already
/// multiplied super-block scale (<c>d * s</c>) and minimum (<c>dmin * m</c>), and
/// the kernel evaluates
///
///   dot = Σ_sub (scale * Σ(v*q) - bias * Σq)
///
/// with <c>v</c> the unsigned 4-bit weight and <c>q</c> the signed int8
/// activation. Nibbles stay packed: a quartet of four values is two bytes per
/// row, so the quant payload is unchanged. The float scale/min pairs are the
/// only growth (about 1.33x the raw Q4_K tensor).
///
/// Technique extends pulsarforge's Q8_0 group-8 repack (MIT,
/// github.com/siris9476/pulsarforge).
/// </summary>
public sealed unsafe class Q4KWideWeights : IDisposable
{
    public const int RowsPerGroup = 8;
    private const int QBlock = 32;
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 144;
    private const int SubBlocks = SuperBlock / QBlock;
    private const int QuartetsPerSubBlock = QBlock / 4;
    private const int ScaleBytes = RowsPerGroup * sizeof(float) * 2;
    private const int QuantBytes = QuartetsPerSubBlock * RowsPerGroup * 2;
    private const int SubBlockBytes = ScaleBytes + QuantBytes;
    private const int BlockBytes = SubBlocks * SubBlockBytes;
    private const long MinParallelWork = 65_536;
    private const int LargeOutputThreshold = 32_768;
    private const int StackScratchBytes = 8 * 1024;
    private const float MinQuantizableMax = 1e-30f;

    /// <summary>Routes Q4_K matmuls through this path where a layer allows it.</summary>
    public static bool Enabled { get; set; } = true;

    /// <summary>Diagnostic A/B knob; 0 uses the production chunk heuristic.</summary>
    public static int ParallelChunkOverride { get; set; }

    private readonly byte[] _data;
    private readonly byte* _p;
    private readonly int _blocks;
    private readonly int _groups;
    private readonly long _repackBytes;
    private int _disposed;

    /// <summary>The raw tensor, retained only when rows past the last group need it.</summary>
    public byte[]? Source { get; }
    public int InFeatures { get; }
    public int OutFeatures { get; }

    public static bool IsSupported(int inFeatures, int outFeatures) =>
        Avx2.IsSupported && Fma.IsSupported && Ssse3.IsSupported &&
        inFeatures >= SuperBlock && inFeatures % SuperBlock == 0 &&
        outFeatures >= RowsPerGroup;

    /// <summary>Repacks <paramref name="raw"/> ([outFeatures, inFeatures] Q4_K).</summary>
    public static Q4KWideWeights? TryCreate(byte[]? raw, int inFeatures, int outFeatures,
        bool keepSource = true)
    {
        if (raw is null || !IsSupported(inFeatures, outFeatures)) return null;

        int blocks = inFeatures / SuperBlock;
        long expected = (long)outFeatures * blocks * RawBlockBytes;
        if (expected != raw.Length) return null;
        long bytes = (long)(outFeatures / RowsPerGroup) * blocks * BlockBytes;
        if (bytes > Array.MaxLength || !WideRepackBudget.TryReserve(bytes)) return null;

        try
        {
            return new Q4KWideWeights(raw, inFeatures, outFeatures, keepSource, bytes);
        }
        catch
        {
            WideRepackBudget.Release(bytes);
            throw;
        }
    }

    /// <summary>Lazy repack keyed on the raw array, rebuilt when that array changes.</summary>
    public sealed class Cache
    {
        private sealed class Slot(byte[] source, Q4KWideWeights? wide)
        {
            public readonly byte[] Source = source;
            public readonly Q4KWideWeights? Wide = wide;
        }

        private readonly Lock _lock = new();
        private Slot? _slot;

        public Q4KWideWeights? Get(byte[] raw, int inFeatures, int outFeatures)
        {
            var slot = _slot;
            if (slot is not null && ReferenceEquals(slot.Source, raw)) return slot.Wide;
            lock (_lock)
            {
                slot = _slot;
                if (slot is null || !ReferenceEquals(slot.Source, raw))
                {
                    slot?.Wide?.Dispose();
                    _slot = slot = new Slot(raw, TryCreate(raw, inFeatures, outFeatures));
                }
                return slot.Wide;
            }
        }

        public void Clear()
        {
            lock (_lock)
            {
                var slot = _slot;
                _slot = null;
                slot?.Wide?.Dispose();
            }
        }

        internal bool HasSlot
        {
            get { lock (_lock) return _slot is not null; }
        }
    }

    private Q4KWideWeights(byte[] raw, int inFeatures, int outFeatures,
        bool keepSource, long repackBytes)
    {
        Source = keepSource ? raw : null;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _repackBytes = repackBytes;
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
        for (int sub = 0; sub < SubBlocks; sub++)
        {
            int chunk = sub / 2;
            int shift = (sub % 2) * 4;
            byte* subDst = dst + sub * SubBlockBytes;
            float* scales = (float*)subDst;
            float* biases = scales + RowsPerGroup;
            byte* quants = subDst + ScaleBytes;

            for (int row = 0; row < RowsPerGroup; row++)
            {
                byte* src = raw + ((long)(group * RowsPerGroup + row) * blocks + block) * RawBlockBytes;
                float d = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(src));
                float dmin = (float)BitConverter.UInt16BitsToHalf(Unsafe.ReadUnaligned<ushort>(src + 2));
                byte* scaleBytes = src + 4;
                byte* qs = src + 16;
                scales[row] = d * QuantizationKernels.GetScaleMinK4_Scale_Scalar(sub, scaleBytes);
                biases[row] = dmin * QuantizationKernels.GetScaleMinK4_Min_Scalar(sub, scaleBytes);

                byte* rowQuants = quants + row * 2;
                byte* chunkQs = qs + chunk * QBlock;
                for (int q = 0; q < QuartetsPerSubBlock; q++)
                {
                    int at = q * 4;
                    int v0 = (chunkQs[at] >> shift) & 0x0F;
                    int v1 = (chunkQs[at + 1] >> shift) & 0x0F;
                    int v2 = (chunkQs[at + 2] >> shift) & 0x0F;
                    int v3 = (chunkQs[at + 3] >> shift) & 0x0F;
                    byte* qDst = rowQuants + q * (RowsPerGroup * 2);
                    qDst[0] = (byte)(v0 | (v1 << 4));
                    qDst[1] = (byte)(v2 | (v3 << 4));
                }
            }
        }
    }

    /// <summary>output[p, r] = row r of the weights · input[p], for p in [0, m).</summary>
    public void MatMul(float* input, float* output, int m)
    {
        int qBytes = checked(m * InFeatures);
        int blockCount = checked(m * (InFeatures / QBlock));
        int scratchBytes = checked(qBytes + blockCount * sizeof(float) + blockCount * sizeof(int));

        if (scratchBytes <= StackScratchBytes)
        {
            byte* scratch = stackalloc byte[scratchBytes];
            MatMulCore(input, output, m,
                (sbyte*)scratch,
                (float*)(scratch + qBytes),
                (int*)(scratch + qBytes + blockCount * sizeof(float)));
            return;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(scratchBytes);
        try
        {
            fixed (byte* scratch = rented)
                MatMulCore(input, output, m,
                    (sbyte*)scratch,
                    (float*)(scratch + qBytes),
                    (int*)(scratch + qBytes + blockCount * sizeof(float)));
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
            byte[] source = Source
                ?? throw new InvalidOperationException(
                    "Q4_K wide weights were created without their raw source but still have tail rows.");
            fixed (byte* pRaw = source)
            {
                for (int row = tailStart; row < n; row++)
                    for (int p = 0; p < m; p++)
                        output[(long)p * n + row] =
                            QuantizationKernels.VecDotQ4K_FMA(input + (long)p * k, pRaw, row, k);
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
        var acc0 = Vector128<float>.Zero;
        var acc1 = Vector128<float>.Zero;
        var ones = Vector128.Create((short)1);
        var mask = Vector128.Create((byte)0x0F);

        for (int b = 0; b < blocks; b++, w += BlockBytes, q += SuperBlock, dq += SubBlocks, sums += SubBlocks)
        {
            for (int sub = 0; sub < SubBlocks; sub++)
            {
                byte* subW = w + sub * SubBlockBytes;
                float* scales = (float*)subW;
                float* biases = scales + RowsPerGroup;
                byte* quants = subW + ScaleBytes;
                sbyte* subQ = q + sub * QBlock;
                int qSum = sums[sub];
                float d = dq[sub];

                var dot0 = Vector128<int>.Zero;
                var dot1 = Vector128<int>.Zero;

                for (int quartet = 0; quartet < QuartetsPerSubBlock; quartet++)
                {
                    var packed = Sse2.LoadVector128(quants + quartet * RowsPerGroup * 2);
                    var lo = Sse2.And(packed, mask);
                    var hi = Sse2.And(
                        Sse2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), mask);
                    var rows03 = Sse2.UnpackLow(lo, hi);
                    var rows47 = Sse2.UnpackHigh(lo, hi);

                    var vq = Vector128.Create(
                        Unsafe.ReadUnaligned<int>(subQ + quartet * 4)).AsByte().AsSByte();
                    dot0 = Sse2.Add(dot0,
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(rows03, vq), ones));
                    dot1 = Sse2.Add(dot1,
                        Sse2.MultiplyAddAdjacent(Ssse3.MultiplyAddAdjacent(rows47, vq), ones));
                }

                var scale0 = Sse2.Multiply(Sse2.LoadVector128(scales), Vector128.Create(d));
                var scale1 = Sse2.Multiply(Sse2.LoadVector128(scales + 4), Vector128.Create(d));
                var bias0 = Sse2.Multiply(Sse2.LoadVector128(biases), Vector128.Create(d * qSum));
                var bias1 = Sse2.Multiply(Sse2.LoadVector128(biases + 4), Vector128.Create(d * qSum));

                acc0 = Fma.MultiplyAdd(scale0, Sse2.ConvertToVector128Single(dot0), acc0);
                acc0 = Sse2.Subtract(acc0, bias0);
                acc1 = Fma.MultiplyAdd(scale1, Sse2.ConvertToVector128Single(dot1), acc1);
                acc1 = Sse2.Subtract(acc1, bias1);
            }
        }

        Sse2.Store(out8, acc0);
        Sse2.Store(out8 + 4, acc1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void QuantizeRow(float* x, int inFeatures, sbyte* q, float* d, int* sums)
    {
        var absMask = Vector256.Create(0x7FFFFFFF).AsSingle();
        int blocks = inFeatures / QBlock;
        for (int b = 0; b < blocks; b++)
        {
            int at = b * QBlock;
            var v0 = Avx.LoadVector256(x + at);
            var v1 = Avx.LoadVector256(x + at + 8);
            var v2 = Avx.LoadVector256(x + at + 16);
            var v3 = Avx.LoadVector256(x + at + 24);
            var mx = Avx.Max(Avx.Max(Avx.And(v0, absMask), Avx.And(v1, absMask)),
                             Avx.Max(Avx.And(v2, absMask), Avx.And(v3, absMask)));
            var m4 = Sse.Max(mx.GetLower(), mx.GetUpper());
            m4 = Sse.Max(m4, Sse.MoveHighToLow(m4, m4));
            float amax = MathF.Max(m4.ToScalar(), m4.GetElement(1));
            d[b] = amax / 127f;
            var id = Vector256.Create(amax >= MinQuantizableMax ? 127f / amax : 0f);
            StoreInt8(q + at, v0, v1, id);
            StoreInt8(q + at + 16, v2, v3, id);

            int sum = 0;
            for (int i = 0; i < QBlock; i++) sum += q[at + i];
            sums[b] = sum;
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

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        WideRepackBudget.Release(_repackBytes);
    }
}
