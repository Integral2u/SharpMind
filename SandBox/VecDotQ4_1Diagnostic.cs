using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;

namespace SandBox;

public static class VecDotQ4_1Diagnostic
{
    private const int QK = 32;
    private const int BLOCK_BYTES = 20;

    public static void Run()
    {
        Console.Error.WriteLine("=== Q4_1 VecDot Diagnostic ===\n");

        bool hasAvx2 = System.Runtime.Intrinsics.X86.Avx2.IsSupported;
        Console.Error.WriteLine($"  AVX2 available: {hasAvx2}\n");

        // --- Part 1: Synthetic tests ---
        var configs = new (int Out, int In, string Label)[]
        {
            (32, 32,   "Square 32x32"),
            (32, 64,   "Wide 32x64"),
            (32, 96,   "Wide 32x96"),
            (1024, 3072, "Gate 1024x3072"),
            (1024, 1024, "Square 1024x1024"),
            (576, 1536, "SmolLM gate 576x1536"),
        };

        foreach (var (outDim, inDim, label) in configs)
        {
            Console.Error.WriteLine($"\n--- {label} ---");
            int nBlk = (inDim + QK - 1) / QK;
            Console.Error.WriteLine($"  out={outDim} in={inDim} blk/row={nBlk}");

            var (raw, deq) = SyntheticQ4_1(outDim, inDim);
            float[] inp = RandInput(inDim, 42);
            int sFail = 0, aFail = 0, vDiff = 0;

            unsafe
            {
                fixed (float* pI = inp) fixed (byte* pR = raw) fixed (float* pD = deq)
                {
                    for (int c = 0; c < outDim; c++)
                    {
                        float s = QuantizationKernels.VecDotQ4_1_Scalar(pI, pR, c, inDim);
                        float a = hasAvx2 ? QuantizationKernels.VecDotQ4_1_AVX2(pI, pR, c, inDim) : s;
                        double dot = 0;
                        for (int i = 0; i < inDim; i++) dot += pI[i] * pD[(long)c * inDim + i];
                        if (Math.Abs(s - (float)dot) > 0.05f) sFail++;
                        if (Math.Abs(a - (float)dot) > 0.05f) aFail++;
                        if (Math.Abs(s - a) > 0.001f) vDiff++;
                    }
                }
            }

            Console.Error.WriteLine($"  Scalar err>{outDim - sFail}/{outDim}  AVX2 err>{outDim - aFail}/{outDim}  S==A>{outDim - vDiff}/{outDim}");
            Console.Error.WriteLine($"  {(sFail == 0 && (!hasAvx2 || aFail == 0) ? "PASS" : "FAIL")}");
        }

        // --- Part 2: Real model data test ---
        Console.Error.WriteLine("\n\n=== Real Q4_1 Model Data ===");
        string modelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\Qwen3-0.6B-Q4_1.gguf";
        if (!System.IO.File.Exists(modelPath)) { Console.Error.WriteLine("  Model not found"); }
        else
        {
            GgufLoaderFactory.Default.Load(modelPath, null, out _, out var mc, out _);
            System.GC.Collect();
            using var w = GgufLoaderFactory.Default.LoadWeightsToTransformerWeights(modelPath, mc);
            Console.Error.WriteLine($"  Blocks: {w.Blocks?.Length ?? 0}");

            if (w.Blocks is { Length: > 0 })
            {
                var b0 = w.Blocks[0];
                foreach (var (raw, flt, name) in new (byte[]?, Tensor<float>?, string)[] {
                    (b0.RawWgate, b0.Wf1, "Wgate"), (b0.RawWq, b0.Wq, "Wq"),
                    (b0.RawWk, b0.Wk, "Wk"), (b0.RawWv, b0.Wv, "Wv"),
                    (b0.RawWo, b0.Wo, "Wo"), (b0.RawWf1, b0.Wf1, "Wf1"),
                    (b0.RawWf2, b0.Wf2, "Wf2"), (b0.RawWup, b0.Wf1, "Wup"),
                })
                {
                    if (raw == null || flt == null) continue;
                    int inDim = flt.Shape[1], outDim = flt.Shape[0], nBlk = (inDim + QK - 1) / QK;
                    Console.Error.WriteLine($"\n  {name}: [{outDim},{inDim}] raw={raw.Length}B");

                    unsafe
                    {
                        fixed (byte* pR = raw)
                        {
                            for (int b = 0; b < Math.Min(2, nBlk); b++)
                            {
                                byte* bp = pR + b * BLOCK_BYTES;
                                float d = QuantizationKernels.HalfToFloat_Scalar(*(ushort*)bp);
                                float m = QuantizationKernels.HalfToFloat_Scalar(*(ushort*)(bp + 2));
                                Console.Error.WriteLine($"    Blk{b}: d={d:G6} m={m:G6}  raw={string.Join("", Enumerable.Range(0, BLOCK_BYTES).Select(x => bp[x].ToString("X2")))}");
                            }
                        }
                    }

                    // VecDot vs float dot
                    float[] inp = RandInput(inDim, 42);
                    float[] deq = DequantAll(raw, outDim, inDim);
                    int mis = 0;
                    unsafe
                    {
                        fixed (float* pI = inp) fixed (byte* pR = raw) fixed (float* pD = deq)
                        {
                            for (int c = 0; c < Math.Min(20, outDim); c++)
                            {
                                float s = QuantizationKernels.VecDotQ4_1_Scalar(pI, pR, c, inDim);
                                float a = hasAvx2 ? QuantizationKernels.VecDotQ4_1_AVX2(pI, pR, c, inDim) : s;
                                double dot = 0;
                                for (int i = 0; i < inDim; i++) dot += pI[i] * pD[(long)c * inDim + i];
                                float e1 = Math.Abs(s - (float)dot);
                                float e2 = Math.Abs(a - (float)dot);
                                if (e1 > 0.05f || e2 > 0.05f) { mis++; if (mis <= 3) Console.Error.WriteLine($"  col={c}: S={s:F6}(err={e1:F6}) A={a:F6}(err={e2:F6})"); }
                            }
                        }
                    }
                    Console.Error.WriteLine($"  VecDot err > 5%: {mis}/{Math.Min(20, outDim)}");
                    break;
                }
            }
        }

        Console.Error.WriteLine("\n=== Done ===");
    }

    private static (byte[] Raw, float[] Deq) SyntheticQ4_1(int outDim, int inDim)
    {
        var rng = new Random(42);
        float[] f = new float[outDim * inDim];
        for (int i = 0; i < f.Length; i++) f[i] = (float)(rng.NextDouble() * 4 - 2);

        int nBlk = (inDim + QK - 1) / QK;
        byte[] raw = new byte[outDim * nBlk * BLOCK_BYTES];
        unsafe
        {
            fixed (byte* pR = raw) fixed (float* pF = f)
            {
                for (int o = 0; o < outDim; o++)
                {
                    for (int b = 0; b < nBlk; b++)
                    {
                        int start = b * QK, end = Math.Min(start + QK, inDim);
                        float min = float.MaxValue, max = float.MinValue;
                        for (int j = start; j < end; j++) { float v = pF[o * inDim + j]; if (v < min) min = v; if (v > max) max = v; }
                        float d = (max - min) / 15f; if (d == 0) d = 1f;
                        byte* bp = pR + (long)o * nBlk * BLOCK_BYTES + b * BLOCK_BYTES;
                        *(ushort*)bp = FloatToHalf(d);
                        *(ushort*)(bp + 2) = FloatToHalf(min);
                        for (int j = start; j < end; j++)
                        {
                            int q = (int)Math.Clamp(MathF.Round((pF[o * inDim + j] - min) / d), 0, 15);
                            int idx = (j - start) / 2, sh = ((j - start) % 2) * 4;
                            bp[4 + idx] = (byte)((bp[4 + idx] & ~(0x0F << sh)) | (q << sh));
                        }
                    }
                }
            }
        }

        float[] deq = DequantAll(raw, outDim, inDim);
        return (raw, deq);
    }

    private static float[] DequantAll(byte[] raw, int outDim, int inDim)
    {
        int nBlk = (inDim + QK - 1) / QK;
        float[] r = new float[outDim * inDim];
        unsafe
        {
            fixed (byte* pR = raw) fixed (float* pD = r)
            {
                for (int o = 0; o < outDim; o++)
                    for (int b = 0; b < nBlk; b++)
                    {
                        byte* bp = pR + ((long)o * nBlk + b) * BLOCK_BYTES;
                        float d = QuantizationKernels.HalfToFloat_Scalar(*(ushort*)bp);
                        float m = QuantizationKernels.HalfToFloat_Scalar(*(ushort*)(bp + 2));
                        for (int j = 0; j < QK && b * QK + j < inDim; j++)
                        {
                            int q = (bp[4 + j / 2] >> ((j & 1) * 4)) & 0x0F;
                            pD[(long)o * inDim + b * QK + j] = m + q * d;
                        }
                    }
            }
        }
        return r;
    }

    // Fused weight test: simulates FFnLayer.SetWeights' RawWgate + RawWup concatenation
    public static void RunFusedTest()
    {
        Console.Error.WriteLine("\n=== Fused Weight Indexing Test ===\n");
        // Small model: hiddenDim=3, ffnDim=6
        int hd = 3, fd = 6;
        int gateOut = hd, gateIn = fd;   // GGUF shape [hd, fd]
        int upOut = hd, upIn = fd;
        int fusedOutF = 2 * fd;          // LinearLayer OutFeatures

        // Create gate weight [hd, fd] with known values
        float[] gateF = new float[hd * fd];
        float[] upF = new float[hd * fd];
        for (int r = 0; r < hd; r++)
            for (int c = 0; c < fd; c++)
            {
                gateF[r * fd + c] = r * 10f + c;  // Row r has values r*10+0, r*10+1, ...
                upF[r * fd + c] = r * 10f + c + 100f;
            }

        Console.Error.WriteLine($"  Gate row 0: {string.Join(", ", gateF[..fd])}");
        Console.Error.WriteLine($"  Gate row 1: {string.Join(", ", gateF[fd..(2*fd)])}");
        Console.Error.WriteLine($"  Up   row 0: {string.Join(", ", upF[..fd])}");
        Console.Error.WriteLine($"  Up   row 1: {string.Join(", ", upF[fd..(2*fd)])}");

        // Quantize gate and up separately (GGUF row-major)
        byte[] rawGate = Q4_1Quantize(gateF);
        byte[] rawUp = Q4_1Quantize(upF);
        Console.Error.WriteLine($"  Raw gate bytes: {rawGate.Length}, raw up bytes: {rawUp.Length}");

        // Fuse: concatenate raw gate + raw up (same as FFnLayer.SetWeights)
        byte[] fused = new byte[rawGate.Length + rawUp.Length];
        Buffer.BlockCopy(rawGate, 0, fused, 0, rawGate.Length);
        Buffer.BlockCopy(rawUp, 0, fused, rawGate.Length, rawUp.Length);
        Console.Error.WriteLine($"  Fused bytes: {fused.Length}");

        // Input vector (length = hiddenDim)
        float[] input = new float[hd];
        var rng = new Random(42);
        for (int i = 0; i < hd; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        // Expected correct results:
        // For LinearLayer with InFeatures=hd=3, OutFeatures=2*fd=12:
        // output[c] = sum_i input[i] * weight[c][i] for i=0..2
        // where weight[c] is row c of the fused weight (In GGUF row-major format)
        //
        // But the fused raw data has rows of fd=6 elements each (the gate/up row length).
        // The VecDot expected row has hd=3 elements.
        // Since 6 != 3, there's a mismatch.

        Console.Error.WriteLine($"\n  Input: {string.Join(", ", input.Select(f => $"{f:F4}"))}");
        Console.Error.WriteLine($"\n  Comparing VecDotQ4_1 vs correct float dot for first few fused outputs:");

        // Dequantize fused data for correct float dot computation
        float[] deqFused = DequantAll(fused, fusedOutF, hd);

        Console.Error.WriteLine($"  DeqFused length: {deqFused.Length}");
        Console.Error.WriteLine($"  Expected: {fusedOutF} outputs x {hd} inputs = {fusedOutF * hd}");

        // Also dequantize in the WRONG way (treating each GGUF row as one output)
        // This is what VecDot effectively does
        Console.Error.WriteLine();
        Console.Error.WriteLine("  col | VecDot | Correct | GateExpected | UpExpected | VecDot Gate | VecDot Up");

        unsafe
        {
            fixed (float* pIn = input)
            fixed (byte* pFused = fused)
            fixed (float* pDeq = deqFused)
            {
                for (int c = 0; c < Math.Min(8, fusedOutF); c++)
                {
                    float vd = QuantizationKernels.VecDotQ4_1_Scalar(pIn, pFused, c, hd);

                    // Correct: output c = sum_{i=0}^{hd-1} input[i] * weight[c][i]
                    // weight[c] is the c-th row of the full weight matrix (in SharpMind convention)
                    // In GGUF data, weight[c] for c < hd is in gate row c
                    // For c < fd (gate outputs):
                    //   weight[c] should equal gate[c][0..hd-1] (but gate[c] has fd elements!)
                    //   Actually, weight[c] = gate[c][0..hd-1] for hd < fd, which is a partial row
                    // This is the transpose of what GGUF stores!
                    // 
                    // The CORRECT weight for VecDot is the transposed matrix.
                    // But the raw data is in GGUF layout (NOT transposed).
                    // So VecDot reading raw GGUF rows incorrectly indexes the data.
                    double correct = 0;
                    for (int i = 0; i < hd; i++)
                        correct += pIn[i] * pDeq[c * hd + i];

                    // Expected gate contribution: gate row (c % hd) dotted with input
                    // But with ALL fd elements of the row... hmm.
                    // Actually, the gate weight for output c is the c-th row of the gate matrix.
                    // In GGUF: gate[c / (fd/hd)] and within that row, elements [c % (fd/hd) * hd ..]

                    // For a proper check, compute what VecDot SHOULD read:
                    // Each VecDot call reads hd elements starting at c * ceil(hd/32) * 34 bytes offset
                    // which maps to... complicated.

                    // The key: does VecDot give the same as the dequantized fused row c truncated to hd?
                    // If the data is stored as [OutGGUF, InGGUF] and VecDot expects [OutLinear, InLinear],
                    // these only match if OutGGUF = OutLinear and InGGUF = InLinear.

                    // For the fused case:
                    // GGUF rows: 2*hd rows of fd elements each (divided into gate and up groups)
                    // Linear expects: 2*fd rows of hd elements each
                    // They DON'T match!

                    Console.Error.WriteLine($"  {c,3} | {vd,7:F3} | {correct,7:F3}");
                }
            }
        }

        Console.Error.WriteLine("\n=== Fused test complete ===");
    }

    private static byte[] Q4_1Quantize(float[] rowMajorData)
    {
        int n = rowMajorData.Length;
        int nBlk = (n + QK - 1) / QK;
        byte[] r = new byte[nBlk * BLOCK_BYTES];
        unsafe
        {
            fixed (byte* pR = r) fixed (float* pD = rowMajorData)
            {
                for (int b = 0; b < nBlk; b++)
                {
                    int start = b * QK, end = Math.Min(start + QK, n);
                    float mn = float.MaxValue, mx = float.MinValue;
                    for (int i = start; i < end; i++) { float v = pD[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
                    float d = (mx - mn) / 15f; if (d == 0) d = 1f;
                    *(ushort*)(pR + b * BLOCK_BYTES) = FloatToHalf(d);
                    *(ushort*)(pR + b * BLOCK_BYTES + 2) = FloatToHalf(mn);
                    for (int i = start; i < end; i++)
                    {
                        int q = (int)Math.Clamp(MathF.Round((pD[i] - mn) / d), 0, 15);
                        int idx = (i - start) / 2, sh = ((i - start) % 2) * 4;
                        pR[b * BLOCK_BYTES + 4 + idx] = (byte)((pR[b * BLOCK_BYTES + 4 + idx] & ~(0x0F << sh)) | (q << sh));
                    }
                }
            }
        }
        return r;
    }

    private static float[] RandInput(int n, int seed)
    {
        var rng = new Random(seed);
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(rng.NextDouble() * 2 - 1);
        return r;
    }

    private static unsafe ushort FloatToHalf(float f)
    {
        uint bits = *(uint*)&f;
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = (bits >> 13) & 0x3FF;
        if (exp <= 0) return (ushort)(sign | mant >> 1);
        if (exp > 31) return (ushort)(sign | 0x7C00 | (mant != 0 ? 1u : 0));
        return (ushort)(sign | (uint)exp << 10 | mant);
    }
}
