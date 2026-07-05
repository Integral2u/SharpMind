using SharpMind;
using SharpMind.Model.Format;
using SharpMind.Model.Config;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Core.Quantization;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;

string basePath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

// Helper: get block size and bytes per block for a quantized dtype
(int blockSize, int bytesPerBlock) GetBlockInfo(GgufDtype dtype) => dtype switch
{
    GgufDtype.Q2_K or GgufDtype.Q2_K_S => (256, 84),
    GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => (256, 110),
    GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => (256, 144),
    GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => (256, 176),
    GgufDtype.Q6_K or GgufDtype.Q6_K_S => (256, 210),
    GgufDtype.Q8_K => (256, 292),
    GgufDtype.Q8_0 => (32, 34),
    GgufDtype.Q8_1 => (32, 36),
    GgufDtype.Q5_0 => (32, 22),
    GgufDtype.Q5_1 => (32, 24),
    GgufDtype.Q4_0 => (32, 18),
    GgufDtype.IQ4_NL => (32, 18),
    GgufDtype.Q4_1 => (32, 20),
    _ => (0, 0)
};

long RowMajorBytes(int[] shape, GgufDtype dtype)
{
    var (bs, bpb) = GetBlockInfo(dtype);
    if (bs == 0 || shape.Length < 2) return 0;
    return (long)((shape[0] + bs - 1) / bs) * shape[1] * bpb;
}

long FlatBytes(int[] shape, GgufDtype dtype)
{
    var (bs, bpb) = GetBlockInfo(dtype);
    if (bs == 0) return 0;
    long total = 1; foreach (var d in shape) total *= d;
    return ((total + bs - 1) / bs) * bpb;
}

string[] testPaths = [
    Path.Combine(basePath, "qwen2-0_5b-instruct-q8_0.gguf"),
    Path.Combine(basePath, "qwen2-0_5b-instruct-q4_k_m.gguf"),
    Path.Combine(basePath, "qwen2-0.5b-instruct-q2_k.gguf"),
    Path.Combine(basePath, "Qwen2-0.5B.Q6_K.gguf"),
];

foreach (var path in testPaths)
{
    if (!File.Exists(path)) { Console.Error.WriteLine($"SKIP: {path}"); continue; }
    var meta = GgufLoader.LoadMeta(path);
    Console.Error.WriteLine($"\n=== {Path.GetFileName(path)} ===");

    // Check all K-quant tensors for size match
    var tensors = meta.Tensors.Where(t => t.Shape.Length >= 2).ToList();
    var sorted = tensors.OrderBy(t => t.Offset).ToList();
    
    for (int i = 0; i < sorted.Count - 1; i++)
    {
        var t = sorted[i];
        var next = sorted[i + 1];
        long actualSize = next.Offset - t.Offset;
        long rowMaj = RowMajorBytes(t.Shape, t.Dtype);
        long flat = FlatBytes(t.Shape, t.Dtype);
        
        bool isKQuant = t.Dtype is GgufDtype.Q2_K or GgufDtype.Q2_K_S or GgufDtype.Q3_K or GgufDtype.Q3_K_S
            or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L or GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M
            or GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M or GgufDtype.Q6_K or GgufDtype.Q6_K_S
            or GgufDtype.Q8_K;

        if (isKQuant || actualSize != rowMaj)
        {
            string match = actualSize == rowMaj ? "RowMajor" : actualSize == flat ? "Flat" : "NEITHER";
            Console.Error.WriteLine($"  {t.Name}: dtype={t.Dtype} shape=[{string.Join(",", t.Shape)}]  actual={actualSize}  rowMajor={rowMaj}  flat={flat}  match={match}");
        }
    }
}
// ── Direct Q5_0 block-level dequant diagnostic ──
// Compare VecDotQ5_0_Scalar output per-block with manual dequant for same raw data.
Console.Error.WriteLine("\n=== Q5_0 BLOCK DIAGNOSTIC ===");
string q5Path2 = Path.Combine(basePath, "qwen2-0_5b-instruct-q4_k_m.gguf");
if (File.Exists(q5Path2))
{
    var meta2 = GgufLoader.LoadMeta(q5Path2);
    ModelConfig c2 = GgufLoader.LoadConfig(meta2)!;
    // Find first Q5_0 tensor in block 0
    var ti2 = meta2.Tensors.FirstOrDefault(t => 
        t.Name.StartsWith("blk.0.", StringComparison.OrdinalIgnoreCase) && t.Dtype == GgufDtype.Q5_0);
    if (ti2.Name == null)
    {
        Console.Error.WriteLine("  No Q5_0 tensor found. Listing available tensors:");
        foreach (var t in meta2.Tensors.Where(t => t.Name.StartsWith("blk.0.", StringComparison.OrdinalIgnoreCase)))
            Console.Error.WriteLine($"    {t.Name}: dtype={t.Dtype} shape=[{string.Join(",", t.Shape)}]");
    }
    else
    {
        Console.Error.WriteLine($"  Using: {ti2.Name} shape=[{string.Join(",", ti2.Shape)}] dtype={ti2.Dtype}  HiddenDim={c2.HiddenDim}");
        using var mmf2 = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
            q5Path2, FileMode.Open, null, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
        using var stream2 = mmf2.CreateViewStream(0, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
        using var reader2 = new BinaryReader(stream2);
        stream2.Position = meta2.DataOffset + ti2.Offset;
        
        int outF2 = (int)ti2.Shape[0];
        int inF2 = (int)ti2.Shape[1];
        int count2 = outF2 * inF2;
        int nBlocks2 = (count2 + 31) / 32;
        int rawSize2 = nBlocks2 * 22;
        byte[] rawData2 = new byte[rawSize2];
        stream2.ReadExactly(rawData2, 0, rawSize2);
        
        stream2.Position = meta2.DataOffset + ti2.Offset;
        float[] dequantFloat2 = new float[count2];
        byte[] blockBuf2 = new byte[22];
        unsafe
        {
            for (int b = 0; b < nBlocks2; b++)
            {
                stream2.ReadExactly(blockBuf2, 0, 22);
                fixed (byte* pBlock2 = blockBuf2)
                {
                    float[] blockDeq2 = GgufLoader.ReadBlock(pBlock2, "Q5_0", 32);
                    int blockStart2 = b * 32;
                    int valid2 = Math.Min(32, count2 - blockStart2);
                    Array.Copy(blockDeq2, 0, dequantFloat2, blockStart2, valid2);
                }
            }
        }
        
        Random rng2 = new(42);
        float[] input2 = new float[c2.HiddenDim];
        for (int i = 0; i < c2.HiddenDim; i++) input2[i] = (float)(rng2.NextDouble() * 2 - 1);
        
        Console.Error.WriteLine($"  Tensor: {ti2.Name} [{string.Join(",", ti2.Shape)}] dtype={ti2.Dtype}");
        Console.Error.WriteLine($"  Shape[0]={outF2} Shape[1]={inF2}  total_elems={count2}");
        
        // Compare VecDot with manual dequant for first few columns
        unsafe
        {
            fixed (float* pInput2 = input2)
            fixed (byte* pRaw2 = rawData2)
            {
                // input2 has c2.HiddenDim elements (typically 896).
                // VecDot must never read past input2 length:
                int safeInF = Math.Min(c2.HiddenDim, inF2);
                int safeOutF = Math.Min(c2.HiddenDim, outF2);
                
                for (int col = 0; col < Math.Min(5, outF2); col++)
                {
                    // Option A: VecDot with inFeatures = inF2 (if safe, else skip to avoid overrun)
                    float vecDotA = safeInF == inF2
                        ? QuantizationKernels.VecDotQ5_0_Scalar(pInput2, pRaw2, col, inF2)
                        : float.NaN;
                    
                    // Option B: VecDot with inFeatures = outF2 (= HiddenDim for attention layers)  
                    float vecDotB = safeOutF == outF2
                        ? QuantizationKernels.VecDotQ5_0_Scalar(pInput2, pRaw2, col, outF2)
                        : float.NaN;
                    
                    // dequantFloat2 is populated by ReadQ5_0 (block-sequential).
                    // For VecDot with inFeatures = K, the raw weights at col K are:
                    //   flat index = col * ceil(K/32)*32 + b*32 + e = col * K_pad + b*32 + e
                    // Since K_pad = K (K is multiple of 32 for 896), flat index = col * K + b*32 + e.
                    // In dequantFloat2, the same position is dequantFloat2[col * K + b*32 + e].
                    
                    // VecDot(inFeatures=outF2) reads outF2 elements starting at col * outF2:
                    double expectedDot = 0;
                    for (int i = 0; i < outF2; i++)
                        expectedDot += input2[i] * dequantFloat2[col * outF2 + i];
                    
                    // [In, Out] correct: output col uses scattered positions i*OutDim + col:
                    double expectedInOut = 0;
                    for (int i = 0; i < outF2; i++)
                        expectedInOut += input2[i] * dequantFloat2[i * inF2 + col];
                    
                    Console.Error.WriteLine($"  col={col}:");
                    Console.Error.WriteLine($"    vecDot(inFeatures=inF2={inF2})={vecDotA:G10}");
                    Console.Error.WriteLine($"    vecDot(inFeatures=outF2={outF2})={vecDotB:G10}");
                    Console.Error.WriteLine($"    expectedDot(VecDot match)={expectedDot:G10}");
                    Console.Error.WriteLine($"    expectedInOut([In,Out])  ={expectedInOut:G10}");
                    double dDot = !float.IsNaN(vecDotB) ? Math.Abs(vecDotB - expectedDot) : double.NaN;
                    double dInOut = !float.IsNaN(vecDotB) ? Math.Abs(vecDotB - expectedInOut) : double.NaN;
                    Console.Error.WriteLine($"    |vecDotB - expDot|={dDot:G4}  |vecDotB - expInOut|={dInOut:G4}");
                }
                
                // ── Direct in-memory comparison: ReadQ5_0 vs VecDotQ5_0 over ALL columns ──
                Console.Error.WriteLine($"\n  === Direct in-memory: ReadQ5_0 dequant vs VecDotQ5_0 ===");
                int nTestCols2 = Math.Min(outF2, 200);
                double totalDiff = 0, maxDiff = 0;
                double totalDiff2 = 0, maxDiff2 = 0; // second formula (In,Out)
                for (int col = 0; col < nTestCols2; col++)
                {
                    // Simulate float path: Σ input[i] * dequant[col * outF2 + i]  (VecDot-match formula)
                    double floatDot = 0;
                    for (int i = 0; i < outF2; i++)
                        floatDot += input2[i] * dequantFloat2[col * outF2 + i];
                    
                    // Simulate [In,Out] formula: Σ input[i] * dequant[i * inF2 + col]
                    double floatInOut = 0;
                    for (int i = 0; i < outF2; i++)
                        floatInOut += input2[i] * dequantFloat2[i * inF2 + col];
                    
                    float vecDot = QuantizationKernels.VecDotQ5_0_Scalar(pInput2, pRaw2, col, outF2);
                    
                    double d1 = Math.Abs(vecDot - floatDot);
                    double d2 = Math.Abs(vecDot - floatInOut);
                    totalDiff += d1; if (d1 > maxDiff) maxDiff = d1;
                    totalDiff2 += d2; if (d2 > maxDiff2) maxDiff2 = d2;
                }
                Console.Error.WriteLine($"  VecDot vs dequant[col*outF2+i] (outF2={outF2}): avg={totalDiff / nTestCols2:G6} max={maxDiff:G6}");
                Console.Error.WriteLine($"  VecDot vs dequant[i*inF2+col] (inF2={inF2}): avg={totalDiff2 / nTestCols2:G6} max={maxDiff2:G6}");
                
                // Now test with the SAME data through QuantizedMatMulQ5_0_AVX2 vs ReadQ5_0
                Console.Error.WriteLine($"\n  === QuantizedMatMul vs manual float dot ===");
                double* matMulResults = stackalloc double[nTestCols2];
                for (int ci = 0; ci < nTestCols2; ci++) matMulResults[ci] = 0;
                
                // Manual matmul: for each block, dequant and accumulate
                int nB = (outF2 + 31) / 32;
                for (int b = 0; b < nB; b++)
                {
                    int be = Math.Min(32, outF2 - b * 32);
                    for (int ci = 0; ci < nTestCols2; ci++)
                    {
                        byte* block = pRaw2 + (long)ci * nB * 22 + b * 22;
                        float d = QuantizationKernels.HalfToFloat_Scalar(*(ushort*)block);
                        uint qh = *(uint*)(block + 2);
                        byte* qs = block + 6;
                        double s = 0;
                        for (int i = 0; i < be; i++)
                        {
                            int h4 = ((int)(qh >> i) & 1) << 4;
                            int nib = ((i & 1) == 0) ? (qs[i / 2] & 0x0F) : (qs[i / 2] >> 4);
                            s += pInput2[b * 32 + i] * (((nib | h4) - 16) * d);
                        }
                        matMulResults[ci] += s;
                    }
                }
                totalDiff = 0; maxDiff = 0;
                for (int ci = 0; ci < nTestCols2; ci++)
                {
                    float vq = QuantizationKernels.VecDotQ5_0_Scalar(pInput2, pRaw2, ci, outF2);
                    double d = Math.Abs(vq - (float)matMulResults[ci]);
                    totalDiff += d; if (d > maxDiff) maxDiff = d;
                }
                Console.Error.WriteLine($"  VecDot vs manual block-loop (scalar): avg={totalDiff / nTestCols2:G6} max={maxDiff:G6}");
            }
        }
    }
}
else Console.Error.WriteLine($"SKIP: {q5Path2}");

// ── Compare quantized forward vs float dequantized forward ──
Console.Error.WriteLine("\n=== Q4_K_M: QUANTIZED vs DEQUANTIZED (Full mode) ===");
if (File.Exists(q5Path2))
{
    var meta = GgufLoader.LoadMeta(q5Path2);
    ModelConfig c = GgufLoader.LoadConfig(meta)!;
    var sc = c.ForModel(HardwareTier.AVX2);
    var w = GgufLoader.LoadWeightsToTransformerWeights(q5Path2, c, null, LoadMode.Full);
    using var model = ModelFactory.CreateSession(w, sc, null, null, false);
    var block = model.GetBlock(0)!;
    
    using var x = new Tensor<float>(1, c.HiddenDim);
    var rng = new Random(42);
    for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);
    
    var layers = new[] {
        ("attn_q", block.Attention.Wq),
        ("attn_k", block.Attention.Wk),
        ("attn_v", block.Attention.Wv),
        ("attn_output", block.Attention.Wo),
    };
    
    foreach (var (name, layer) in layers)
    {
        var savedRaw = layer.RawQuantizedData;
        var savedDtype = layer.QuantDtype;
        layer.RawQuantizedData = null;
        using var yf = layer.Forward(x, model.Ops);
        layer.RawQuantizedData = savedRaw;
        layer.QuantDtype = savedDtype;
        using var yq = layer.Forward(x, model.Ops);
        
        double diff = 0, maxDiff = 0;
        for (int i = 0; i < yf.ElementCount; i++)
        {
            double d = Math.Abs(yf.Data[i] - yq.Data[i]);
            diff += d; if (d > maxDiff) maxDiff = d;
        }
        double avg = diff / yf.ElementCount;
        double norm = 0;
        for (int i = 0; i < yf.ElementCount; i++) norm += Math.Abs(yf.Data[i]);
        norm /= yf.ElementCount;
        Console.Error.WriteLine($"  {name} ({layer.InFeatures}x{layer.OutFeatures} dtype={layer.QuantDtype}): avg={avg:G6} max={maxDiff:G6} avg|yf|={norm:G6}");
        Console.Error.WriteLine($"    [0] float={yf.Data[0]:G8} quant={yq.Data[0]:G8} d={Math.Abs(yf.Data[0] - yq.Data[0]):G4}");
    }
    
    model.Dispose();
}

// Same for Q8_0 baseline
Console.Error.WriteLine("\n=== Q8_0: QUANTIZED vs DEQUANTIZED (Full mode) ===");
string q8Path = Path.Combine(basePath, "qwen2-0_5b-instruct-q8_0.gguf");
if (File.Exists(q8Path))
{
    var meta = GgufLoader.LoadMeta(q8Path);
    ModelConfig c = GgufLoader.LoadConfig(meta)!;
    var sc = c.ForModel(HardwareTier.AVX2);
    var w = GgufLoader.LoadWeightsToTransformerWeights(q8Path, c, null, LoadMode.Full);
    using var model = ModelFactory.CreateSession(w, sc, null, null, false);
    var block = model.GetBlock(0)!;
    
    using var x = new Tensor<float>(1, c.HiddenDim);
    var rng = new Random(42);
    for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);
    
    var layers = new[] {
        ("attn_q", block.Attention.Wq),
        ("attn_v", block.Attention.Wv),
    };
    
    foreach (var (name, layer) in layers)
    {
        var savedRaw = layer.RawQuantizedData;
        var savedDtype = layer.QuantDtype;
        layer.RawQuantizedData = null;
        using var yf = layer.Forward(x, model.Ops);
        layer.RawQuantizedData = savedRaw;
        layer.QuantDtype = savedDtype;
        using var yq = layer.Forward(x, model.Ops);
        
        double diff = 0, maxDiff = 0;
        for (int i = 0; i < yf.ElementCount; i++)
        {
            double d = Math.Abs(yf.Data[i] - yq.Data[i]);
            if (d > 1e-3) Console.Error.WriteLine($"    LARGE: [{i}] float={yf.Data[i]:G8} quant={yq.Data[i]:G8} d={d:G4}");
            diff += d; if (d > maxDiff) maxDiff = d;
        }
        double avg = diff / yf.ElementCount;
        double norm = 0;
        for (int i = 0; i < yf.ElementCount; i++) norm += Math.Abs(yf.Data[i]);
        norm /= yf.ElementCount;
        Console.Error.WriteLine($"  {name} ({layer.InFeatures}x{layer.OutFeatures} dtype={layer.QuantDtype}): avg={avg:G6} max={maxDiff:G6} avg|yf|={norm:G6}");
        Console.Error.WriteLine($"    [0] float={yf.Data[0]:G8} quant={yq.Data[0]:G8} d={Math.Abs(yf.Data[0] - yq.Data[0]):G4}");
    }
    model.Dispose();
}

Console.Error.WriteLine("\nDone!");
Console.In.ReadLine();
