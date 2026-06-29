using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using System.Text;

namespace SandBox;

public static class LogitDiagnostic
{
    public static async Task RunAsync()
    {
        string modelName = "qwen2-0.5b-instruct-q2_k";
        string ggufPath = Path.Combine(
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets",
            $"{modelName}.gguf");
        if (!File.Exists(ggufPath)) { Console.Error.WriteLine($"Missing {ggufPath}"); return; }

        GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out _);
        Console.Error.WriteLine($"  File: {meta.TensorCount} tensors");
        foreach (var ti in meta.Tensors.Take(5))
            Console.Error.WriteLine($"  tensor '{ti.Name}': dtype={(uint)ti.Dtype}({ti.Dtype}) shape=[{string.Join(",",ti.Shape)}]");
        var sharpConfig = modelConfig.ForModel();
        using var weights = GgufLoader.LoadWeightsToTransformerWeights(ggufPath, modelConfig);
        using var model = ModelFactory.CreateSession(weights, sharpConfig);

        // Single token (BOS=151643 for Qwen2)
        using var input = Tensor<int>.From(new int[] { 151643 }, 1, 1);

        int vocabSize = modelConfig.VocabSize;
        Console.Error.WriteLine($"Vocab size: {vocabSize} layers={modelConfig.NumLayers}");

        // Check embedding
        var emb = model.ForwardEmbedding(input);
        bool embNan = false;
        for (int i = 0; i < Math.Min(10, emb.ElementCount); i++)
        {
            if (float.IsNaN(emb.Data[i]) || float.IsInfinity(emb.Data[i])) embNan = true;
        }
        Console.Error.WriteLine($"Embedding has NaN: {embNan}, first 5: {emb.Data[0]:G6} {emb.Data[1]:G6} {emb.Data[2]:G6} {emb.Data[3]:G6} {emb.Data[4]:G6}");

        bool HasNan(Tensor<float> t, int n) { for (int i = 0; i < Math.Min(n, t.ElementCount); i++) if (float.IsNaN(t.Data[i]) || float.IsInfinity(t.Data[i])) return true; return false; }
        double AbsMean(Tensor<float> t, int n) { double s = 0; int c = Math.Min(n, t.ElementCount); for (int i = 0; i < c; i++) s += Math.Abs(t.Data[i]); return s / c; }

        // Trace inside layer 0
        var h = emb;
        var blk = model.GetBlock(0);
        var normed1 = blk.Norm1.Forward(h, null);
        Console.Error.WriteLine($"After norm1: NaN={HasNan(normed1,100)} meanAbs={AbsMean(normed1,100):G6}");

        // Check raw data sizes
        Console.Error.WriteLine($"Wq raw: {blk.Attention.Wq.RawQuantizedData?.Length} bytes, inF={blk.Attention.Wq.InFeatures} outF={blk.Attention.Wq.OutFeatures}");
        Console.Error.WriteLine($"Wk raw: {blk.Attention.Wk.RawQuantizedData?.Length} bytes, inF={blk.Attention.Wk.InFeatures} outF={blk.Attention.Wk.OutFeatures}");
        Console.Error.WriteLine($"Wv raw: {blk.Attention.Wv.RawQuantizedData?.Length} bytes, inF={blk.Attention.Wv.InFeatures} outF={blk.Attention.Wv.OutFeatures}");

        unsafe
        {
            fixed (float* pIn = normed1.Data)
            {
                var rawWq = blk.Attention.Wq.RawQuantizedData;
                var qOps = blk.Attention.Wq.QuantizationOps;
                fixed (byte* pRaw = rawWq)
                {
                    // Check ALL blocks for bad d/dmin
                    int nBlocksWq = rawWq.Length / 144;
                    // Try 142-byte stride too
                    int nBlocks142 = rawWq.Length / 142;
                    Console.Error.WriteLine($"Wq: {nBlocksWq} blocks (144B) or {nBlocks142} blocks (142B) rawLen={rawWq.Length}");
                    
                    // Read block 0 with 144B stride and 142B stride, check if byte 144 matches byte 142
                    byte* b144 = pRaw + 144;
                    byte* b142 = pRaw + 142;
                    Console.Error.WriteLine($"  byte 144 = 0x{b144[0]:X2}{b144[1]:X2} byte 142 = 0x{b142[0]:X2}{b142[1]:X2}");

                    // Print first 4 blocks with 144B stride
                    for (int bi = 0; bi < 4; bi++)
                    {
                        byte* blk2 = pRaw + bi * 144;
                        ushort dRaw = *(ushort*)blk2;
                        ushort minRaw = *(ushort*)(blk2 + 2);
                        Console.Error.WriteLine($"  block {bi} @144B: d=0x{dRaw:X4}({qOps.HalfToFloat(dRaw):G6}) min=0x{minRaw:X4}({qOps.HalfToFloat(minRaw):G6})");
                    }
                    // Same with 142B stride
                    for (int bi = 0; bi < 4; bi++)
                    {
                        byte* blk2 = pRaw + bi * 142;
                        ushort dRaw = *(ushort*)blk2;
                        ushort minRaw = *(ushort*)(blk2 + 2);
                        Console.Error.WriteLine($"  block {bi} @142B: d=0x{dRaw:X4}({qOps.HalfToFloat(dRaw):G6}) min=0x{minRaw:X4}({qOps.HalfToFloat(minRaw):G6})");
                    }
                    int nanDBlock = -1, nanMinBlock = -1, negDCount = 0;
                    for (int bi = 0; bi < nBlocksWq; bi++)
                    {
                        byte* blk2 = pRaw + bi * 144;
                        float d = qOps.HalfToFloat(*(ushort*)blk2);
                        float min = qOps.HalfToFloat(*(ushort*)(blk2 + 2));
                        if (float.IsNaN(d) && nanDBlock < 0) nanDBlock = bi;
                        if (float.IsNaN(min) && nanMinBlock < 0) nanMinBlock = bi;
                        if (d < 0) negDCount++;
                        if (float.IsNaN(d) || float.IsNaN(min) || float.IsInfinity(d) || float.IsInfinity(min))
                        {
                            ushort dRaw = *(ushort*)blk2;
                            ushort minRaw = *(ushort*)(blk2 + 2);
                            Console.Error.WriteLine($"  Wq bad block {bi}: d={d:G6}(0x{dRaw:X4}) min={min:G6}(0x{minRaw:X4})");
                        }
                    }
                    Console.Error.WriteLine($"Wq: negDCount={negDCount}/{nBlocksWq} firstNanD={nanDBlock} firstNanMin={nanMinBlock}");

                    // Full VecDotQ4K scan for NaN cols
                    int nanCount = 0, infCount = 0, maxNanCol = -1;
                    float maxAbs = 0; int maxAbsCol = -1;
                    for (int c = 0; c < 896; c++)
                    {
                        float val = qOps.VecDotQ4K(pIn, pRaw, c, 896);
                        if (float.IsNaN(val)) { nanCount++; if (maxNanCol < 0) maxNanCol = c; }
                        if (float.IsInfinity(val)) infCount++;
                        float a = Math.Abs(val);
                        if (a > maxAbs) { maxAbs = a; maxAbsCol = c; }
                    }
                    Console.Error.WriteLine($"VecDotQ4K: NaN={nanCount}/{896} Inf={infCount} maxAbs={maxAbs:G6}@col{maxAbsCol} firstNaN=col{maxNanCol}");
                }
                
                var rawWv = blk.Attention.Wv.RawQuantizedData;
                var qOps2 = blk.Attention.Wv.QuantizationOps;
                fixed (byte* pRaw = rawWv)
                {
                    // Quick Wv block check
                    int nBlocksWv = rawWv.Length / 22;
                    int nanCount = 0, infCount = 0;
                    float maxAbs = 0;
                    for (int c = 0; c < 128; c++)
                    {
                        float val2 = qOps2.VecDotQ5_0(pIn, pRaw, c, 896);
                        if (float.IsNaN(val2)) nanCount++;
                        if (float.IsInfinity(val2)) infCount++;
                        float a = Math.Abs(val2);
                        if (a > maxAbs) maxAbs = a;
                    }
                    Console.Error.WriteLine($"VecDotQ5_0({nBlocksWv} blocks): NaN={nanCount}/{128} Inf={infCount} maxAbs={maxAbs:G6}");
                }
            }
        }

        // Check Q, K, V projections individually
        int hiddenDim = modelConfig.HiddenDim;
        int numHeads = modelConfig.NumHeads;
        int numKvHeads = modelConfig.NumKvHeads;
        int headDim = hiddenDim / numHeads;
        Console.Error.WriteLine($"hidden={hiddenDim} heads={numHeads} kv={numKvHeads} headDim={headDim}");

        var q = blk.Attention.Wq.Forward(normed1, model.Ops, null);
        var k = blk.Attention.Wk.Forward(normed1, model.Ops, null);
        var v = blk.Attention.Wv.Forward(normed1, model.Ops, null);
        Console.Error.WriteLine($"Q: NaN={HasNan(q,100)} meanAbs={AbsMean(q,100):G6}");
        Console.Error.WriteLine($"K: NaN={HasNan(k,100)} meanAbs={AbsMean(k,100):G6}");
        Console.Error.WriteLine($"V: NaN={HasNan(v,100)} meanAbs={AbsMean(v,100):G6}");
        q.Dispose(); k.Dispose(); v.Dispose();

        var attnOut = blk.Attention.Forward(normed1, model.Ops, 0, true, null, null);
        Console.Error.WriteLine($"After attn: NaN={HasNan(attnOut,100)} meanAbs={AbsMean(attnOut,100):G6}");
        normed1.Dispose();

        TensorOps.AddInPlace(h, attnOut);
        attnOut.Dispose();
        Console.Error.WriteLine($"After attn+residual: NaN={HasNan(h,100)} meanAbs={AbsMean(h,100):G6}");

        var normed2 = blk.Norm2.Forward(h, null);
        Console.Error.WriteLine($"After norm2: NaN={HasNan(normed2,100)} meanAbs={AbsMean(normed2,100):G6}");

        var ffnOut = blk.Ffn.Forward(normed2, null);
        Console.Error.WriteLine($"After ffn: NaN={HasNan(ffnOut,100)} meanAbs={AbsMean(ffnOut,100):G6}");
        normed2.Dispose();

        TensorOps.AddInPlace(h, ffnOut);
        ffnOut.Dispose();
        Console.Error.WriteLine($"After ffn+residual: NaN={HasNan(h,100)} meanAbs={AbsMean(h,100):G6}");

        using var logits = model.ForwardLastLogits(input, null, 0);
        bool allNan = true;
        for (int i = 0; i < Math.Min(10, vocabSize); i++)
        {
            if (!float.IsNaN(logits.Data[i])) { allNan = false; break; }
        }
        Console.Error.WriteLine($"All logits NaN: {allNan}");
        Console.Error.WriteLine($"  logits[0..4]: {logits.Data[0]:G6} {logits.Data[1]:G6} {logits.Data[2]:G6} {logits.Data[3]:G6} {logits.Data[4]:G6}");

        Console.Error.WriteLine("Done.");
    }
}
