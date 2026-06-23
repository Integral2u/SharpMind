using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpMind;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q4KTests
{
    private readonly ITestOutputHelper _output;

    public Q4KTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ4K_ValidBlock()
    {
        // 144-byte block: d[2] + dmin[2] + scales[12] + qs[128]
        var block = new byte[144];
        
        // d=1.0 (0x3C00), min=0.0 (0x0000)
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        
        // scales[0] = 0x11 (sc=1, m=1)
        block[4] = 0x11;
        
        // qs[0]=0x11 (4 values: 1, 1)
        block[16] = 0x11;
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoader.ReadQ4K(reader, data.AsSpan(), 256);
        
        // Val = d * sc * actual = 1.0 * 17 * 1 - 0 = 17.0
        
        _output.WriteLine($"data[0]={data[0]}");
        
        Assert.Equal(17.0f, data[0]);
    }
    
    [Fact]
    public void Diagnostic_Q4KffnDownVerify()
    {
        string q8Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf";
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q8Path) || !File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        var m8 = GgufLoader.LoadMeta(q8Path);
        var m4 = GgufLoader.LoadMeta(q4Path);

        // Q4_K ffn_down layers: 2,4,5,11,12,14,15,17,18,20,22,23
        int[] layers = [2, 4, 5, 11, 12];
        foreach (int layer in layers)
        {
            string name = $"blk.{layer}.ffn_down.weight";
            var t8 = m8.Tensors.FirstOrDefault(t => t.Name == name);
            var t4 = m4.Tensors.FirstOrDefault(t => t.Name == name);
            if (t8.Name == null || t4.Name == null) { _output.WriteLine($"Missing {name}"); continue; }
            if (t4.Dtype != GgufDtype.Q4_K) { _output.WriteLine($"{name} has dtype {t4.Dtype} not Q4_K"); continue; }

            int count = 1; foreach (int d in t8.Shape) count *= d;
            float[] d8 = new float[count], d4 = new float[count];

            using (var fs = File.OpenRead(q8Path))
            using (var br = new BinaryReader(fs))
            {
                fs.Position = m8.DataOffset + t8.Offset;
                GgufLoader.ReadQ8_0(br, d8.AsSpan(), count);
            }

            using (var fs = File.OpenRead(q4Path))
            using (var br = new BinaryReader(fs))
            {
                fs.Position = m4.DataOffset + t4.Offset;
                GgufLoader.ReadQ4K(br, d4.AsSpan(), count);
                _output.WriteLine($"ReadQ4K returned, count={count}");
            }

            _output.WriteLine($"\n{name}: Q8_0 vs Q4_K (shape [{string.Join(",", t8.Shape)}], {count} elems)");
            _output.WriteLine("  First 8:");
            for (int i = 0; i < 8; i++)
                _output.WriteLine($"  [{i}] Q8={d8[i]:G6}  Q4={d4[i]:G6}  diff={Math.Abs(d8[i] - d4[i]):G4}");

            double sumSq = 0; for (int i = 0; i < count; i++) { double diff = d8[i] - d4[i]; sumSq += diff * diff; }
            double rms = Math.Sqrt(sumSq / count);
            _output.WriteLine($"  RMS error: {rms:G6}");
            _output.WriteLine($"  Range Q8: [{d8.Min():G6}, {d8.Max():G6}]");
            _output.WriteLine($"  Range Q4: [{d4.Min():G6}, {d4.Max():G6}]");

            // Check for silent zeros (would indicate exception during read)
            int zeroCount = 0;
            for (int i = 0; i < count; i++) if (d4[i] == 0) zeroCount++;
            _output.WriteLine($"  Q4 zero elements: {zeroCount}/{count} ({100f * zeroCount / count:F2}%)");
            double maxAbs = 0; for (int i = 0; i < count; i++) { double a = Math.Abs(d4[i]); if (a > maxAbs) maxAbs = a; }
            _output.WriteLine($"  Q4 max abs value: {maxAbs:G6}");
        }
    }

    [Fact]
    public void Diagnostic_CheckAllTensorsNonZero()
    {
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        var meta = GgufLoader.LoadMeta(q4Path);
        int tensorsChecked = 0, allZeroTensors = 0, nanTensors = 0;

        foreach (var info in meta.Tensors)
        {
            if (info.Dtype == GgufDtype.F32) continue; // skip F32 biases/norms

            int count = 1; foreach (int d in info.Shape) count *= d;
            // Only check first portion to keep it fast
            int limit = Math.Min(count, 32000);
            float[] data = new float[limit];

            using (var fs = File.OpenRead(q4Path))
            using (var br = new BinaryReader(fs))
            {
                fs.Position = meta.DataOffset + info.Offset;
                switch (info.Dtype)
                {
                    case GgufDtype.Q5_0: GgufLoader.ReadQ5_0(br, data.AsSpan(), limit); break;
                    case GgufDtype.Q4_K: GgufLoader.ReadQ4K(br, data.AsSpan(), limit); break;
                    case GgufDtype.Q6_K: GgufLoader.ReadQ6K(br, data.AsSpan(), limit); break;
                    case GgufDtype.Q8_0: GgufLoader.ReadQ8_0(br, data.AsSpan(), limit); break;
                    default: continue;
                }
            }

            tensorsChecked++;
            int zeros = 0;
            for (int i = 0; i < limit; i++) if (data[i] == 0) zeros++;
            int nans = 0;
            for (int i = 0; i < limit; i++) if (float.IsNaN(data[i]) || float.IsInfinity(data[i])) nans++;

            if (zeros == limit)
            {
                _output.WriteLine($"ALL ZERO: {info.Name} ({info.Dtype})");
                allZeroTensors++;
            }
            else if (zeros > limit / 2)
            {
                _output.WriteLine($"MOSTLY ZERO: {info.Name} ({info.Dtype}) - {100f * zeros / limit:F1}% zeros");
            }
            if (nans > 0)
            {
                _output.WriteLine($"NaN/Inf: {info.Name} ({info.Dtype}) - {nans} values");
                nanTensors++;
            }
        }
        _output.WriteLine($"\nChecked {tensorsChecked} non-F32 tensors");
        _output.WriteLine($"All-zero tensors: {allZeroTensors}");
        _output.WriteLine($"NaN/Inf tensors: {nanTensors}");
    }

    [Fact]
    public void Diagnostic_FirstBlockVerify()
    {
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        var meta = GgufLoader.LoadMeta(q4Path);
        var t4 = meta.Tensors.FirstOrDefault(t => t.Name == "blk.0.attn_q.weight");
        if (t4.Name == null) { _output.WriteLine("Tensor not found"); return; }

        long pos = meta.DataOffset + t4.Offset;
        _output.WriteLine($"Tensor: {t4.Name}, dtype={t4.Dtype}, shape=[{string.Join(",", t4.Shape)}]");
        _output.WriteLine($"offset={t4.Offset}, dataOffset={meta.DataOffset}, filePos={pos}");

        // Step 1: Read raw block 0
        byte[] rawBlock = new byte[22];
        using (var fs = File.OpenRead(q4Path))
        {
            fs.Position = pos;
            fs.Read(rawBlock, 0, 22);
        }
        _output.WriteLine($"Raw block 0: {BitConverter.ToString(rawBlock)}");
        // Decode d from bytes 0-1 (little-endian half-float)
        ushort dRaw = (ushort)(rawBlock[0] | (rawBlock[1] << 8));
        uint qhRaw = (uint)(rawBlock[2] | (rawBlock[3] << 8) | (rawBlock[4] << 16) | (rawBlock[5] << 24));
        byte[] qsRaw = new byte[16];
        Array.Copy(rawBlock, 6, qsRaw, 0, 16);
        // GgufLoader.HalfToFloat is private, so replicate it
        float DQ5_0(ushort v) {
            int sig = (v >> 15) & 1, exp = (v >> 10) & 0x1F, man = v & 0x3FF;
            if (exp == 0) return (sig == 0 ? 1 : -1) * (man / 1024.0f) * (1.0f / 16384.0f);
            if (exp == 31) return man == 0 ? (sig == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
            return (sig == 0 ? 1 : -1) * (1.0f + man / 1024.0f) * (1 << (exp - 15));
        }
        float dVal = DQ5_0(dRaw);
        _output.WriteLine($"d=0x{dRaw:X4} ({dVal:G8}) qh=0x{qhRaw:X8}");

        // Step 2: Manual dequant with & 1
        _output.WriteLine("\nManual dequant (& 1):");
        for (int j = 0; j < 32; j++)
        {
            int nib = (j % 2 == 0) ? (qsRaw[j/2] & 0x0F) : (qsRaw[j/2] >> 4);
            int high = (int)((qhRaw >> j) & 1);
            int val = nib | (high << 4);
            _output.WriteLine($"  [{j,2}]: nib={nib,2} high={high} val={val,2}  deq={dVal * (val - 16),10:G8}");
        }

        // Step 3: Manual dequant with & 3 (at 2*j)
        _output.WriteLine("\nManual dequant (& 3 at 2j):");
        for (int j = 0; j < 32; j++)
        {
            int nib = (j % 2 == 0) ? (qsRaw[j/2] & 0x0F) : (qsRaw[j/2] >> 4);
            int high = (int)((qhRaw >> (2*j)) & 3);
            int val = nib | (high << 4);
            _output.WriteLine($"  [{j,2}]: nib={nib,2} high={high} val={val,2}  deq={dVal * (val - 16),10:G8}");
        }

        // Step 4: GgufLoader.ReadQ5_0 output
        using (var fs = File.OpenRead(q4Path))
        using (var br = new BinaryReader(fs))
        {
            fs.Position = pos;
            var data = new float[32];
            GgufLoader.ReadQ5_0(br, data.AsSpan(), 32);
            _output.WriteLine("\nGgufLoader.ReadQ5_0:");
            for (int i = 0; i < 32; i++)
                _output.WriteLine($"  [{i,2}] = {data[i]:G8}");
        }

        // Step 5: Q8_0 comparison
        string q8Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf";
        if (File.Exists(q8Path))
        {
            var m8 = GgufLoader.LoadMeta(q8Path);
            var t8 = m8.Tensors.FirstOrDefault(t => t.Name == "blk.0.attn_q.weight");
            if (t8.Name != null)
            {
                using (var fs = File.OpenRead(q8Path))
                using (var br = new BinaryReader(fs))
                {
                    fs.Position = m8.DataOffset + t8.Offset;
                    var q8data = new float[32];
                    GgufLoader.ReadQ8_0(br, q8data.AsSpan(), 32);
                    _output.WriteLine("\nQ8_0 first 32:");
                    for (int i = 0; i < 32; i++)
                        _output.WriteLine($"  [{i,2}] = {q8data[i]:G8}");
                }
            }
        }
    }

    [Fact]
    public void Diagnostic_CompareForwardPass()
    {
        string q8Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf";
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q8Path) || !File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        _output.WriteLine("Loading Q8_0 model...");
        var sw = Stopwatch.StartNew();
        GgufLoader.Load(q8Path, null, out var m8Meta, out var m8Config, out _);
        var sc8 = m8Config.ForModel(HardwareTier.AVX2);
        var w8 = GgufLoader.LoadWeightsToTransformerWeights(q8Path, m8Config);
        var model8 = ModelFactory.CreateSession(w8, sc8);
        _output.WriteLine($"  done in {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        _output.WriteLine("Loading Q4_K_M model...");
        GgufLoader.Load(q4Path, null, out var m4Meta, out var m4Config, out _);
        var sc4 = m4Config.ForModel(HardwareTier.AVX2);
        var w4 = GgufLoader.LoadWeightsToTransformerWeights(q4Path, m4Config);
        var model4 = ModelFactory.CreateSession(w4, sc4);
        _output.WriteLine($"  done in {sw.Elapsed.TotalSeconds:F1}s");

        // Create single-token input [0]
        using var input = Tensor<int>.From(new int[] { 0 }, 1, 1);

        // Compare embeddings
        var emb8 = model8.ForwardEmbedding(input);
        var emb4 = model4.ForwardEmbedding(input);
        double embDiff = 0;
        for (int i = 0; i < emb8.ElementCount; i++)
            embDiff += Math.Abs(emb8.Data[i] - emb4.Data[i]);
        _output.WriteLine($"Embedding diff sum: {embDiff:G6}");
        for (int i = 0; i < 4; i++)
            _output.WriteLine($"  emb[{i}] Q8={emb8.Data[i]:G8} Q4={emb4.Data[i]:G8}");

        // Per-layer comparison
        int numLayers = m8Config.NumLayers;
        _output.WriteLine($"\nPer-layer comparison ({numLayers} layers):");

        var h8 = new Tensor<float>(emb8.Shape);
        emb8.Data.CopyTo(h8.Data);
        var h4 = new Tensor<float>(emb4.Shape);
        emb4.Data.CopyTo(h4.Data);
        emb8.Dispose();
        emb4.Dispose();

        int divergedAt = -1;
        for (int layer = 0; layer < numLayers; layer++)
        {
            var block8 = model8.GetBlock(layer);
            var block4 = model4.GetBlock(layer);
            if (block8 == null || block4 == null) { _output.WriteLine($"  Layer {layer}: block null"); break; }

            if (layer == 0)
            {
                double Norm(Tensor<float> t) { double s = 0; for (int i = 0; i < t.ElementCount; i++) s += t.Data[i] * (double)t.Data[i]; return Math.Sqrt(s); }
                var gated = block4.Ffn.WGated!.Weight;
                var d4 = block4.Ffn.WDown.Weight;
                int ffnDim = gated.Shape[1] / 2;
                _output.WriteLine($"  WGated norm={Norm(gated):G6} shape=[{gated.Shape.Rows},{gated.Shape.Cols}]");
                _output.WriteLine($"  Wdown norm={Norm(d4):G6} shape=[{d4.Shape.Rows},{d4.Shape.Cols}]");
                _output.WriteLine($"  WGated max={gated.Data.ToArray().Max():G6} min={gated.Data.ToArray().Min():G6}");
                _output.WriteLine($"  Wdown first 8: {string.Join(", ", Enumerable.Range(0,8).Select(i=>d4.Data[i].ToString("G6")))}");
                _output.WriteLine($"  Wdown last 8:  {string.Join(", ", Enumerable.Range(d4.ElementCount-8,8).Select(i=>d4.Data[i].ToString("G6")))}");

                // Also check Q8_0 Wdown
                var d8 = block8.Ffn.WDown.Weight;
                _output.WriteLine($"  Q8 Wdown norm={Norm(d8):G6} first={d8.Data[0]:G6} last={d8.Data[d8.ElementCount-1]:G6}");
            }

            // Forward through block (in-place)
            var next8 = block8.Forward(h8, null, 0, true);
            var next4 = block4.Forward(h4, null, 0, true);

            // Compare
            double diff = 0, maxDiff = 0;
            for (int i = 0; i < next8.ElementCount; i++)
            {
                double d = Math.Abs(next8.Data[i] - next4.Data[i]);
                diff += d;
                if (d > maxDiff) maxDiff = d;
            }
            double avgDiff = diff / next8.ElementCount;

            if (divergedAt < 0 && avgDiff > 1e-4)
            {
                divergedAt = layer;
                _output.WriteLine($"  *** DIVERGENCE at layer {layer}! avgDiff={avgDiff:G6} maxDiff={maxDiff:G6}");
                for (int i = 0; i < Math.Min(8, next8.ElementCount); i++)
                    _output.WriteLine($"  [{i}] Q8={next8.Data[i]:G8}  Q4={next4.Data[i]:G8}  diff={Math.Abs(next8.Data[i] - next4.Data[i]):G4}");
            }
            if (layer % 6 == 5 || layer == numLayers - 1)
            {
                double norm8 = 0, norm4 = 0;
                for (int i = 0; i < next8.ElementCount; i++) { norm8 += next8.Data[i] * next8.Data[i]; norm4 += next4.Data[i] * next4.Data[i]; }
                _output.WriteLine($"  Layer {layer}: norm8={Math.Sqrt(norm8):G6} norm4={Math.Sqrt(norm4):G6} avgDiff={avgDiff:G6}");
            }

            h8 = next8;
            h4 = next4;
        }

        if (divergedAt >= 0)
        {
            _output.WriteLine($"\n*** BUG LOCATED: Forward pass diverges at layer {divergedAt} ***");
        }
        else
        {
            _output.WriteLine($"\nNo divergence detected — both models produce identical forward pass.");
        }

        // === Diagnostics: VecDot vs dequant for actual weight dtypes ===
        _output.WriteLine($"\n=== VecDot Diagnostics (dtype-aware) ===");
        var saveW4 = model4.ForwardEmbedding(Tensor<int>.From(new int[] { 0 }, 1, 1));
        _output.WriteLine($"  Config: layers={m8Config.NumLayers} hidden={m8Config.HiddenDim} ffn={m8Config.FfnDim} heads={m8Config.NumHeads} kv={m8Config.NumKvHeads} eps={m8Config.NormEps:G4}");
        _output.WriteLine($"  Q4 Config: layers={m4Config.NumLayers} hidden={m4Config.HiddenDim} ffn={m4Config.FfnDim} heads={m4Config.NumHeads} kv={m4Config.NumKvHeads} eps={m4Config.NormEps:G4}");

        var blk0 = w4.Blocks[0];
        _output.WriteLine($"  Raw sizes: Wq={blk0.RawWq?.Length} Wgate={blk0.RawWgate?.Length} Wup={blk0.RawWup?.Length} Wf2={blk0.RawWf2?.Length} Wk={blk0.RawWk?.Length} Wv={blk0.RawWv?.Length} Wo={blk0.RawWo?.Length}");

        unsafe
        {
            var inputEmb = saveW4;
            int hidden = m4Config.HiddenDim;
            int ffn = m4Config.FfnDim;

            // Check if input has any NaN
            bool inputHasNan = false;
            for (int i = 0; i < inputEmb.ElementCount; i++) { if (float.IsNaN(inputEmb.Data[i]) || float.IsInfinity(inputEmb.Data[i])) { inputHasNan = true; break; } }
            _output.WriteLine($"  Input has NaN/Inf: {inputHasNan}");

            string DetectDtype(byte[] raw, int inF, int outF)
            {
                if (raw.Length == (outF * inF + 31) / 32 * 22) return "Q5_0";
                if (raw.Length == (outF * inF + 255) / 256 * 210) return "Q6_K";
                if (raw.Length == (outF * inF + 31) / 32 * 34) return "Q8_0";
                if (raw.Length == (outF * inF + 255) / 256 * 144) return "Q4_K";
                if (raw.Length == (outF * inF + 255) / 256 * 176) return "Q5_K";
                return "unknown";
            }

            void TestVecDot(string name, byte[] rawData, int inFeatures, int outFeatures)
            {
                string dtype = DetectDtype(rawData, inFeatures, outFeatures);
                _output.WriteLine($"  {name}: detected dtype={dtype} bytes={rawData.Length} inF={inFeatures} outF={outFeatures}");

                int blockBytes, blkSize;
                switch (dtype)
                {
                    case "Q5_0": blockBytes = 22; blkSize = 32; break;
                    case "Q6_K": blockBytes = 210; blkSize = 256; break;
                    case "Q8_0": blockBytes = 34; blkSize = 32; break;
                    default: _output.WriteLine($"    unsupported dtype {dtype}, skipping"); return;
                }

                int expBlocks = (inFeatures * outFeatures + blkSize - 1) / blkSize;
                int expBytes = expBlocks * blockBytes;
                if (rawData.Length != expBytes)
                {
                    _output.WriteLine($"    WARNING: size mismatch expected={expBytes} actual={rawData.Length}");
                    return;
                }

                int nColBlocks = (inFeatures + blkSize - 1) / blkSize;

                fixed (byte* pRaw = rawData)
                fixed (float* pIn = inputEmb.Data)
                {
                    var qOps = QuantizationFactory.Create(HardwareTier.Scalar);

                    for (int col = 0; col < Math.Min(3, outFeatures); col++)
                    {
                        // Dequantize and compute dot product
                        double expected = 0;
                        int startBlock = (col * inFeatures) / blkSize;
                        int colOff = (col * inFeatures) % blkSize;
                        for (int b = 0; b < nColBlocks; b++)
                        {
                            byte* block = pRaw + (long)(startBlock + b) * blockBytes;
                            float[] deq = GgufLoader.ReadBlock(block, dtype, blkSize);
                            int curStart = (b == 0) ? colOff : 0;
                            int curEnd = Math.Min(blkSize, inFeatures + colOff - b * blkSize);
                            for (int j = curStart; j < curEnd; j++)
                                expected += pIn[b * blkSize + j - colOff] * deq[j];
                        }

                        // VecDot
                        float vecDotVal = dtype switch
                        {
                            "Q5_0" => qOps.VecDotQ5_0(pIn, pRaw, col, inFeatures),
                            "Q6_K" => qOps.VecDotQ6K(pIn, pRaw, col, inFeatures),
                            "Q8_0" => qOps.VecDotQ8_0(pIn, pRaw, col, inFeatures),
                            _ => float.NaN
                        };

                        double relDiff = Math.Abs((double)vecDotVal - expected) / Math.Max(1.0, Math.Abs(expected));
                        string status = relDiff > 1e-4 ? $"FAIL({relDiff:G3})" : "OK";
                        _output.WriteLine($"  {name}[col={col}]: VecDot={vecDotVal:G6} expected={expected:G6} {status}");
                    }
                }
            }

            if (blk0.RawWq != null) TestVecDot("Wq", blk0.RawWq, hidden, hidden);
            if (blk0.RawWk != null) TestVecDot("Wk", blk0.RawWk, hidden, hidden);
            if (blk0.RawWv != null) TestVecDot("Wv", blk0.RawWv, hidden, hidden);
            if (blk0.RawWo != null) TestVecDot("Wo", blk0.RawWo, hidden, hidden);
            if (blk0.RawWgate != null) TestVecDot("Wgate", blk0.RawWgate, hidden, ffn);
            if (blk0.RawWup != null) TestVecDot("Wup", blk0.RawWup, hidden, ffn);
            if (blk0.RawWf2 != null) TestVecDot("Wf2", blk0.RawWf2, ffn, hidden);

            saveW4.Dispose();
        }

        // Compare final logits
        _output.WriteLine($"\nComparing full-model logits:");
        using var logits8 = model8.Forward(input);
        using var logits4 = model4.Forward(input);
        double logitSum = 0, logitMaxDiff = 0;
        int top8 = 0, top4 = 0;
        float maxLogit8 = float.MinValue, maxLogit4 = float.MinValue;
        for (int i = 0; i < logits8.ElementCount; i++)
        {
            double d = Math.Abs(logits8.Data[i] - logits4.Data[i]);
            logitSum += d;
            if (d > logitMaxDiff) logitMaxDiff = d;
            if (logits8.Data[i] > maxLogit8) { maxLogit8 = logits8.Data[i]; top8 = i; }
            if (logits4.Data[i] > maxLogit4) { maxLogit4 = logits4.Data[i]; top4 = i; }
        }
        _output.WriteLine($"  avgDiff={logitSum / logits8.ElementCount:G6} maxDiff={logitMaxDiff:G6}");
        _output.WriteLine($"  Q8_0 top token={top8} ({maxLogit8:G4})");
        _output.WriteLine($"  Q4_K_M top token={top4} ({maxLogit4:G4})");
        _output.WriteLine($"  Same top token? {(top8 == top4 ? "YES" : "NO")}");

        model8.Dispose();
        model4.Dispose();
        w8.Dispose();
        w4.Dispose();
    }

    [Fact]
    public void Diagnostic_MinimalMatMul()
    {
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        _output.WriteLine("Loading Q4_K_M model metadata...");
        GgufLoader.Load(q4Path, null, out var m4Meta, out var m4Config, out _);
        var sc4 = m4Config.ForModel(HardwareTier.AVX2);
        _output.WriteLine("Loading weights...");
        var w4 = GgufLoader.LoadWeightsToTransformerWeights(q4Path, m4Config);
        _output.WriteLine("Creating session...");
        var model4 = ModelFactory.CreateSession(w4, sc4);
        _output.WriteLine("Session created.");
        model4.Dispose();
        w4.Dispose();
    }

    [Fact]
    public void Diagnostic_Q6K_MatMul_Direct()
    {
        string q4Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        if (!File.Exists(q4Path)) { _output.WriteLine("SKIP"); return; }

        // Load the first Q6_K tensor (blk.0.ffn_down.weight)
        GgufLoader.Load(q4Path, null, out var meta, out var config, out _);
        var t6 = meta.Tensors.FirstOrDefault(t => t.Name == "blk.0.ffn_down.weight");
        if (t6.Name == null) { _output.WriteLine("Tensor not found"); return; }
        _output.WriteLine($"Tensor: {t6.Name}, dtype={t6.Dtype}, shape=[{string.Join(",", t6.Shape)}]");

        // Read raw data
        int inF = (int)t6.Shape[0];  // 4864
        int outF = (int)t6.Shape[1];  // 896
        int totalBytes = 896 * 210 * (4864 / 256);
        using var fs = File.OpenRead(q4Path);
        long pos = meta.DataOffset + t6.Offset;
        fs.Position = pos;
        byte[] rawData = new byte[totalBytes];
        int read = fs.Read(rawData, 0, totalBytes);
        _output.WriteLine($"Read {read} bytes, expected {totalBytes}");

        // Create input data (random-like)
        float[] inputData = new float[inF];
        for (int i = 0; i < inF; i++) inputData[i] = (float)(Math.Sin(i * 0.1) * 0.5);

        // Allocate output
        float[] outputData = new float[outF];

        unsafe
        {
            fixed (byte* pRaw = rawData)
            fixed (float* pIn = inputData)
            fixed (float* pOut = outputData)
            {
                _output.WriteLine("Calling QuantizedMatMulQ6K_Scalar...");
                QuantizationKernels.QuantizedMatMulQ6K_Scalar(pIn, pRaw, pOut, 1, inF, outF);
                _output.WriteLine("Done.");
            }
        }

        _output.WriteLine($"Output[0]={outputData[0]:G6} [1]={outputData[1]:G6} [2]={outputData[2]:G6}");
    }
}
