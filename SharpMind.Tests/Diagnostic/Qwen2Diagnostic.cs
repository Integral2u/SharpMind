using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using SharpMind;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Diagnostic;

public class Qwen2Diagnostic
{
    private readonly ITestOutputHelper _output;

    public Qwen2Diagnostic(ITestOutputHelper output) => _output = output;

    /// <summary>Compare quantized vs float matmul. Loads proper dequantized float weights from GGUF
/// so the comparison is valid (previously float weights were zero because quantized tensors
/// skip float loading in LoadWeightsToTransformerWeights).</summary>
    [Fact]
    public void Verify_Quantized_vs_Float_MatMul()
    {
        string[] paths = [
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0.5b-instruct-q2_k.gguf",
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf"
        ];
        foreach (var path in paths)
        {
            if (!File.Exists(path)) { _output.WriteLine($"SKIP {path}"); continue; }
            string label = path.Contains("q2_k") ? "Q2_K" : "Q8_0";
            _output.WriteLine($"\n=== {label} ===");

            var meta = GgufLoader.LoadMeta(path);
            ModelConfig c = GgufLoader.LoadConfig(meta)!;
            // Use AVX2 for the actual model
            var sc = c.ForModel(HardwareTier.AVX2);
            var w = GgufLoader.LoadWeightsToTransformerWeights(path, c);
            var m = ModelFactory.CreateSession(w, sc);
            var block = m.GetBlock(0)!;

            // Load dequantized float weights for proper comparison (only the 3 tensors we need)
            using var ggufStream = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(ggufStream);
            var tensorsByName = meta.Tensors.ToDictionary(t => t.Name, t => t);

            using var x = new Tensor<float>(1, c.HiddenDim);
            var rng = new Random(42);
            for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);

            var layers = new (LinearLayer? layer, string name, string ggufName, float eps)[] {
                (block.Attention.Wv, "V", "blk.0.attn_v.weight", 0.01f),
                (block.Attention.Wq, "Q", "blk.0.attn_q.weight", 0.01f),
                (block.Attention.Wo, "O", "blk.0.attn_output.weight", 0.01f),
            };
            foreach (var (layer, name, ggufName, eps) in layers)
            {
                if (layer == null) { _output.WriteLine($"  {name}: SKIP (null)"); continue; }

                // Load proper dequantized float weight for just this tensor
                if (tensorsByName.TryGetValue(ggufName, out var tensorInfo))
                {
                    ggufStream.Position = meta.DataOffset + tensorInfo.Offset;
                    int count = 1;
                    foreach (int d in tensorInfo.Shape) count *= d;
                    var floatBuf = new float[count];
                    GgufLoader.ReadTensorInto(reader, tensorInfo.Dtype, tensorInfo.Shape, floatBuf.AsSpan());
                    layer.LoadWeightTransposed(floatBuf.AsSpan());
                }
                else
                {
                    _output.WriteLine($"  WARN: {ggufName} not found in GGUF meta!");
                }

                layer.UseQuantizedForward = false;
                using var yf = layer.Forward(x, m.Ops);
                layer.UseQuantizedForward = true;
                using var yq = layer.Forward(x, m.Ops);
                double diff = 0;
                for (int i = 0; i < yf.ElementCount; i++)
                    diff += Math.Abs(yf.Data[i] - yq.Data[i]);
                double avg = diff / yf.ElementCount;
                string dt = layer.QuantDtype?.ToString() ?? "?";
                _output.WriteLine($"  {name} ({layer.InFeatures}x{layer.OutFeatures} dtype={dt}): diff_avg={avg:G6}");
                bool pass = avg < eps;
                _output.WriteLine($"    {(pass ? "PASS" : "FAIL")} (threshold={eps})");
                if (!pass)
                {
                    for (int i = 0; i < Math.Min(4, yf.ElementCount); i++)
                        _output.WriteLine($"    [{i}] float={yf.Data[i]:G8} quant={yq.Data[i]:G8} diff={Math.Abs(yf.Data[i] - yq.Data[i]):G4}");
                }
            }
            m.Dispose();
        }
    }

    [Fact]
    public void Diagnose_Qwen2_Q2K_vs_Q8_Forward()
    {
        string q2Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0.5b-instruct-q2_k.gguf";
        string q8Path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf";
        if (!File.Exists(q2Path) || !File.Exists(q8Path))
        {
            _output.WriteLine("SKIP -- model files not found");
            return;
        }

        var m2 = GgufLoader.LoadMeta(q2Path);
        var m8 = GgufLoader.LoadMeta(q8Path);
        ModelConfig c2 = GgufLoader.LoadConfig(m2) ?? throw new InvalidOperationException("Q2_K config is null");
        ModelConfig c8 = GgufLoader.LoadConfig(m8) ?? throw new InvalidOperationException("Q8_0 config is null");
        _output.WriteLine("=== Configs ===");
        _output.WriteLine($"Q2_K: arch={c2.Architecture} layers={c2.NumLayers} hidden={c2.HiddenDim} heads={c2.NumHeads} kv={c2.NumKvHeads} ffn={c2.FfnDim}");
        _output.WriteLine($"Q8_0: arch={c8.Architecture} layers={c8.NumLayers} hidden={c8.HiddenDim} heads={c8.NumHeads} kv={c8.NumKvHeads} ffn={c8.FfnDim}");

        // All Q2_K blk.0 tensors
        _output.WriteLine("\n=== All Q2_K blk.0 tensors ===");
        foreach (var t in m2.Tensors.Where(t => t.Name.Contains("blk.0.")))
            _output.WriteLine($"  {t.Name}: shape=[{string.Join(",", t.Shape)}] dtype={t.Dtype}");

        // All Q8_0 blk.0 tensors
        _output.WriteLine("\n=== All Q8_0 blk.0 tensors ===");
        foreach (var t in m8.Tensors.Where(t => t.Name.Contains("blk.0.")))
            _output.WriteLine($"  {t.Name}: shape=[{string.Join(",", t.Shape)}] dtype={t.Dtype}");

        // Check token_embd and output.weight
        var emb = m2.Tensors.First(t => t.Name == "token_embd.weight");
        _output.WriteLine($"\ntoken_embd: shape=[{string.Join(",", emb.Shape)}] dtype={emb.Dtype}");
        bool hasOut = m2.Tensors.Any(t => t.Name == "output.weight");
        _output.WriteLine($"output.weight exists: {hasOut}");

        // Build SharpMindConfigs
        SharpMindConfig sc2 = c2.ForModel(HardwareTier.AVX2);
        SharpMindConfig sc8 = c8.ForModel(HardwareTier.AVX2);

        // Load models
        var w2 = GgufLoader.LoadWeightsToTransformerWeights(q2Path, c2);
        var model2 = ModelFactory.CreateSession(w2, sc2);
        var w8 = GgufLoader.LoadWeightsToTransformerWeights(q8Path, c8);
        var model8 = ModelFactory.CreateSession(w8, sc8);

        // Embedding (token 0)
        using var input = Tensor<int>.From(new int[] { 0 }, 1, 1);
        using var emb2 = model2.ForwardEmbedding(input);
        using var emb8 = model8.ForwardEmbedding(input);
        _output.WriteLine("\n=== Embedding (token 0) ===");
        double embDiff = 0;
        int nEmb = Math.Min(emb2.ElementCount, emb8.ElementCount);
        for (int i = 0; i < nEmb; i++)
            embDiff += Math.Abs(emb2.Data[i] - emb8.Data[i]);
        _output.WriteLine($"Diff sum: {embDiff:G6}");
        for (int i = 0; i < 4; i++)
            _output.WriteLine($"  [{i}] Q2={emb2.Data[i]:G8} Q8={emb8.Data[i]:G8}");

        // Create hidden state tensors for both models
        var h2 = new Tensor<float>(emb2.Shape);
        emb2.Data.CopyTo(h2.Data);
        var h8 = new Tensor<float>(emb8.Shape);
        emb8.Data.CopyTo(h8.Data);

        // Sublayer-level diagnostic on layer 0
        _output.WriteLine("\n=== Layer 0 sublayer breakdown ===");
        var b02 = model2.GetBlock(0);
        var b08 = model8.GetBlock(0);

        using var n1_2 = b02!.Norm1.Forward(h2);
        using var n1_8 = b08!.Norm1.Forward(h8);
        double d1 = 0;
        int n1 = Math.Min(n1_2.ElementCount, n1_8.ElementCount);
        for (int i = 0; i < n1; i++) d1 += Math.Abs(n1_2.Data[i] - n1_8.Data[i]);
        _output.WriteLine($"After norm1: diff sum={d1:G6} (avg={d1 / n1:G6})");

        using var attn2 = b02.Attention.Forward(n1_2, model2.Ops, 0, true, null, null);
        using var attn8 = b08.Attention.Forward(n1_8, model8.Ops, 0, true, null, null);
        double da = 0;
        int na = Math.Min(attn2.ElementCount, attn8.ElementCount);
        for (int i = 0; i < na; i++) da += Math.Abs(attn2.Data[i] - attn8.Data[i]);
        _output.WriteLine($"After attention: diff sum={da:G6} (avg={da / na:G6})");
        for (int i = 0; i < Math.Min(4, na); i++)
            _output.WriteLine($"  attn [{i}] Q2={attn2.Data[i]:G8} Q8={attn8.Data[i]:G8}");

        using var res2 = new Tensor<float>(h2.Shape);
        h2.Data.CopyTo(res2.Data);
        TensorOps.AddInPlace(res2, attn2);
        using var res8 = new Tensor<float>(h8.Shape);
        h8.Data.CopyTo(res8.Data);
        TensorOps.AddInPlace(res8, attn8);
        using var n2_2 = b02.Norm2.Forward(res2);
        using var n2_8 = b08.Norm2.Forward(res8);
        double d2 = 0;
        int nn = Math.Min(n2_2.ElementCount, n2_8.ElementCount);
        for (int i = 0; i < nn; i++) d2 += Math.Abs(n2_2.Data[i] - n2_8.Data[i]);
        _output.WriteLine($"After norm2: diff sum={d2:G6} (avg={d2 / nn:G6})");

        using var ffn2 = b02.Ffn.Forward(n2_2, null);
        using var ffn8 = b08.Ffn.Forward(n2_8, null);
        double df = 0;
        int nf = Math.Min(ffn2.ElementCount, ffn8.ElementCount);
        for (int i = 0; i < nf; i++) df += Math.Abs(ffn2.Data[i] - ffn8.Data[i]);
        _output.WriteLine($"After FFN: diff sum={df:G6} (avg={df / nf:G6})");
        for (int i = 0; i < Math.Min(4, nf); i++)
            _output.WriteLine($"  ffn [{i}] Q2={ffn2.Data[i]:G8} Q8={ffn8.Data[i]:G8}");

        // Per-block forward comparison
        int numLayers = c2.NumLayers;
        _output.WriteLine($"\n=== Per-block forward ({numLayers} layers) ===");

        int divergedAt = -1;
        for (int layer = 0; layer < numLayers; layer++)
        {
            var b2 = model2.GetBlock(layer);
            var b8 = model8.GetBlock(layer);
            if (b2 == null || b8 == null) { _output.WriteLine($"  Block {layer} is null"); break; }

            var n2 = b2.Forward(h2, null, 0, true);
            var n8 = b8.Forward(h8, null, 0, true);

            int n = Math.Min(n2.ElementCount, n8.ElementCount);
            double diff = 0, maxDiff = 0;
            for (int i = 0; i < n; i++)
            {
                double d = Math.Abs(n2.Data[i] - n8.Data[i]);
                diff += d;
                if (d > maxDiff) maxDiff = d;
            }
            double avgDiff = diff / n;

            if (divergedAt < 0 && avgDiff > 1e-2)
            {
                divergedAt = layer;
                _output.WriteLine($"  *** DIVERGENCE at layer {layer}! avgDiff={avgDiff:G6} maxDiff={maxDiff:G6}");
                for (int i = 0; i < Math.Min(8, n); i++)
                    _output.WriteLine($"    [{i}] Q2={n2.Data[i]:G8}  Q8={n8.Data[i]:G8}  diff={Math.Abs(n2.Data[i] - n8.Data[i]):G4}");
            }

            h2.Dispose();
            h2 = n2;
            h8.Dispose();
            h8 = n8;
            GC.Collect();
        }

        if (divergedAt < 0)
            _output.WriteLine("\nNo divergence via per-block comparison.");

        // Final norm
        using var fn2 = model2.FinalNorm.Forward(h2);
        using var fn8 = model8.FinalNorm.Forward(h8);
        double fnDiff = 0;
        for (int i = 0; i < fn2.ElementCount; i++)
            fnDiff += Math.Abs(fn2.Data[i] - fn8.Data[i]);
        _output.WriteLine($"\nFinal norm diff sum: {fnDiff:G6}");

        // Full logits
        using var logits2 = model2.Forward(input);
        using var logits8 = model8.Forward(input);
        double lDiff = 0;
        int nLog = Math.Min(logits2.ElementCount, logits8.ElementCount);
        for (int i = 0; i < nLog; i++)
            lDiff += Math.Abs(logits2.Data[i] - logits8.Data[i]);
        _output.WriteLine($"Logit diff sum: {lDiff:G6}");
        var top2 = TensorOps.ArgTopK(logits2, 5);
        var top8 = TensorOps.ArgTopK(logits8, 5);
        _output.WriteLine($"Top-5 Q2_K: {string.Join(", ", top2)}");
        _output.WriteLine($"Top-5 Q8_0: {string.Join(", ", top8)}");
        for (int i = 0; i < 12; i++)
            _output.WriteLine($"  logit[{i}] Q2={logits2.Data[i]:G8} Q8={logits8.Data[i]:G8} diff={Math.Abs(logits2.Data[i] - logits8.Data[i]):G4}");

        h2.Dispose();
        h8.Dispose();
        model2.Dispose();
        model8.Dispose();
    }

    /// <summary>Directly test IQ4_NL VecDot correctness by comparing Scalar vs AVX2 on real model data.</summary>
    [Fact]
    public void Verify_IQ4NL_VecDot_Scalar_vs_AVX2()
    {
        string path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0.5b-instruct-q2_k.gguf";
        if (!File.Exists(path)) { _output.WriteLine("SKIP"); return; }

        var meta = GgufLoader.LoadMeta(path);
        ModelConfig c = GgufLoader.LoadConfig(meta)!;

        // Load raw quantized data for Q projection (IQ4_NL)
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        var tInfo = meta.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
        int inF = (int)tInfo.Shape[1]; // GGUF [Out, In]
        int outF = (int)tInfo.Shape[0];
        long rawSize = GgufLoader.GetRawTensorByteCount(tInfo.Shape, tInfo.Dtype);
        fs.Position = meta.DataOffset + tInfo.Offset;
        byte[] rawData = new byte[rawSize];
        fs.ReadExactly(rawData);
        _output.WriteLine($"Q weight: GGUF shape=[{inF},{outF}] dtype={tInfo.Dtype} rawBytes={rawSize}");

        // Also load float version
        fs.Position = meta.DataOffset + tInfo.Offset;
        int count = inF * outF;
        float[] floatBuf = new float[count];
        GgufLoader.ReadTensorInto(reader, tInfo.Dtype, tInfo.Shape, floatBuf.AsSpan());

        // Create random input
        using var x = new Tensor<float>(1, inF);
        var rng = new Random(42);
        for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);

        // Compare per-column: scalar VecDot vs float dot product
        int nBlocks = (inF + 31) / 32;
        int colToTest = Math.Min(5, outF);
        _output.WriteLine($"\nComparing IQ4_NL VecDot (first {colToTest} columns of {outF}):");
        for (int col = 0; col < colToTest; col++)
        {
            // Float reference
            double floatSum = 0;
            for (int i = 0; i < inF; i++)
                floatSum += x.Data[i] * floatBuf[col * inF + i];

            // Scalar VecDot
            unsafe
            {
                fixed (byte* pRaw = rawData)
                {
                    double scalarSum = QuantizationKernels.VecDotQ4_NL_Scalar(x.DataPtr, pRaw, col, inF);
                    double avx2Sum = QuantizationKernels.VecDotQ4_NL_AVX2(x.DataPtr, pRaw, col, inF);
                    _output.WriteLine($"  col={col}: float={floatSum:G8} scalar={scalarSum:G8} avx2={avx2Sum:G8}");
                }
            }
        }
    }

    /// <summary>Verify every K-quant VecDot against dequantized float for all failing models.</summary>
    [Fact]
    public void Verify_KQuant_VecDots_Against_Float()
    {
        var models = new[] {
            ("TinyLlama-1.1B-Chat-v1.0.Q4_K_M", @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf", "blk.0.attn_q.weight"),
            ("DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf", "blk.0.attn_q.weight"),
        };

        foreach (var (label, path, tensorName) in models)
        {
            if (!File.Exists(path)) { _output.WriteLine($"SKIP {label}"); continue; }
            _output.WriteLine($"\n=== {label} ===");
            var meta = GgufLoader.LoadMeta(path);
            ModelConfig c = GgufLoader.LoadConfig(meta)!;
            var sc = c.ForModel(HardwareTier.AVX2);
            var w = GgufLoader.LoadWeightsToTransformerWeights(path, c);
            var m = ModelFactory.CreateSession(w, sc);
            var block = m.GetBlock(0)!;

            using var x = new Tensor<float>(1, c.HiddenDim);
            var rng = new Random(42);
            for (int i = 0; i < x.ElementCount; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);

            // Test each attention projection
            var layers = new (LinearLayer? layer, string name, string ggufName)[] {
                (block.Attention.Wq, "Q", tensorName),
                (block.Attention.Wk, "K", tensorName.Replace("attn_q", "attn_k")),
                (block.Attention.Wv, "V", tensorName.Replace("attn_q", "attn_v")),
                (block.Attention.Wo, "O", tensorName.Replace("attn_q", "attn_output")),
            };

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fs);
            var tensorsByName = meta.Tensors.ToDictionary(t => t.Name, t => t);

            foreach (var (layer, lname, gname) in layers)
            {
                if (layer == null) { _output.WriteLine($"  {lname}: SKIP (null)"); continue; }

                // Load dequantized float weight
                if (tensorsByName.TryGetValue(gname, out var tInfo))
                {
                    fs.Position = meta.DataOffset + tInfo.Offset;
                    int count = 1;
                    foreach (int d in tInfo.Shape) count *= d;
                    var floatBuf = new float[count];
                    GgufLoader.ReadTensorInto(reader, tInfo.Dtype, tInfo.Shape, floatBuf.AsSpan());
                    layer.LoadWeightTransposed(floatBuf.AsSpan());
                }

                layer.UseQuantizedForward = false;
                using var yf = layer.Forward(x, m.Ops);
                layer.UseQuantizedForward = true;
                using var yq = layer.Forward(x, m.Ops);
                double diff = 0;
                for (int i = 0; i < yf.ElementCount; i++)
                    diff += Math.Abs(yf.Data[i] - yq.Data[i]);
                double avg = diff / yf.ElementCount;
                string dt = layer.QuantDtype?.ToString() ?? "?";
                string pass = avg < 0.01 ? "PASS" : "FAIL";
                _output.WriteLine($"  {lname} ({layer.InFeatures}x{layer.OutFeatures} dtype={dt}): diff_avg={avg:G6} {(pass)}");
                if (avg > 0.01)
                    for (int i = 0; i < Math.Min(2, yf.ElementCount); i++)
                        _output.WriteLine($"    [{i}] float={yf.Data[i]:G8} quant={yq.Data[i]:G8} diff={Math.Abs(yf.Data[i] - yq.Data[i]):G4}");
            }
            m.Dispose();
        }
    }

    /// <summary>Verify F16 model: check F16 dequantization produces valid values.</summary>
    [Fact]
    public void Verify_F16_Dequant()
    {
        string path = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\Qwen2.5-1.5B-Instruct-f16.gguf";
        if (!File.Exists(path)) { _output.WriteLine("SKIP"); return; }
        var meta = GgufLoader.LoadMeta(path);
        var qInfo = meta.Tensors.First(t => t.Name == "blk.0.attn_q.weight");
        _output.WriteLine($"attn_q: shape=[{string.Join(",", qInfo.Shape)}] dtype={qInfo.Dtype}");

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        fs.Position = meta.DataOffset + qInfo.Offset;
        int count = 1;
        foreach (int d in qInfo.Shape) count *= d;
        var buf = new float[count];
        GgufLoader.ReadTensorInto(reader, qInfo.Dtype, qInfo.Shape, buf.AsSpan());

        double sum = 0;
        int nanCount = 0;
        for (int i = 0; i < count; i++) { if (float.IsNaN(buf[i])) nanCount++; sum += Math.Abs(buf[i]); }
        _output.WriteLine($"F16 dequant: avg_abs={(sum / count):G6} nanCount={nanCount} ({(nanCount == 0 ? "PASS" : "FAIL")})");
    }
}
