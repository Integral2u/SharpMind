using SharpMind;
using SharpMind.Model.Format;
using SharpMind.Model.Config;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Core.Quantization;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using System.Runtime.InteropServices;

await SharpMind.Samples.Examples.KnownFailingModels.RunAsync("Hello");
Console.In.ReadLine();
return;

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

// -- Full mode: compare quantized forward vs float dequantized forward --
Console.Error.WriteLine("\n=== Q4_K_M: QUANTIZED vs DEQUANTIZED (Full mode) ===");
string q5Path = Path.Combine(basePath, "qwen2-0_5b-instruct-q4_k_m.gguf");
if (File.Exists(q5Path))
{
    var meta = GgufLoader.LoadMeta(q5Path);
    ModelConfig c = GgufLoader.LoadConfig(meta)!;
    var sc = c.ForModel(HardwareTier.AVX2);
    var w = GgufLoader.LoadWeightsToTransformerWeights(q5Path, c, null, LoadMode.Full);
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
            diff += d; if (d > maxDiff) maxDiff = d;
        }
        double avg = diff / yf.ElementCount;
        double norm = 0;
        for (int i = 0; i < yf.ElementCount; i++) norm += Math.Abs(yf.Data[i]);
        norm /= yf.ElementCount;
        Console.Error.WriteLine($"  {name} ({layer.InFeatures}x{layer.OutFeatures} dtype={layer.QuantDtype}): avg={avg:G6} max={maxDiff:G6} avg|yf|={norm:G6}");
    }
    model.Dispose();
}

Console.Error.WriteLine("\nDone!");
Console.In.ReadLine();
