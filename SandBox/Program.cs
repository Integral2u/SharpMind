using SharpMind;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Core.Tensors;
using System.Runtime.Intrinsics.X86;
using System.Diagnostics;

await SharpMind.Samples.Examples.MultiTestInteractive.RunAsync("Hello");
/*
var modelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
var ggufPath = Path.Combine(modelPath, "qwen2-0_5b-instruct-q8_0.gguf");
var hw = Avx2.IsSupported ? HardwareTier.AVX2 : HardwareTier.Scalar;

Console.Write("Loading... "); Console.Out.Flush();
GgufLoader.Load(ggufPath, null, out GgufMeta meta, out ModelConfig mc, out Tokenizer? tokenizer);
if (tokenizer is null) return;
var cfg = mc.ForModel(hw);
GC.Collect(); GC.WaitForPendingFinalizers();
var model = ModelFactory.Create(mc, cfg);
GC.Collect(); GC.WaitForPendingFinalizers();
GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
var ops = InferenceOpsFactory.Create(cfg, InferenceConfig.Default);
Console.WriteLine("OK"); Console.Out.Flush();

// Warmup to trigger weight transpose
using var gen = new Generator(model, tokenizer, ops);
await foreach (var _ in gen.GenerateFromTokensAsync(
    tokenizer.Encode("Hello", addBos: true, addEos: false),
    SamplingConfig.Greedy, new GenerationConfig { MaxNewTokens = 3, Stream = true })) { }

// Single-step timing
gen.ResetCache();
int[] prompt = tokenizer.Encode("Hi!", addBos: true, addEos: false);

// Do one prefill + first decode
var sw = Stopwatch.StartNew();
await foreach (var frag in gen.GenerateFromTokensAsync(
    prompt, SamplingConfig.Greedy,
    new GenerationConfig { MaxNewTokens = 5, Stream = true }))
{
    if (sw.Elapsed.TotalSeconds > 10) break;
    Console.Write(frag); Console.Out.Flush();
}
sw.Stop();
Console.WriteLine($"\n{tokens} tokens in {sw.Elapsed.TotalSeconds:F2}s");
model.Dispose();
*/