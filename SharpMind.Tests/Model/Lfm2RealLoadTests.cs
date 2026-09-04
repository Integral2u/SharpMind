using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Model;

/// <summary>
/// Loads the real LFM2.5 file and isolates where an access violation (0xC0000005)
/// happens during model assembly. Test-only; the real model file is not committed.
/// </summary>
public class Lfm2RealLoadTests
{
    private readonly ITestOutputHelper _out;

    public Lfm2RealLoadTests(ITestOutputHelper output) => _out = output;

    private const string Lfm2Path = @"C:\Users\tarra\SharpMind\Models\LFM2.5-2.6B-Q8_0.gguf";

    [Fact]
    [Trait("Category", "ModelLoad")]
    public void LoadAndInspectWeights()
    {
        if (!File.Exists(Lfm2Path))
        {
            _out.WriteLine($"Model file not found: {Lfm2Path}");
            return;
        }

        var meta = ModelFormatHelpers.GetModelMetaHelperFor(SharpMind.Model.Format.ModelFormat.Gguf);
        meta.Load(Lfm2Path, null, out var m, out var c, out var _);
        _out.WriteLine($"Arch: {c.Architecture}, Hidden={c.HiddenDim}, Layers={c.NumLayers}, NumKvHeads={c.NumKvHeads}");
        _out.WriteLine($"IsShortConvLayer array len: {(c.LayerKvHeads?.Length.ToString() ?? "null")}");
        _out.WriteLine($"ShortConvCacheLength: {c.ShortConvCacheLength}");
        _out.WriteLine($"VocabSize(config)={c.VocabSize}");
        _out.WriteLine($"HeadDim={c.HeadDim}, FfnDim={c.FfnDim}");

        var sharpConfig = c.ForModel(hw: HardwareTier.Scalar);
        var mapping = sharpConfig.ToJigSawMapping(parallel: false);
        var qOps = QuantizationFactory.Create(mapping);

        var weights = ModelFactory.CreateWeights(c, sharpConfig, qOps, Lfm2Path, LoadMode.Full, quantizedResident: true);        weights.InitializeWeights(new Progress<float>(p => _out.WriteLine($"  load: {p:P0}")));
        _out.WriteLine("Weights loaded OK");

        var transformer = ModelFactory.CreateTransformer(weights, sharpConfig, mapping, optimizeMemory: true);
        _out.WriteLine("Transformer created OK");
        _out.WriteLine($"Embedding shape: [{string.Join(",", transformer.EmbeddingWeight.Shape)}]");
        _out.WriteLine($"RawEmbeddingDtype: {transformer.RawEmbeddingDtype?.ToString() ?? "null"}, bytes: {(transformer.RawEmbedding?.Length.ToString() ?? "null")}");
        _out.WriteLine($"RawLmHeadDtype: {weights.RawLmHeadDtype?.ToString() ?? "null"}, bytes: {(weights.RawLmHead?.Length.ToString() ?? "null")}");
        _out.WriteLine($"LmHead tensor: {(transformer.LmHead is null ? "null" : $"[{string.Join(",", transformer.LmHead.Shape)}]")}");
        var fnw = transformer.FinalNorm.NormWeight;
        float fnSum = 0;
        for (int i = 0; i < fnw.Shape[0]; i++) fnSum += Math.Abs(fnw.Data[i]);
        _out.WriteLine($"FinalNorm weight: rows={fnw.Shape[0]} absSum={fnSum:F3} first5={string.Join(",", Enumerable.Range(0, 5).Select(i => fnw.Data[i].ToString("F4")))}");
    }

    [Fact]
    [Trait("Category", "ModelLoad")]
    public async Task GeneratePlausibleTokens()
    {
        if (!File.Exists(Lfm2Path))
        {
            _out.WriteLine($"Model file not found: {Lfm2Path}");
            return;
        }

        var meta = ModelFormatHelpers.GetModelMetaHelperFor(SharpMind.Model.Format.ModelFormat.Gguf);
        meta.Load(Lfm2Path, null, out var m, out var c, out var tokenizer);
        _out.WriteLine($"Tokenizer: {(tokenizer is null ? "null" : "present")}, vocab={tokenizer?.VocabSize}");

        if (tokenizer is null)
        {
            _out.WriteLine("No tokenizer loaded; skipping generation check.");
            return;
        }

        var sharpConfig = c.ForModel(hw: HardwareTier.Scalar);
        var mapping = sharpConfig.ToJigSawMapping(parallel: false);
        var qOps = QuantizationFactory.Create(mapping);

        var weights = ModelFactory.CreateWeights(c, sharpConfig, qOps, Lfm2Path, LoadMode.Full, quantizedResident: true);
        weights.InitializeWeights(new Progress<float>(p => { }));
        var transformer = ModelFactory.CreateTransformer(weights, sharpConfig, mapping, optimizeMemory: true);

        // Encode a simple prompt
        var prompt = "The capital of France is";
        var ids = tokenizer.Encode(prompt, addBos: true, addEos: false);
        _out.WriteLine($"Prompt {prompt} -> {ids.Length} tokens");

        // Run a single forward pass through the full model (standard generator).
        GeneratorDiagnostics.DumpTopLogits = true;
        var gen = new StandardGenerator<KVCacherBuilder>(transformer, tokenizer, addBos: true, addEos: false, seed: 42);
        _out.WriteLine("Generator created OK");

        int count = 0;
        var sb = new System.Text.StringBuilder();
        await foreach (var frag in gen.GenerateAsync(prompt,
            new SamplingConfig { Temperature = 0 },
            new GenerationConfig { MaxNewTokens = 50 }))
        {
            sb.Append(frag);
            _out.WriteLine($"  frag[{count}]: repr=[{Repr(frag)}]");
            count++;
            if (count > 50) break;
        }
        _out.WriteLine($"Generated {count} fragments, total chars={sb.Length}, EosId={tokenizer.EosId}, BOS={tokenizer.BosId}");
        _out.WriteLine($"FULL: [{Repr(sb.ToString())}]");
    }

    private static string Repr(string s)
    {
        var sb2 = new System.Text.StringBuilder();
        foreach (char c in s)
        {
            if (char.IsControl(c) || c == ' ') sb2.Append($"<U+{(int)c:X4}>");
            else sb2.Append(c);
        }
        return sb2.ToString();
    }
}
