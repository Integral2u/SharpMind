using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.GPU;

/// <summary>
/// End-to-end smoke test against a real quantized GGUF model (a Q8_0-only Qwen2 0.5B),
/// loaded exactly the way the production host loads one for pure inference —
/// <c>CreateWeights(..., quantizedResident: true)</c> + <c>CreateTransformer</c>, so every
/// block linear, the embedding, and the weight-tied LM head read raw Q8_0 bytes. This is the
/// load the GPU engine previously crashed on (the placeholder [In,1] float weight backed the
/// raw bytes); it must now be <i>accepted</i> by <see cref="GpuInferenceEngine.ValidateSupported"/>
/// and produce greedy-decode output identical to the pure-CPU generator.
///
/// Gated on the model file being present so CI (no model downloads) still passes.
/// </summary>
[Collection("GPU")]
public sealed class GpuQuantizedResidentTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";
    private const int MaxCacheLength = 64;
    private const int MaxPromptTokens = 32;

    [Fact]
    public async Task RealQ8_0Model_ValidateSupportedAccepts_AndGpuMatchesCpuGreedy()
    {
        if (!File.Exists(ModelPath)) return;

        var metaHelper = SharpMind.Model.Format.ModelFormatHelpers.GetModelMetaHelperFor(SharpMind.Model.Format.ModelFormat.Gguf);
        metaHelper.Load(ModelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
        Assert.NotNull(tokenizer);

        var sharpConfig = modelConfig.ForModel(hw: HardwareTier.Auto);
        var mapping = sharpConfig.ToJigSawMapping(parallel: true);
        var qOps = QuantizationFactory.Create(mapping);

        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, ModelPath, LoadMode.Full, quantizedResident: true);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        // Was NotSupportedException before the Q8_0 engine; must be accepted now.
        GpuInferenceEngine.ValidateSupported(model, sharpConfig);

        using var engine = new GpuInferenceEngine(GpuTestDevice.Device, model, sharpConfig, MaxCacheLength, MaxPromptTokens);
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, tokenizer, addBos: false, addEos: false, numLayers: modelConfig.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(model, tokenizer, addBos: false, addEos: false, seed: 1);

        int[] prompt = tokenizer.Encode("Hello, how are you doing today?", addBos: false, addEos: false);
        Assert.NotEmpty(prompt);

        var engineIds = await GreedyIds(engineGen, prompt, maxNew: 8);
        var cpuIds = await GreedyIds(cpuGen, prompt, maxNew: 8);

        Assert.Equal(cpuIds, engineIds);
    }

    private static async Task<List<int>> GreedyIds(IGenerator<KVCacherBuilder> gen, int[] prompt, int maxNew)
    {
        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, sampling: SamplingConfig.Greedy,
            generation: new GenerationConfig { MaxNewTokens = maxNew, Stream = false }))
        {
        }
        return gen.CurrentGeneratedIds?.ToList() ?? new List<int>();
    }
}
