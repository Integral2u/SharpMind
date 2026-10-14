using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// SharpMind does not keep a denylist of architectures. A denylist has to name
/// every architecture the decoder cannot run, which is unknowable in advance: it
/// risks rejecting a model that would have loaded, and it goes stale the moment a
/// family is implemented. Instead any architecture is attempted against its own
/// tensors, and a genuinely unsupported one fails loudly from the shape/quant
/// validation that first disagrees with the file.
/// </summary>
public sealed class UnsupportedArchitectureTests
{
    private static ModelConfig Cfg(string architecture) => new()
    {
        Architecture = architecture,
        VocabSize = 128,
        HiddenDim = 16,
        NumLayers = 2,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 32,
        MaxSeqLen = 32,
    };

    private static Exception? Load(string architecture)
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        return Record.Exception(() => ModelFactory.CreateWeights(
            Cfg(architecture), sharpConfig, qOps, "does-not-exist.gguf"));
    }

    /// <summary>
    /// No architecture is pre-rejected. These all get past the former gate and go
    /// on to fail on the missing file instead.
    /// </summary>
    [Theory]
    [InlineData("gemma4")]
    [InlineData("GEMMA4")]
    [InlineData("gemma3n")]
    [InlineData("qwen2")]
    [InlineData("llama")]
    [InlineData("gemma2")]
    [InlineData("")]
    public void NoArchitecture_IsPreRejected(string architecture)
        => Assert.IsNotType<NotSupportedException>(Load(architecture));

    /// <summary>
    /// <see cref="SharpMindConfig.ForModel"/> is resolved by the app pipeline
    /// <em>before</em> <see cref="ModelFactory.CreateWeights"/>. It used to be the
    /// only coverage, and it passes a prebuilt <c>SharpMindConfig.Gpt</c> — nothing
    /// ever called <c>ForModel</c> with a real architecture string.
    ///
    /// That let <c>ForModel</c> reject any architecture it did not list, which broke
    /// models that had always worked: qwen2 was never listed (it silently used the
    /// catch-all default), so rejecting the catch-all took down qwen2, qwen2vl, phi3,
    /// deepseek and every other family that relied on it. The preset table is a
    /// convenience, not a gate.
    [Theory]
    [InlineData("qwen2", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("qwen2vl", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("qwen2moe", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.MoE, NormKind.RMSNorm)]
    [InlineData("qwen3", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("llama", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("mistral", ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("gemma2", ActivationKind.GELU, GateKind.GeGLU, FfnKind.Gated, NormKind.RMSNorm)]
    [InlineData("bert", ActivationKind.GELU, GateKind.None, FfnKind.Dense, NormKind.LayerNorm)]
    [InlineData("opt", ActivationKind.ReLU, GateKind.None, FfnKind.Dense, NormKind.LayerNorm)]
    [InlineData("GPT2", ActivationKind.GELU, GateKind.None, FfnKind.Dense, NormKind.LayerNorm)]
    public void ForModel_ResolvesKnownArchitectures(
        string architecture,
        ActivationKind act, GateKind gate, FfnKind ffn, NormKind norm)
    {
        var cfg = SharpMindConfig.ForModel(8, 8, architecture, HardwareTier.Scalar);

        Assert.Equal(act, cfg.Activation);
        Assert.Equal(gate, cfg.Gate);
        Assert.Equal(ffn, cfg.Ffn);
        Assert.Equal(norm, cfg.Norm);
    }

    [Theory]
    // An architecture SharpMind has never heard of must still resolve: the fallback
    // is the standard decoder preset, which is what essentially every modern LLM is.
    [InlineData("gemma-4")]
    public void ForModel_DoesNotThrow_AndUsesTheDecoderPreset_Gemma4(string architecture)
    {
        var cfg = SharpMindConfig.ForModel(8, 8, architecture, HardwareTier.Scalar);

        // gemma4 is a known family: expect its correct preset (GELU/GeGLU/RMSNorm decoder).
        Assert.Equal(ActivationKind.GELU, cfg.Activation);
        Assert.Equal(GateKind.GeGLU, cfg.Gate);
        Assert.Equal(FfnKind.Gated, cfg.Ffn);
        Assert.Equal(NormKind.RMSNorm, cfg.Norm);
        Assert.Equal(ArchKind.Decoder, cfg.Arch);
    }

    [Theory]
    [InlineData("some-future-arch")]
    [InlineData("internlm2")]
    [InlineData("glm4")]
    [InlineData("deepseek3")]
    public void ForModel_DoesNotThrow_AndUsesTheDecoderPresetFallback(string architecture)
    {
        var cfg = SharpMindConfig.ForModel(8, 8, architecture, HardwareTier.Scalar);

        Assert.Equal(ActivationKind.SiLU, cfg.Activation);
        Assert.Equal(GateKind.SwiGLU, cfg.Gate);
        Assert.Equal(FfnKind.Gated, cfg.Ffn);
        Assert.Equal(NormKind.RMSNorm, cfg.Norm);
        Assert.Equal(ArchKind.Decoder, cfg.Arch);
    }

    [Fact]
    public void ForModel_NoArchitecture_UsesTheDecoderPreset()
    {
        foreach (string? arch in new[] { null, "", "   " })
        {
            var cfg = SharpMindConfig.ForModel(8, 8, arch, HardwareTier.Scalar);

            Assert.Equal(ActivationKind.SiLU, cfg.Activation);
            Assert.Equal(FfnKind.Gated, cfg.Ffn);
            Assert.Equal(ArchKind.Decoder, cfg.Arch);
        }
    }

    [Theory]
    [InlineData(8, 8, AttentionKind.MHA)]
    [InlineData(8, 1, AttentionKind.MQA)]
    [InlineData(8, 4, AttentionKind.GQA)]
    public void ForModel_MapsHeadCounts(int heads, int kvHeads, AttentionKind expected)
        => Assert.Equal(expected, SharpMindConfig.ForModel(heads, kvHeads, "qwen2").Attention);
}
