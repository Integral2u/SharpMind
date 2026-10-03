using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// A model whose architecture the decoder does not implement must say so, before
/// anything is allocated.
///
/// gemma-4 (gemma-3n family) used to derive layer shapes from its config, disagree
/// with its own tensors, and surface as a byte-count mismatch at the first matmul —
/// which reads as a corrupt file rather than an unsupported architecture. Its
/// attn_q is [1536, 2048] while head_count 8 x key_length 512 implies 4096.
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

    [Theory]
    [InlineData("gemma4")]
    [InlineData("GEMMA4")]
    [InlineData("gemma3n")]
    public void UnsupportedArchitecture_FailsWithItsName(string architecture)
    {
        var ex = Assert.IsType<NotSupportedException>(Load(architecture));

        Assert.Contains(architecture, ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not supported", ex.Message, StringComparison.OrdinalIgnoreCase);
        // Names why, so nobody re-derives it from a tensor size.
        Assert.Contains("per-layer", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// The gate is a denylist, so an architecture that is not on it must get past
    /// this check — here it goes on to fail on the missing file instead.
    /// </summary>
    [Theory]
    [InlineData("qwen2")]
    [InlineData("llama")]
    [InlineData("gemma2")]
    [InlineData("")]
    public void OtherArchitectures_AreNotBlocked(string architecture)
        => Assert.IsNotType<NotSupportedException>(Load(architecture));

    /// <summary>
    /// <see cref="SharpMindConfig.ForModel"/> is resolved by the app pipeline
    /// <em>before</em> <see cref="ModelFactory.CreateWeights"/>, so the denylist above
    /// never sees the architecture it is supposed to judge. It used to be the only
    /// coverage, and it passes a prebuilt <c>SharpMindConfig.Gpt</c> — nothing ever
    /// called <c>ForModel</c> with a real architecture string.
    ///
    /// That let <c>ForModel</c> reject any architecture it did not list, which broke
    /// models that had always worked: qwen2 was never listed (it silently used the
    /// catch-all default), so rejecting the catch-all took down qwen2, qwen2vl, phi3,
    /// deepseek and every other family that relied on it. The preset table is a
    /// convenience, not a gate; only the denylist in <c>ModelFactory</c> rejects.
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
    [InlineData("some-future-arch")]
    [InlineData("internlm2")]
    [InlineData("glm4")]
    [InlineData("deepseek3")]
    // gemma4 is on ModelFactory's denylist, but ForModel is a preset table and must
    // still answer for it rather than throw a second, less informative error that
    // pre-empts the denylist's real explanation.
    [InlineData("gemma4")]
    [InlineData("gemma3n")]
    public void ForModel_DoesNotThrow_AndUsesTheDecoderPreset(string architecture)
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
