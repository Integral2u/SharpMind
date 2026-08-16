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
}
