using System;
using System.IO;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// <c>FfnKind</c> selects the layer type, so it has to follow what is in the file
/// rather than the architecture string: a routed file built as a gated FFN cannot
/// bind its expert planes, and the error surfaces far from the table that caused
/// it. These pin both halves - the tensor-directory signal the loader records and
/// the reconciliation that turns it into the FFN kind the builder uses.
/// </summary>
public sealed class FileDerivedFfnTests
{
    private static ModelConfig Config(bool routed) => new()
    {
        // "llama" is listed as a gated family; the file still has the last word.
        Architecture = "llama",
        VocabSize = 32,
        HiddenDim = 16,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 32,
        MaxSeqLen = 32,
        NumExperts = routed ? 4 : 1,
        TopKExperts = 1,
        ExpertFfnDim = routed ? 8 : 0,
        HasRoutedExperts = routed,
    };

    [Theory]
    [InlineData(FfnKind.Dense)]
    [InlineData(FfnKind.Gated)]
    public void RoutedFile_OverridesDenseAndGatedPresets(FfnKind preset) =>
        Assert.Equal(FfnKind.MoE, ModelFactory.ResolveFfnKind(Config(routed: true), preset));

    [Theory]
    [InlineData(FfnKind.Dense)]
    [InlineData(FfnKind.Gated)]
    [InlineData(FfnKind.MoE)]
    public void DenseFile_KeepsThePreset(FfnKind preset) =>
        Assert.Equal(preset, ModelFactory.ResolveFfnKind(Config(routed: false), preset));

    [Theory]
    [InlineData("blk.0.ffn_gate_exps.weight", true)]    // fused plane, qwen2moe/llama4 spelling
    [InlineData("blk.0.ffn_up_exps.weight", true)]
    [InlineData("blk.0.ffn_gate.exps.3.weight", true)]  // per-expert export
    [InlineData("blk.0.ffn_gate_inp.weight", true)]     // router without an exps segment
    [InlineData("blk.0.ffn_gate.weight", false)]        // plain gated FFN
    [InlineData("blk.0.ffn_up.weight", false)]
    public void GgufLoader_RecordsRoutedExpertsFromTheTensorDirectory(string tensorName, bool expected)
    {
        string path = Path.Combine(Path.GetTempPath(), $"FileDerivedFfn_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensor = new GgufTensor
            {
                Name = tensorName,
                Shape = [16, 32],
                Dtype = QuantDType.F32,
                GetBytes = () => new byte[16 * 32 * 4],
            };
            GgufWriter.Write(
                path,
                [new GgufKvPair { Key = "general.architecture", Value = "llama" }],
                [tensor]);

            var config = GgufLoader.LoadConfig(GgufLoader.LoadMeta(path));

            Assert.NotNull(config);
            Assert.Equal(expected, config!.HasRoutedExperts);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
