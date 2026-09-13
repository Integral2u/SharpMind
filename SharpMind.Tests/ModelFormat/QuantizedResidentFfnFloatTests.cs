using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Training;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// A quantized-resident load keeps the gate/up weights as raw bytes: the fused gated layer runs
/// off them and never reads a float copy. The float target for ffn_gate/ffn_up used to be
/// created on demand anyway, so every layer dequantized both tensors into a dead
/// [hidden, 2 * ffn] float tensor — 0.9 GB on a 0.6B model, 3.6 GB on a 1.5B once the
/// power-of-two allocation rounding is counted. Blocks whose gate/up arrived as raw bytes must
/// not get one, blocks without raw bytes must still get one, and the logits must be unchanged.
/// </summary>
public sealed class QuantizedResidentFfnFloatTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    public static IEnumerable<object[]> ExportDtypes()
    {
        yield return [QuantDType.F32];
        yield return [QuantDType.F16];
        yield return [QuantDType.Q8_0];
    }

    private static ModelConfig Cfg() => new()
    {
        Architecture = "qwen2",
        // Every tensor dimension a multiple of 32, so the Q8_0 export can block-quantize it.
        VocabSize = 64,
        HiddenDim = 64,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 128,
        MaxSeqLen = 32,
    };

    private static SharpMindConfig Sharp() => SharpMindConfig.Qwen with { Hardware = HardwareTier.Scalar };

    [Theory]
    [MemberData(nameof(ExportDtypes))]
    public void QuantizedResidentLoad_DoesNotMaterialiseGateUpFloats(QuantDType dtype)
    {
        string path = ExportGatedModel(dtype);
        using var weights = Load(path, quantizedResident: true);

        Assert.All(weights.Blocks, b =>
        {
            Assert.True(b.RawWgate is not null || b.RawWup is not null, "fixture must carry raw gate/up bytes");
            Assert.Null(b.Wf1);
        });
    }

    /// <summary>The float target is still created for a block whose gate/up came without raw bytes.</summary>
    [Fact]
    public void BlockWithoutRawGateUp_StillGetsAFloatTarget()
    {
        string path = ExportGatedModel(QuantDType.F32);
        using var weights = Load(path, quantizedResident: true);
        var block = weights.Blocks[0];
        byte[]? gate = block.RawWgate, up = block.RawWup;

        Assert.Null(weights.ResolveFloatTarget("blk.0.ffn_up.weight"));

        block.RawWgate = null;
        block.RawWup = null;
        var target = weights.ResolveFloatTarget("blk.0.ffn_up.weight");
        Assert.NotNull(target);
        Assert.Equal(new[] { Cfg().HiddenDim, 2 * Cfg().FfnDim }, target!.Shape.Dims.ToArray());

        block.RawWgate = gate;
        block.RawWup = up;
    }

    [Theory]
    [MemberData(nameof(ExportDtypes))]
    public void QuantizedResidentLoad_LogitsMatchFullLoad(QuantDType dtype)
    {
        string path = ExportGatedModel(dtype);
        using var residentWeights = Load(path, quantizedResident: true);
        using var fullWeights = Load(path, quantizedResident: false);
        var mapping = Sharp().ToJigSawMapping();
        using var resident = ModelFactory.CreateTransformer(residentWeights, Sharp(), mapping);
        using var full = ModelFactory.CreateTransformer(fullWeights, Sharp(), mapping);

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        using var a = resident.Forward(tokens);
        using var b = full.Forward(tokens);
        Assert.Equal(b.ElementCount, a.ElementCount);
        for (int i = 0; i < a.ElementCount; i++)
            Assert.True(BitConverter.SingleToInt32Bits(b.Data[i]) == BitConverter.SingleToInt32Bits(a.Data[i]),
                $"{dtype} logit {i}: full={b.Data[i]} resident={a.Data[i]}");
    }

    private string ExportGatedModel(QuantDType dtype)
    {
        string path = Path.Combine(_temp.Path, $"gated-{dtype}-{Guid.NewGuid():N}.smm");
        using var weights = ModelFactory.CreateForTraining(Cfg(), Sharp());
        WeightInitializer.InitializeRandomly(weights, seed: 1685);
        var options = dtype == QuantDType.F32
            ? new SmmWriteOptions { Source = "training" }
            : new SmmWriteOptions { Source = "training", QuantizationLevel = dtype };
        SmmTrainingExporter.Export(weights, tokenizer: null, path, options);
        return path;
    }

    private static TransformerWeights Load(string path, bool quantizedResident)
    {
        SmmLoader.Load(path, null, out _, out var config, out _);
        var qOps = QuantizationFactory.Create(Sharp().ResolvedHardware);
        var weights = ModelFactory.CreateWeights(config, Sharp(), qOps, path, LoadMode.Full, quantizedResident: quantizedResident);
        weights.InitializeWeights();
        return weights;
    }
}
