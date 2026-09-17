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

    [Theory]
    [MemberData(nameof(ExportDtypes))]
    public void StreamingLoad_LogitsMatchFullLoad(QuantDType dtype)
    {
        // #58 follow-up: a streaming load builds its LinearLayers before any layer's bytes
        // are read, so the per-field dtype properties (QuantDtypeWgate/QuantDtypeWup) are
        // still null and LinearLayerFactory used to bake an F32 kernel into the assembled
        // type. The raw bytes are Q8_0, so Forward saw RawQuantizedData != null and called
        // the F32 kernel over a buffer ~3.76x smaller than it reads — an out-of-bounds read
        // (0xc0000005). The construction dtype now comes from TensorMeta, which
        // InitializeWeights populates before the blocks are built.
        string path = ExportGatedModel(dtype);
        SmmLoader.Load(path, null, out _, out var config, out _);
        var qOps = QuantizationFactory.Create(Sharp().ResolvedHardware);

        using var streamingWeights = ModelFactory.CreateWeights(config, Sharp(), qOps, path, LoadMode.Streaming);
        streamingWeights.InitializeWeights();
        using var streaming = ModelFactory.CreateTransformer(streamingWeights, Sharp()); // owns the weights

        using var fullWeights = Load(path, quantizedResident: false);
        using var full = ModelFactory.CreateTransformer(fullWeights, Sharp());

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        using var a = streaming.Forward(tokens);
        using var b = full.Forward(tokens);
        Assert.Equal(b.ElementCount, a.ElementCount);
        for (int i = 0; i < a.ElementCount; i++)
            Assert.True(BitConverter.SingleToInt32Bits(b.Data[i]) == BitConverter.SingleToInt32Bits(a.Data[i]),
                $"{dtype} logit {i}: full={b.Data[i]} streaming={a.Data[i]}");
    }

    [Theory]
    [MemberData(nameof(ExportDtypes))]
    public void StreamingLoad_GemmaStylePostNorms_LogitsMatchFullLoad(QuantDType dtype)
    {
        // Gemma-3 carries post-attention / post-FFN norms in addition to the input
        // norms. They are 1D and have no raw field, so the streaming metadata scan
        // never allocated them; BuildBlock then saw null and built the block without
        // those two NormLayers, so the forward dropped both norms and generated
        // garbage (the full load was fine). Pre-allocating them during the scan lets
        // BuildBlock wire the layers and the per-layer reload fill them.
        string path = ExportGatedModel(dtype, withPostNorms: true);
        SmmLoader.Load(path, null, out _, out var config, out _);
        var qOps = QuantizationFactory.Create(Sharp().ResolvedHardware);

        using var streamingWeights = ModelFactory.CreateWeights(config, Sharp(), qOps, path, LoadMode.Streaming);
        streamingWeights.InitializeWeights();
        Assert.All(streamingWeights.Blocks, b =>
        {
            Assert.NotNull(b.PostNorm1W);
            Assert.NotNull(b.PostNorm2W);
        });
        using var streaming = ModelFactory.CreateTransformer(streamingWeights, Sharp());
        for (int i = 0; i < config.NumLayers; i++)
        {
            var blk = streaming.GetBlock(i)!;
            Assert.NotNull(blk.PostAttnNorm);
            Assert.NotNull(blk.PostFfnNorm);
        }

        using var fullWeights = Load(path, quantizedResident: false);
        using var full = ModelFactory.CreateTransformer(fullWeights, Sharp());

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        using var a = streaming.Forward(tokens);
        using var b = full.Forward(tokens);
        Assert.Equal(b.ElementCount, a.ElementCount);
        for (int i = 0; i < a.ElementCount; i++)
            Assert.True(BitConverter.SingleToInt32Bits(b.Data[i]) == BitConverter.SingleToInt32Bits(a.Data[i]),
                $"{dtype} logit {i}: full={b.Data[i]} streaming={a.Data[i]}");
    }

    [Theory]
    [MemberData(nameof(ExportDtypes))]
    public void StreamingLoad_SecondForwardAfterLayerFree_MatchesFullLoad(QuantDType dtype)
    {
        // Streaming frees every layer at the end of a forward (CompleteForward), and
        // ReleaseLayerData used to dispose and null the block norms too. The built
        // NormLayer captured its tensor at construction and TransformerBlock never
        // repoints it (it only copies data in), so the second forward reloaded the
        // layer but reused the disposed norm tensor and threw ObjectDisposedException
        // on the norm's first read. Run two passes and compare both to the full load.
        string path = ExportGatedModel(dtype);
        SmmLoader.Load(path, null, out _, out var config, out _);
        var qOps = QuantizationFactory.Create(Sharp().ResolvedHardware);

        using var streamingWeights = ModelFactory.CreateWeights(config, Sharp(), qOps, path, LoadMode.Streaming);
        streamingWeights.InitializeWeights();
        using var streaming = ModelFactory.CreateTransformer(streamingWeights, Sharp());

        using var fullWeights = Load(path, quantizedResident: false);
        using var full = ModelFactory.CreateTransformer(fullWeights, Sharp());

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        using var b = full.Forward(tokens);

        for (int pass = 0; pass < 2; pass++)
        {
            using var a = streaming.Forward(tokens);
            Assert.Equal(b.ElementCount, a.ElementCount);
            for (int i = 0; i < a.ElementCount; i++)
                Assert.True(BitConverter.SingleToInt32Bits(b.Data[i]) == BitConverter.SingleToInt32Bits(a.Data[i]),
                    $"{dtype} pass {pass} logit {i}: full={b.Data[i]} streaming={a.Data[i]}");
        }

        // A completed forward frees every layer; the norm tensors the NormLayers
        // captured must survive that so the next pass can reload into them.
        Assert.All(streamingWeights.Blocks, blk => Assert.NotNull(blk.Norm1W));
        Assert.All(streamingWeights.Blocks, blk => Assert.NotNull(blk.Norm2W));
    }

    [Fact]
    public void ReleaseLayerData_KeepsTensorsTheLayersHoldByReference()
    {
        // The streaming free path must drop only the large 2D weights and raw
        // bytes. NormLayer keeps the norm tensor it was built with and LinearLayer
        // keeps the bias tensor; neither is repointed on reload, so releasing them
        // left a disposed NativeBuffer in the forward path.
        string path = ExportGatedModel(QuantDType.Q8_0);
        using var weights = Load(path, quantizedResident: false);
        var block = weights.Blocks[0];

        var norm1 = block.Norm1W;
        var norm2 = block.Norm2W;
        var bias = block.WqBias;
        Assert.NotNull(norm1);
        Assert.NotNull(norm2);
        Assert.NotNull(bias);
        Assert.True(block.RawWq is not null && block.RawWgate is not null, "fixture must carry raw bytes");

        block.ReleaseLayerData();

        Assert.Null(block.Wq);
        Assert.Null(block.RawWq);
        Assert.Null(block.RawWgate);
        Assert.Same(norm1, block.Norm1W);
        Assert.Same(norm2, block.Norm2W);
        Assert.Same(bias, block.WqBias);
    }

    private string ExportGatedModel(QuantDType dtype, bool withPostNorms = false)
    {
        string path = Path.Combine(_temp.Path, $"gated-{dtype}-{Guid.NewGuid():N}.smm");
        using var weights = ModelFactory.CreateForTraining(Cfg(), Sharp());
        WeightInitializer.InitializeRandomly(weights, seed: 1685);
        if (withPostNorms)
        {
            foreach (var b in weights.Blocks)
            {
                b.PostNorm1W = new Tensor<float>(Cfg().HiddenDim);
                b.PostNorm1W.Data.Fill(0.5f + 0.1f * (b.LayerIndex + 1));
                b.PostNorm2W = new Tensor<float>(Cfg().HiddenDim);
                b.PostNorm2W.Data.Fill(1.3f + 0.2f * (b.LayerIndex + 1));
            }
        }
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
