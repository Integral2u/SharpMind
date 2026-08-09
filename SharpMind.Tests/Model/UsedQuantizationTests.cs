using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.Model;

/// <summary>
/// Verifies <see cref="TransformerWeights.GetUsedQuantizations"/> (and the
/// <see cref="Transformer"/> passthrough) reports the distinct dtypes used by a
/// model's weight tensors, from raw dtype fields, per-block dtype fields,
/// expert dictionaries, metadata, and the all-float fallback.
/// </summary>
public sealed class UsedQuantizationTests
{
    private static ModelConfig Cfg() => new()
    {
        VocabSize = 64,
        HiddenDim = 8,
        NumLayers = 2,
        NumHeads = 1,
        NumKvHeads = 1,
        FfnDim = 16,
        MaxSeqLen = 64,
    };

    private static SharpMindConfig Scalar() => SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

    private static ModelMetaData Meta(params (string Name, QuantDType Dtype)[] tensors) => new()
    {
        Tensors = tensors.Select(t => new TensorInfo
        {
            Name = t.Name,
            Dtype = t.Dtype,
            Shape = [8, 8],
            Offset = 0,
        }).ToList(),
    };

    [Fact]
    public void AllFloatWeights_ReturnF32()
    {
        using var weights = ModelFactory.CreateForTraining(Cfg(), Scalar());
        var used = weights.GetUsedQuantizations();

        Assert.Equal([QuantDType.F32], used);
    }

    [Fact]
    public void QuantizedEmbedding_And_Blocks_AreReported()
    {
        using var weights = ModelFactory.CreateForTraining(Cfg(), Scalar());

        weights.RawEmbedding = new byte[] { 0, 1, 2, 3 };
        weights.RawEmbeddingDtype = QuantDType.F16;

        weights.Blocks[0].RawWq = new byte[] { 9 };
        weights.Blocks[0].QuantDtypeWq = QuantDType.Q8_0;
        weights.Blocks[1].RawWf1 = new byte[] { 8 };
        weights.Blocks[1].QuantDtypeWf1 = QuantDType.Q4_0;

        var used = weights.GetUsedQuantizations();

        // F32 comes from the remaining of the resident float blocks
        // (Wq/Wk/Wv/Wo/Ffn still resident with no dtype); the reported dtype
        // fields surface on top.
        Assert.Equal([QuantDType.F32, QuantDType.F16, QuantDType.Q4_0, QuantDType.Q8_0], used);
    }

    [Fact]
    public void MetaDictionary_IsConsulted_TogetherWithDtypeFields()
    {
        using var weights = ModelFactory.CreateForTraining(Cfg(), Scalar());

        weights.Blocks[0].RawWk = new byte[] { 1 };
        weights.Blocks[0].QuantDtypeWk = QuantDType.Q8_0;
        TransformerWeights.SetTensorMeta(weights.Blocks[0], "RawWv", offset: 0, size: 16, QuantDType.Q4_0);

        var used = weights.GetUsedQuantizations();

        Assert.Equal([QuantDType.F32, QuantDType.Q4_0, QuantDType.Q8_0], used);
    }

    [Fact]
    public void ExpertDtypeDictionaries_AreCollected()
    {
        using var weights = ModelFactory.CreateForTraining(Cfg(), Scalar());

        weights.Blocks[0].QuantDtypeWupExp = new Dictionary<int, QuantDType> { [0] = QuantDType.Q5_0 };
        weights.Blocks[1].QuantDtypeWdownExp = new Dictionary<int, QuantDType> { [2] = QuantDType.Q6_K };

        var used = weights.GetUsedQuantizations();

        Assert.Equal([QuantDType.F32, QuantDType.Q5_0, QuantDType.Q6_K], used);
    }

    [Fact]
    public void SortedAndDistinct_AcrossBlocksAndGlobal()
    {
        using var weights = ModelFactory.CreateForTraining(Cfg(), Scalar());

        weights.RawEmbeddingDtype = QuantDType.F16;
        weights.Blocks[0].QuantDtypeWo = QuantDType.Q8_0;
        weights.Blocks[1].QuantDtypeWo = QuantDType.Q8_0;

        var used = weights.GetUsedQuantizations();

        Assert.Equal([QuantDType.F32, QuantDType.F16, QuantDType.Q8_0], used);
    }

    [Fact]
    public void MetaData_CollectsDistinctDtypesFromTensors()
    {
        var meta = Meta(
            ("token_embd.weight", QuantDType.F16),
            ("blk.0.attn_q.weight", QuantDType.Q8_0),
            ("blk.0.attn_k.weight", QuantDType.Q8_0),
            ("blk.0.ffn_gate.weight", QuantDType.F16));

        Assert.Equal([QuantDType.F16, QuantDType.Q8_0], meta.GetUsedQuantizations());
    }

    [Fact]
    public void MetaData_EmptyTensors_ReturnsEmpty()
    {
        var meta = Meta();

        Assert.Empty(meta.GetUsedQuantizations());
    }
}