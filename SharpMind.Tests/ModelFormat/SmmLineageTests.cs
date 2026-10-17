using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// A7: an .SMM may carry a lineage block — free-form provenance strings naming
/// where the model came from (base, recipe, dataset and tokenizer fingerprints,
/// the producing tool version, and when it was made). It is emitted under
/// <c>lineage</c> in the meta JSON and read back through
/// <see cref="SmmLoader.LoadLineageFromMeta"/>/<see cref="SmmLoader.LoadLineage"/>.
/// Files with no provenance load unchanged with an empty lineage.
/// </summary>
public sealed class SmmLineageTests : IDisposable
{
    private readonly TempDirectory _temp = new();
    public void Dispose() => _temp.Dispose();

    private static ModelConfig Config => new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 8,
        NormEps = 1e-3f,
    };

    private static SmmTensorData Tensor(string name, int count)
        => new()
        {
            Name = name,
            Shape = [count],
            Dtype = QuantDType.F32,
            GetBytes = () => new byte[count * 4],
        };

    private string Write(string name, SmmWriteOptions options)
    {
        string path = Path.Combine(_temp.Path, name);
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Tensor("token_embd.weight", 64),
            Tensor("output_norm.weight", 16),
        ], options);
        return path;
    }

    [Fact]
    public void Lineage_RoundTrips()
    {
        var expected = new Dictionary<string, string>
        {
            ["base"] = "smollm2-135m@acme",
            ["recipe"] = "sft-lora-r8",
            ["dataset"] = "clientset-v3",
            ["tokenizer"] = "tok-a1b2c3",
            ["tool"] = "SharpMind 1.0.0",
            ["created"] = "2026-10-16T12:00:00Z",
        };

        string path = Write("with-lineage.smm", new SmmWriteOptions
        {
            Source = "training",
            Lineage = expected,
        });

        var lineage = SmmLoader.LoadLineage(path);

        Assert.Equal(expected.Count, lineage.Count);
        foreach (var (key, value) in expected)
            Assert.Equal(value, lineage[key]);
    }

    [Fact]
    public void NoLineage_LoadsWithEmptyMap()
    {
        string path = Write("no-lineage.smm", new SmmWriteOptions { Source = "training" });

        Assert.Empty(SmmLoader.LoadLineage(path));
    }

    [Fact]
    public void EmptyLineageDictionary_WritesNoBlock()
    {
        string path = Write("empty-lineage.smm", new SmmWriteOptions
        {
            Source = "training",
            Lineage = new Dictionary<string, string>(),
        });

        Assert.Empty(SmmLoader.LoadLineage(path));
    }

    [Fact]
    public void FilesWithoutLineage_StillLoadConfigAndTensors()
    {
        // Backward compatibility: the lineage block is additive; a reader that
        // ignores it (every existing path) sees a perfectly normal model.
        string path = Write("compat.smm", new SmmWriteOptions
        {
            Source = "training",
            Lineage = new Dictionary<string, string> { ["base"] = "x" },
        });

        var meta = SmmLoader.LoadMeta(path);
        Assert.NotNull(SmmLoader.LoadConfig(meta));
        Assert.Equal(2, SmmLoader.ReadTensorIndex(path).Count);
    }
}
