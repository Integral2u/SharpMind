using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// <see cref="Checkpoint.LoadForWarmRestart"/> copies whatever embedding rows
/// still fit, leaves the new rows untouched (the caller initialises them), and
/// keeps every other tensor strictly matched. This is the weight half of a
/// vocabulary-growing warm restart.
/// </summary>
public sealed class CheckpointWarmRestartTests
{
    private const string Embedding = "EmbeddingTable.weight";

    [Fact]
    public void LoadForWarmRestart_CopiesOverlappingRows_AndLeavesNewRowsAlone()
    {
        string dir = Temp();
        try
        {
            using var src = Tensor<float>.From([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f], 4, 3);
            using var pEmbed = new Parameter(Embedding, src);
            using var pOther = new Parameter("layer.weight", Tensor<float>.From([42f, 43f], 2));
            Checkpoint.Save(dir, [pEmbed, pOther], step: 7);

            using var dst = new Tensor<float>(6, 3); // all zeros
            using var qEmbed = new Parameter(Embedding, dst);
            using var qOther = new Parameter("layer.weight", new Tensor<float>(2));
            var meta = Checkpoint.LoadForWarmRestart(dir, [qEmbed, qOther], Embedding);

            Assert.Equal(7, meta.Step);

            // First four rows copied verbatim.
            var data = qEmbed.Data.Data;
            Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f },
                data[..12].ToArray());
            // New rows left untouched.
            Assert.Equal(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, data[12..].ToArray());

            // Non-growable tensors still load exactly.
            Assert.Equal(new[] { 42f, 43f }, qOther.Data.Data.ToArray());
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void LoadForWarmRestart_Throws_WhenEmbeddingWidthsDiffer()
    {
        string dir = Temp();
        try
        {
            using var src = Tensor<float>.From(new float[12], 4, 3);
            using var pEmbed = new Parameter(Embedding, src);
            Checkpoint.Save(dir, [pEmbed]);

            using var dst = new Tensor<float>(6, 4);
            using var qEmbed = new Parameter(Embedding, dst);

            var ex = Assert.Throws<InvalidDataException>(
                () => Checkpoint.LoadForWarmRestart(dir, [qEmbed], Embedding));
            Assert.Contains("columns", ex.Message);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void LoadForWarmRestart_Throws_WhenOtherTensorSizeMismatches()
    {
        string dir = Temp();
        try
        {
            using var pEmbed = new Parameter(Embedding, Tensor<float>.From(new float[12], 4, 3));
            using var pOther = new Parameter("layer.weight", Tensor<float>.From([1f, 2f, 3f], 3));
            Checkpoint.Save(dir, [pEmbed, pOther]);

            using var qEmbed = new Parameter(Embedding, new Tensor<float>(5, 3));
            using var qOther = new Parameter("layer.weight", new Tensor<float>(2));

            var ex = Assert.Throws<InvalidDataException>(
                () => Checkpoint.LoadForWarmRestart(dir, [qEmbed, qOther], Embedding));
            Assert.Contains("layer.weight", ex.Message);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void LoadForWarmRestart_Throws_WhenGrowableTensorMissing()
    {
        string dir = Temp();
        try
        {
            using var pOther = new Parameter("layer.weight", Tensor<float>.From([1f, 2f], 2));
            Checkpoint.Save(dir, [pOther]);

            using var qEmbed = new Parameter(Embedding, new Tensor<float>(2, 2));
            var ex = Assert.Throws<InvalidDataException>(
                () => Checkpoint.LoadForWarmRestart(dir, [qEmbed], Embedding));
            Assert.Contains(Embedding, ex.Message);
        }
        finally { Cleanup(dir); }
    }

    [Fact]
    public void InitializeNewEmbeddingRows_FillsOnlyNewRows_AndLeavesOldOnesUntouched()
    {
        using var embedding = new Tensor<float>(6, 3);
        var data = embedding.Data;

        // Give the existing rows a recognisable, non-trivial spread.
        for (int i = 0; i < 12; i++) data[i] = (i % 3) - 1f;
        var before = data[..12].ToArray();

        WeightInitializer.InitializeNewEmbeddingRows(embedding, existingRows: 4, seed: 123);

        Assert.Equal(before, data[..12].ToArray());

        bool anyNonZero = false;
        foreach (float v in data[12..])
        {
            Assert.True(float.IsFinite(v));
            if (v != 0) anyNonZero = true;
        }
        Assert.True(anyNonZero, "new embedding rows should be initialised, not left at zero");
    }

    private static string Temp() =>
        Path.Combine(Path.GetTempPath(), "sm-warm-" + Guid.NewGuid().ToString("N"));

    private static void Cleanup(string dir)
    {
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
}
