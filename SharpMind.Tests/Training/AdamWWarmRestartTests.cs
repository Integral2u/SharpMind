using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using Xunit;

namespace SharpMind.Tests.Training;

public sealed class AdamWWarmRestartTests
{
    [Fact]
    public void LoadStateForWarmRestart_CopiesOverlappingEmbeddingRows_AndZerosNewRows()
    {
        using var sourceEmbedding = Tensor<float>.Ones(2, 2);
        using var sourceOther = Tensor<float>.Ones(2);
        using var sourceEmbParam = new Parameter("EmbeddingTable.weight", sourceEmbedding);
        using var sourceOtherParam = new Parameter("other.weight", sourceOther);
        using var sourceOptimizer = new AdamW(
            [sourceEmbParam, sourceOtherParam], lr: 0.1f, weightDecay: 0f);
        sourceEmbParam.Grad.Data.Fill(0.5f);
        sourceOtherParam.Grad.Data.Fill(-0.25f);
        sourceOptimizer.Update();
        sourceOptimizer.Update();

        byte[] sourceBytes = Save(sourceOptimizer);
        var (_, _, sourceArrays) = Read(sourceBytes, [4, 2]);

        using var targetEmbedding = Tensor<float>.Ones(3, 2);
        using var targetOther = Tensor<float>.Ones(2);
        using var targetEmbParam = new Parameter("EmbeddingTable.weight", targetEmbedding);
        using var targetOtherParam = new Parameter("other.weight", targetOther);
        using var targetOptimizer = new AdamW(
            [targetEmbParam, targetOtherParam], lr: 0.1f, weightDecay: 0f);

        using (var reader = new BinaryReader(new MemoryStream(sourceBytes)))
        {
            targetOptimizer.LoadStateForWarmRestart(reader, step: 7,
            [
                new ParameterLayout("EmbeddingTable.weight", [2, 2]),
                new ParameterLayout("other.weight", [2]),
            ]);
        }

        byte[] targetBytes = Save(targetOptimizer);
        var (step, lr, targetArrays) = Read(targetBytes, [6, 2]);
        Assert.Equal(7, step);
        Assert.Equal(0.1f, lr, precision: 6);

        // Embedding row 0/1 (4 floats) migrated; new row 2 stays zero.
        Assert.Equal(sourceArrays[0].Take(4), targetArrays[0].Take(4));
        Assert.Equal(sourceArrays[1].Take(4), targetArrays[1].Take(4));
        Assert.All(targetArrays[0].Skip(4), v => Assert.Equal(0f, v));
        Assert.All(targetArrays[1].Skip(4), v => Assert.Equal(0f, v));

        // Unchanged parameter state is copied verbatim.
        Assert.Equal(sourceArrays[2], targetArrays[2]);
        Assert.Equal(sourceArrays[3], targetArrays[3]);
    }

    [Fact]
    public void CheckpointLoadForWarmRestart_MigratesOptimizerState()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sm-warm-opt-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var sourceEmbedding = Tensor<float>.Ones(2, 2);
            using var sourceOther = Tensor<float>.Ones(2);
            using var sourceEmbParam = new Parameter("EmbeddingTable.weight", sourceEmbedding);
            using var sourceOtherParam = new Parameter("other.weight", sourceOther);
            using var sourceOptimizer = new AdamW(
                [sourceEmbParam, sourceOtherParam], lr: 0.1f, weightDecay: 0f);
            sourceEmbParam.Grad.Data.Fill(0.5f);
            sourceOtherParam.Grad.Data.Fill(-0.25f);
            sourceOptimizer.Update();
            sourceOptimizer.Update();
            Checkpoint.Save(dir, [sourceEmbParam, sourceOtherParam], sourceOptimizer, step: 7);
            var (_, _, sourceArrays) = Read(Save(sourceOptimizer), [4, 2]);

            using var targetEmbedding = Tensor<float>.Ones(3, 2);
            using var targetOther = Tensor<float>.Ones(2);
            using var targetEmbParam = new Parameter("EmbeddingTable.weight", targetEmbedding);
            using var targetOtherParam = new Parameter("other.weight", targetOther);
            using var targetOptimizer = new AdamW(
                [targetEmbParam, targetOtherParam], lr: 0.1f, weightDecay: 0f);

            var meta = Checkpoint.LoadForWarmRestart(
                dir, [targetEmbParam, targetOtherParam], "EmbeddingTable.weight", targetOptimizer);

            Assert.Equal(7, meta.Step);
            Assert.Equal(sourceEmbedding.Data[0], targetEmbedding.Data[0]);
            var (step, _, targetArrays) = Read(Save(targetOptimizer), [6, 2]);
            Assert.Equal(7, step);
            Assert.Equal(sourceArrays[0].Take(4), targetArrays[0].Take(4));
            Assert.Equal(sourceArrays[1].Take(4), targetArrays[1].Take(4));
            Assert.All(targetArrays[0].Skip(4), v => Assert.Equal(0f, v));
            Assert.All(targetArrays[1].Skip(4), v => Assert.Equal(0f, v));
            Assert.Equal(sourceArrays[2], targetArrays[2]);
            Assert.Equal(sourceArrays[3], targetArrays[3]);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    private static byte[] Save(IOptimizer optimizer)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            optimizer.SaveState(writer);
        return stream.ToArray();
    }

    private static (int Step, float Lr, List<float[]> Arrays) Read(byte[] bytes, int[] lengths)
    {
        using var reader = new BinaryReader(new MemoryStream(bytes));
        int step = reader.ReadInt32();
        float lr = reader.ReadSingle();
        var arrays = new List<float[]>();
        foreach (int length in lengths)
        {
            arrays.Add(ReadFloats(reader, length));
            arrays.Add(ReadFloats(reader, length));
        }
        return (step, lr, arrays);
    }

    private static float[] ReadFloats(BinaryReader reader, int count)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = reader.ReadSingle();
        return values;
    }
}
