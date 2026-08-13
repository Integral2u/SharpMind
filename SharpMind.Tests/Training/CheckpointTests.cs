using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training;

namespace SharpMind.Tests.Training;

/// <summary>
/// Verifies <see cref="Checkpoint"/> save/load, in particular that parameters
/// with duplicate names (NormLayer yields a bare <c>LayerNormLayer.weight</c>
/// with no layer index) round-trip correctly by occurrence order.
/// </summary>
public sealed class CheckpointTests
{
    [Fact]
    public void RoundTrip_RestoresDuplicateNamedParametersInOrder()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sm-final-checkpoint-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var a1 = Tensor<float>.From([1f, 2f, 3f], 3);
            using var a2 = Tensor<float>.From([4f, 5f], 2);
            using var e1 = Tensor<float>.From([9f], 1);
            using var pA = new Parameter("LayerNormLayer.weight", a1);
            using var pB = new Parameter("LayerNormLayer.weight", a2);
            using var pE = new Parameter("embedding.weight", e1);
            var source = new List<Parameter> { pA, pB, pE };

            Checkpoint.Save(dir, source);

            using var b1 = Tensor<float>.From([0f, 0f, 0f], 3);
            using var b2 = Tensor<float>.From([0f, 0f], 2);
            using var e2 = Tensor<float>.From([0f], 1);
            using var qA = new Parameter("LayerNormLayer.weight", b1);
            using var qB = new Parameter("LayerNormLayer.weight", b2);
            using var qE = new Parameter("embedding.weight", e2);
            var restored = new List<Parameter> { qA, qB, qE };

            var meta = Checkpoint.Load(dir, restored);

            Assert.Equal(new[] { 1f, 2f, 3f }, Copy(qA));
            Assert.Equal(new[] { 4f, 5f }, Copy(qB));
            Assert.Equal(new[] { 9f }, Copy(qE));
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void FindLatest_PicksHighestStepDirectory()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sm-latest-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(dir, "step-0000002"));
            Directory.CreateDirectory(Path.Combine(dir, "step-0000001"));
            Directory.CreateDirectory(Path.Combine(dir, "step-0000009"));
            Directory.CreateDirectory(Path.Combine(dir, "step-0000009-final"));
            Directory.CreateDirectory(Path.Combine(dir, "step-0000123"));
            Directory.CreateDirectory(Path.Combine(dir, "unrelated"));

            // Highest parsed step wins (123 > 9 > 2 > 1); non-step dirs ignored.
            Assert.Equal(Path.Combine(dir, "step-0000123"), Checkpoint.FindLatest(dir));
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void FindLatest_ReturnsNull_WhenMissingOrEmpty()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sm-latest-" + Guid.NewGuid().ToString("N"));
        Assert.Null(Checkpoint.FindLatest(dir));
        try
        {
            Directory.CreateDirectory(dir);
            Directory.CreateDirectory(Path.Combine(dir, "not-a-step"));
            Assert.Null(Checkpoint.FindLatest(dir));
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    private static float[] Copy(Parameter p)
    {
        var span = p.Data.Data;
        var result = new float[span.Length];
        span.CopyTo(result);
        return result;
    }
}