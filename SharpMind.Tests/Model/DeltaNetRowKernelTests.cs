using SharpMind.Model.Layers.DeltaNet;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The DeltaNet recurrence uses vector row helpers. Each must produce the same
/// bits as the scalar loop it replaces, for every length: the vector body, the
/// scalar tail, and the case where the row is shorter than one vector.
/// </summary>
public class DeltaNetRowKernelTests
{
    private static float[] Random(Random rng, int n) =>
        Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();

    private static void AssertBitsEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"index {i}: expected={expected[i]} actual={actual[i]}");
    }

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(7)] [InlineData(8)]
    [InlineData(9)] [InlineData(16)] [InlineData(17)] [InlineData(64)] [InlineData(128)]
    public void ScaleInPlace_MatchesScalar(int n)
    {
        var rng = new Random(n);
        var x = Random(rng, n);
        var expected = (float[])x.Clone();
        float s = 0.37f;
        for (int i = 0; i < expected.Length; i++) expected[i] *= s;

        DeltaNetLayer.ScaleInPlace(x, s);
        AssertBitsEqual(expected, x);
    }

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(7)] [InlineData(8)]
    [InlineData(9)] [InlineData(16)] [InlineData(17)] [InlineData(64)] [InlineData(128)]
    public void MulAddRow_MatchesScalar(int n)
    {
        var rng = new Random(100 + n);
        var dst = Random(rng, n);
        var row = Random(rng, n);
        var expected = (float[])dst.Clone();
        float s = -1.3f;
        for (int i = 0; i < expected.Length; i++) expected[i] += row[i] * s;

        DeltaNetLayer.MulAddRow(dst, row, s);
        AssertBitsEqual(expected, dst);
    }

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(7)] [InlineData(8)]
    [InlineData(9)] [InlineData(16)] [InlineData(17)] [InlineData(64)] [InlineData(128)]
    public void AddScaledRow_MatchesScalar(int n)
    {
        var rng = new Random(200 + n);
        var dst = Random(rng, n);
        var x = Random(rng, n);
        var expected = (float[])dst.Clone();
        float s = 0.91f;
        for (int i = 0; i < expected.Length; i++) expected[i] += s * x[i];

        DeltaNetLayer.AddScaledRow(dst, x, s);
        AssertBitsEqual(expected, dst);
    }
}
