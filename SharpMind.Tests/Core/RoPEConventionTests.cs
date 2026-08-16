using SharpMind.Core.Embeddings;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers.Attention;
using Xunit;

namespace SharpMind.Tests.Core;

/// <summary>
/// RoPE has two incompatible pairing conventions and the right one depends on
/// the model architecture:
///
///   adjacent ("NORM", GPT-J/ggml): rotates (2i, 2i+1)
///   NeoX     (HF "rotate_half"):   rotates (d, d + ropeDim/2)
///
/// llama.cpp fixes this per architecture rather than storing it in the GGUF.
/// LLaMA-family files are converted with Q/K permuted so adjacent pairing
/// reproduces rotate_half; Qwen/Gemma/Phi/StableLM are not permuted and need
/// true NeoX. Applying adjacent pairing to a Qwen2 model leaves it locally
/// fluent but unable to use position — it repeats and cannot recall facts
/// ("The capital of France is" -> " a city in the north of the of the of").
/// </summary>
public sealed class RoPEConventionTests
{
    private const int HeadDim = 4;
    private const float Theta = 10_000f;

    // freqs[i] = theta^(-2i/headDim)  ->  [1.0, 0.01] for headDim 4
    private static readonly float Freq0 = 1f;
    private static readonly float Freq1 = 1f / MathF.Pow(Theta, 2f * 1 / HeadDim);

    /// <summary>[SeqLen=1, NumHeads=1, HeadDim=4] holding 1,2,3,4.</summary>
    private static Tensor<float> Input() => Tensor<float>.From([1f, 2f, 3f, 4f], 1, 1, HeadDim);

    [Fact]
    public void Adjacent_RotatesPairs_2i_2iPlus1()
    {
        var rope = new RoPE(HeadDim, maxSeqLen: 8, Theta, neoxStyle: false);
        using var x = Input();

        rope.Apply(x, positionOffset: 1);

        float c0 = MathF.Cos(Freq0), s0 = MathF.Sin(Freq0);
        float c1 = MathF.Cos(Freq1), s1 = MathF.Sin(Freq1);

        Assert.Equal(1f * c0 - 2f * s0, x.Data[0], precision: 4);
        Assert.Equal(2f * c0 + 1f * s0, x.Data[1], precision: 4);
        Assert.Equal(3f * c1 - 4f * s1, x.Data[2], precision: 4);
        Assert.Equal(4f * c1 + 3f * s1, x.Data[3], precision: 4);
    }

    [Fact]
    public void Neox_RotatesPairs_d_dPlusHalf()
    {
        var rope = new RoPE(HeadDim, maxSeqLen: 8, Theta, neoxStyle: true);
        using var x = Input();

        rope.Apply(x, positionOffset: 1);

        float c0 = MathF.Cos(Freq0), s0 = MathF.Sin(Freq0);
        float c1 = MathF.Cos(Freq1), s1 = MathF.Sin(Freq1);

        // pairs are (0,2) and (1,3)
        Assert.Equal(1f * c0 - 3f * s0, x.Data[0], precision: 4);
        Assert.Equal(2f * c1 - 4f * s1, x.Data[1], precision: 4);
        Assert.Equal(3f * c0 + 1f * s0, x.Data[2], precision: 4);
        Assert.Equal(4f * c1 + 2f * s1, x.Data[3], precision: 4);
    }

    [Fact]
    public void Conventions_Differ_AtNonZeroPositions()
    {
        var adjacent = new RoPE(HeadDim, maxSeqLen: 8, Theta, neoxStyle: false);
        var neox = new RoPE(HeadDim, maxSeqLen: 8, Theta, neoxStyle: true);
        using var a = Input();
        using var n = Input();

        adjacent.Apply(a, positionOffset: 1);
        neox.Apply(n, positionOffset: 1);

        Assert.NotEqual(a.Data[0], n.Data[0], precision: 3);
    }

    [Fact]
    public void Position0_IsIdentity_ForBothConventions()
    {
        foreach (bool neox in new[] { false, true })
        {
            var rope = new RoPE(HeadDim, maxSeqLen: 8, Theta, neoxStyle: neox);
            using var x = Input();

            rope.Apply(x, positionOffset: 0);

            Assert.Equal(1f, x.Data[0], precision: 5);
            Assert.Equal(2f, x.Data[1], precision: 5);
            Assert.Equal(3f, x.Data[2], precision: 5);
            Assert.Equal(4f, x.Data[3], precision: 5);
        }
    }

    /// <summary>
    /// The cos/sin tables are shared through a static cache. It used to be keyed
    /// by HashCode.Combine of the config, so a 32-bit collision would hand a model
    /// another configuration's rotary tables. Distinct configs must stay distinct.
    /// </summary>
    [Fact]
    public void DifferentConfigs_DoNotShareCachedTables()
    {
        using var a = Input();
        using var b = Input();
        using var c = Input();

        new RoPE(HeadDim, maxSeqLen: 8, theta: 10_000f).Apply(a, positionOffset: 1);
        new RoPE(HeadDim, maxSeqLen: 8, theta: 500_000f).Apply(b, positionOffset: 1);
        // Same theta as the first, so it must reproduce the first result exactly.
        new RoPE(HeadDim, maxSeqLen: 8, theta: 10_000f).Apply(c, positionOffset: 1);

        // Pair 0 rotates at freq theta^0 == 1 for every theta, so only pair 1
        // (indices 2 and 3) distinguishes the two configurations.
        Assert.NotEqual(a.Data[2], b.Data[2], precision: 4);
        Assert.Equal(a.Data[2], c.Data[2], precision: 6);
        Assert.Equal(a.Data[3], c.Data[3], precision: 6);
    }

    [Theory]
    [InlineData("qwen2", true)]
    [InlineData("qwen3", true)]
    [InlineData("gemma2", true)]
    [InlineData("phi3", true)]
    [InlineData("stablelm", true)]
    [InlineData("llama", false)]
    [InlineData("baichuan", false)]
    [InlineData("", false)]
    [InlineData(null, false)]
    public void ArchitectureSelectsConvention(string? architecture, bool expected)
        => Assert.Equal(expected, AttentionLayer.UsesNeoxRope(architecture));
}
