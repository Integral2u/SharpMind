using SharpMind.Inference;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Sampler.Sample accepts an optional Random so a caller drives one sequence
/// across a generation. When none is passed and the config carries a Seed, the
/// sampler must still advance a single stream per thread/seed — re-creating
/// `new Random(seed)` on every draw would pick the same quantile each step.
/// </summary>
public sealed class SamplerTests
{
    /// <summary>
    /// The fused temperature/softmax path must pick exactly the token the plain
    /// reference pipeline would, for the same RNG draw. Covers the unfiltered
    /// configuration (no top-k/top-p/min-p), which is where the fused pass runs.
    /// </summary>
    [Theory]
    [InlineData(0.5f)]
    [InlineData(1.0f)]
    [InlineData(1.7f)]
    public void FusedSoftmax_MatchesReferencePipeline(float temperature)
    {
        var data = new Random(7);
        for (int trial = 0; trial < 300; trial++)
        {
            int n = 1 + data.Next(200);
            var logits = new float[n];
            for (int i = 0; i < n; i++) logits[i] = (float)(data.NextDouble() * 40 - 20);

            int seed = 1000 + trial;
            var config = new SamplingConfig { Temperature = temperature, TopK = 0, TopP = 1.0f, Seed = seed };

            int actual = Sampler.Sample(logits, config, new Random(seed));
            int expected = ReferenceSample(logits, temperature, new Random(seed));
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void TopK_NeverSamplesOutsideTheKLargestLogits()
    {
        var data = new Random(11);
        var logits = new float[64];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(data.NextDouble() * 10);

        var top3 = logits.Select((v, i) => (v, i)).OrderByDescending(p => p.v).Take(3).Select(p => p.i).ToHashSet();
        var config = new SamplingConfig { Temperature = 1.0f, TopK = 3, TopP = 1.0f };

        for (int seed = 0; seed < 500; seed++)
            Assert.Contains(Sampler.Sample(logits, config, new Random(seed)), top3);
    }

    [Fact]
    public void TopP_KeepsOnlyTheNucleusOfTheRenormalisedCandidates()
    {
        // Softmax of these logits is exactly [0.6, 0.3, 0.1]. The top token alone
        // reaches p=0.5, so the nucleus is {0} and every draw must be token 0.
        var logits = new float[] { MathF.Log(0.6f), MathF.Log(0.3f), MathF.Log(0.1f) };
        var config = new SamplingConfig { Temperature = 1.0f, TopK = 0, TopP = 0.5f };

        for (int seed = 0; seed < 200; seed++)
            Assert.Equal(0, Sampler.Sample(logits, config, new Random(seed)));
    }

    [Fact]
    public void NonFiniteLogit_CollapsesToTokenZero()
    {
        var logits = new float[] { 0.1f, float.NaN, 0.3f };
        var config = new SamplingConfig { Temperature = 1.0f, TopK = 0, TopP = 1.0f, Seed = 1 };
        Assert.Equal(0, Sampler.Sample(logits, config, new Random(1)));
    }

    /// <summary>Naive softmax over logits/temperature followed by the categorical draw.</summary>
    private static int ReferenceSample(float[] logits, float temperature, Random rng)
    {
        double max = double.NegativeInfinity;
        var scaled = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = logits[i] / (double)temperature;
            max = Math.Max(max, scaled[i]);
        }
        double sum = 0;
        var probs = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = Math.Exp(scaled[i] - max);
            sum += probs[i];
        }
        float r = rng.NextSingle();
        double cum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i] / sum;
            if (r <= cum) return i;
        }
        return probs.Length - 1;
    }

    [Fact]
    public void SeededSampling_WithoutExplicitRng_AdvancesTheStream()
    {
        var config = new SamplingConfig { Temperature = 1.0f, TopK = 0, TopP = 1.0f, Seed = 12345 };
        var logits = new float[32];
        for (int i = 0; i < logits.Length; i++) logits[i] = 0.5f; // uniform → every draw is a dice roll

        // If each call re-derived `new Random(seed)`, the quantile would be fixed
        // and every draw would land on the same token. An advancing stream must
        // spread across the vocabulary.
        var seen = new HashSet<int>();
        for (int i = 0; i < 200; i++)
            seen.Add(Sampler.Sample(logits, config));

        Assert.True(seen.Count > 5, "Expected a seeded sampler without an explicit RNG to spread across tokens; got " + seen.Count + " distinct tokens.");
    }

    [Fact]
    public void SeededSampling_FirstDraw_MatchesFreshSeededRandom()
    {
        // Byte-for-byte parity with an explicitly-seeded Random on the very first
        // draw guarantees the cache did not shift the starting point.
        var config = new SamplingConfig { Temperature = 1.0f, TopK = 0, TopP = 1.0f, Seed = 4242 };
        var logits = new float[16];
        for (int i = 0; i < logits.Length; i++) logits[i] = 0.5f;

        var fresh = new Random(4242);
        int expected = default;
        {
            // Replicate Sampler.SampleFromProbs' categorical draw over a uniform
            // distribution exactly: temperature 1, softmax of uniform stays uniform.
            float r = fresh.NextSingle();
            float cum = 0f;
            for (int i = 0; i < 16; i++)
            {
                cum += 1f / 16f;
                if (r <= cum) { expected = i; break; }
            }
        }

        Assert.Equal(expected, Sampler.Sample(logits, config));
    }

    [Fact]
    public void Greedy_IgnoresSeed_AndPicksArgmax()
    {
        var config = new SamplingConfig { Temperature = 0f, Seed = 7 };
        var logits = new float[8];
        for (int i = 0; i < logits.Length; i++) logits[i] = -i;
        logits[4] = 10f;

        Assert.Equal(4, Sampler.Sample(logits, config));
    }
}