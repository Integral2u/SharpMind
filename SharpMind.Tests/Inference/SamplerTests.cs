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