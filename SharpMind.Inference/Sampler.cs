using SharpMind.Core.Tensors;

namespace SharpMind.Inference;

/// <summary>
/// Stateless sampling functions. Each method takes raw logits and returns a token ID.
/// </summary>
public static class Sampler
{
    // ── Entry point ───────────────────────────────────────────────────────

    /// <summary>
    /// Selects the next token from <paramref name="logits"/> using the given config.
    /// Applies temperature → top-k → top-p / min-p → sample (or argmax if greedy).
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingConfig config, Random? rng = null)
    {
        // Greedy shortcut — no allocation needed
        if (config.Temperature <= 0f)
            return Argmax(logits);

        // Work on a copy so we don't mutate the model output
        float[] probs = logits.ToArray();

        ApplyTemperature(probs, config.Temperature);
        Softmax(probs);

        if (config.MinP > 0f)           ApplyMinP(probs, config.MinP);
        if (config.TopK > 0)            ApplyTopK(probs, config.TopK);
        if (config.TopP < 1.0f)         ApplyTopP(probs, config.TopP);

        // Renormalise after filtering
        Normalise(probs);

        return SampleFromProbs(probs, rng ?? (config.Seed.HasValue
            ? new Random(config.Seed.Value)
            : Random.Shared));
    }

    // ── Greedy ────────────────────────────────────────────────────────────

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int   best  = 0;
        float max   = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) { max = logits[i]; best = i; }
        return best;
    }

    // ── Temperature ───────────────────────────────────────────────────────

    private static void ApplyTemperature(Span<float> logits, float temperature)
    {
        float inv = 1f / temperature;
        for (int i = 0; i < logits.Length; i++) logits[i] *= inv;
    }

    // ── Softmax ───────────────────────────────────────────────────────────

    private static void Softmax(Span<float> x)
    {
        float max = x[0];
        foreach (float v in x) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < x.Length; i++) x[i] *= inv;
    }

    // ── Top-k ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Zeroes all but the top-k highest-probability tokens.
    /// Uses a partial sort — O(n log k) rather than full sort O(n log n).
    /// </summary>
    private static void ApplyTopK(Span<float> probs, int k)
    {
        if (k >= probs.Length) return;

        // Find the k-th largest value via a min-heap of size k
        var heap = new SortedList<float, int>(k + 1, FloatDescComparer.Instance);
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] <= 0f) continue;
            heap.TryAdd(probs[i], i);
            if (heap.Count > k) heap.RemoveAt(heap.Count - 1);
        }

        // Zero everything not in the top-k set
        var keep = new HashSet<int>(heap.Values);
        for (int i = 0; i < probs.Length; i++)
            if (!keep.Contains(i)) probs[i] = 0f;
    }

    // ── Top-p (nucleus) ───────────────────────────────────────────────────

    /// <summary>
    /// Zeroes tokens outside the nucleus whose cumulative probability exceeds p.
    /// Tokens are sorted descending; we keep until the running sum exceeds p.
    /// </summary>
    private static void ApplyTopP(Span<float> probs, float p)
    {
        if (p <= 0f || probs.IsEmpty) return;

        int n = probs.Length;

        // Build sorted indices list (descending by probability)
        var indices = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (probs[i] > 0f)
            {
                indices.Add(i);
            }
        }

        // Sort descending (simple bubble sort for clarity)
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = i + 1; j < indices.Count; j++)
            {
                if (probs[indices[i]] < probs[indices[j]])
                {
                    (indices[j], indices[i]) = (indices[i], indices[j]);
                }
            }
        }

        // Keep only tokens within cumulative probability threshold
        float cumulative = 0f;
        for (int i = 0; i < indices.Count; i++)
        {
            int idx = indices[i];
            cumulative += probs[idx];

            if (cumulative >= p)
            {
                for (int j = i + 1; j < indices.Count; j++)
                {
                    probs[indices[j]] = 0f;
                }
                break;
            }
        }
    }

    // ── Min-p ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Zeroes tokens whose probability is below min_p × max_probability.
    /// Simpler than top-p and often produces better results for creative tasks.
    /// </summary>
    private static void ApplyMinP(Span<float> probs, float minP)
    {
        float max = 0f;
        foreach (float v in probs) if (v > max) max = v;
        float threshold = minP * max;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] < threshold) probs[i] = 0f;
    }

    // ── Normalise ─────────────────────────────────────────────────────────

    private static void Normalise(Span<float> probs)
    {
        float sum = 0f;
        foreach (float v in probs) sum += v;
        if (sum <= 0f) return;
        float inv = 1f / sum;
        for (int i = 0; i < probs.Length; i++) probs[i] *= inv;
    }

    // ── Categorical sample ────────────────────────────────────────────────

    private static int SampleFromProbs(Span<float> probs, Random rng)
    {
        float r = rng.NextSingle();
        float cumSum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (r <= cumSum) return i;
        }
        // Floating point edge case — return last non-zero token
        for (int i = probs.Length - 1; i >= 0; i--)
            if (probs[i] > 0f) return i;
        return 0;
    }

    private sealed class FloatDescComparer : IComparer<float>
    {
        public static readonly FloatDescComparer Instance = new();
        public int Compare(float x, float y) => y.CompareTo(x);
    }
}
