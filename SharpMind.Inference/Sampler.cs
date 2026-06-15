using System.Buffers;

namespace SharpMind.Inference;

/// <summary>
/// Stateless sampling functions. Each method takes raw logits and returns a token ID.
/// </summary>
public static class Sampler
{
    // Entry point

    /// <summary>
    /// Selects the next token from <paramref name="logits"/> using the given config.
    /// Applies temperature → top-k → top-p / min-p → sample (or argmax if greedy).
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingConfig config, Random? rng = null)
    {
        // Greedy shortcut — no allocation needed
        if (config.Temperature <= 0f)
            return Argmax(logits);

        // Work on a copy so we don't mutate the model output — ArrayPool avoids a per-step GC allocation.
        int n = logits.Length;
        float[] rented = ArrayPool<float>.Shared.Rent(n);
        try
        {
            Span<float> probs = rented.AsSpan(0, n);
            logits.CopyTo(probs);

            ApplyTemperature(probs, config.Temperature);
            Softmax(probs);

            if (config.MinP > 0f) ApplyMinP(probs, config.MinP);
            if (config.TopK > 0) ApplyTopK(probs, config.TopK);
            if (config.TopP < 1.0f) ApplyTopP(probs, config.TopP);

            Normalise(probs);

            return SampleFromProbs(probs, rng ?? (config.Seed.HasValue
                ? new Random(config.Seed.Value)
                : Random.Shared));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    // Greedy

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int   best  = 0;
        float max   = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) { max = logits[i]; best = i; }
        return best;
    }

    // Temperature

    private static void ApplyTemperature(Span<float> logits, float temperature)
    {
        float inv = 1f / temperature;
        for (int i = 0; i < logits.Length; i++) logits[i] *= inv;
    }

    // Softmax

    private static void Softmax(Span<float> x)
    {
        // Guard: if any input has Infinity/NaN, replace with zeros to prevent cascade
        bool hasBad = false;
        for (int i = 0; i < x.Length; i++)
        {
            if (float.IsInfinity(x[i]) || float.IsNaN(x[i])) { hasBad = true; break; }
        }
        if (hasBad)
        {
            x.Clear();
            if (x.Length > 0) x[0] = 1f;
            return;
        }
        
        float max = x[0];
        foreach (float v in x) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < x.Length; i++) 
        {
            float expVal = x[i] - max;
            // Guard against exp overflow
            if (expVal > 80f) expVal = 80f;
            else if (expVal < -80f) expVal = -80f;
            x[i] = MathF.Exp(expVal); 
            sum += x[i]; 
        }
        float inv = 1f / sum;
        for (int i = 0; i < x.Length; i++) x[i] *= inv;
    }

    // Top-k

    /// <summary>
    /// Zeroes all but the top-k highest-probability tokens.
    /// Uses an O(n log k) min-heap select instead of O(n log n) full sort.
    /// No per-call GC allocations — uses ArrayPool.
    /// </summary>
    private static void ApplyTopK(Span<float> probs, int k)
    {
        if (k >= probs.Length) return;

        int n = probs.Length;
        float[] rentedVals = ArrayPool<float>.Shared.Rent(n);
        int[] rentedIdxs = ArrayPool<int>.Shared.Rent(n);
        try
        {
            Span<float> vals = rentedVals.AsSpan(0, n);
            Span<int> idxs = rentedIdxs.AsSpan(0, n);
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                float v = probs[i];
                if (v > 0f) { vals[count] = v; idxs[count] = i; count++; }
            }

            if (count <= k) return;

            // Build initial min-heap from first k entries
            for (int i = k / 2 - 1; i >= 0; i--)
                SiftDown(vals, idxs, i, k);

            // Heap-select: swap larger candidates with heap root, then sift down.
            // Ejected indices naturally land in idxs[k..count-1].
            for (int i = k; i < count; i++)
            {
                if (vals[i] > vals[0])
                {
                    (vals[0], vals[i]) = (vals[i], vals[0]);
                    (idxs[0], idxs[i]) = (idxs[i], idxs[0]);
                    SiftDown(vals, idxs, 0, k);
                }
            }

            // Zero all ejected (non-top-k) probabilities in one linear pass
            for (int i = k; i < count; i++)
                probs[idxs[i]] = 0f;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedVals);
            ArrayPool<int>.Shared.Return(rentedIdxs);
        }
    }

    private static void SiftDown(Span<float> vals, Span<int> idxs, int pos, int heapSize)
    {
        while (true)
        {
            int smallest = pos;
            int left = 2 * pos + 1;
            int right = 2 * pos + 2;
            if (left < heapSize && vals[left] < vals[smallest]) smallest = left;
            if (right < heapSize && vals[right] < vals[smallest]) smallest = right;
            if (smallest == pos) break;
            (vals[pos], vals[smallest]) = (vals[smallest], vals[pos]);
            (idxs[pos], idxs[smallest]) = (idxs[smallest], idxs[pos]);
            pos = smallest;
        }
    }

    // Top-p (nucleus)

    /// <summary>
    /// Zeroes tokens outside the nucleus whose cumulative probability exceeds p.
    /// Tokens are sorted descending (via ascending sort then reverse traversal);
    /// we keep until the running sum exceeds p.
    /// No per-call GC allocations — uses ArrayPool.
    /// </summary>
    private static void ApplyTopP(Span<float> probs, float p)
    {
        if (p <= 0f || probs.Length == 0) return;

        int n = probs.Length;
        float[] rentedVals = ArrayPool<float>.Shared.Rent(n);
        int[] rentedIdxs = ArrayPool<int>.Shared.Rent(n);
        try
        {
            Span<float> vals = rentedVals.AsSpan(0, n);
            Span<int> idxs = rentedIdxs.AsSpan(0, n);
            int count = 0;

            for (int i = 0; i < n; i++)
            {
                float v = probs[i];
                if (v > 0f) { vals[count] = v; idxs[count] = i; count++; }
            }

            if (count == 0) return;

            // Sort ascending by value — O(n log n)
            vals[..count].Sort(idxs[..count]);

            // Traverse descending (largest probability first)
            float cumulative = 0f;
            for (int i = count - 1; i >= 0; i--)
            {
                cumulative += vals[i];
                if (cumulative >= p)
                {
                    // Zero everything below this threshold
                    for (int j = i - 1; j >= 0; j--)
                        probs[idxs[j]] = 0f;
                    break;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedVals);
            ArrayPool<int>.Shared.Return(rentedIdxs);
        }
    }

    // Min-p

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

    // Normalise

    private static void Normalise(Span<float> probs)
    {
        float sum = 0f;
        foreach (float v in probs) sum += v;
        if (sum <= 0f) return;
        float inv = 1f / sum;
        for (int i = 0; i < probs.Length; i++) probs[i] *= inv;
    }

    // Categorical sample

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

}
