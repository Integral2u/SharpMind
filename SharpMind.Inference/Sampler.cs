using System.Runtime.CompilerServices;
using SharpMind.Core.Memory;

namespace SharpMind.Inference;

/// <summary>
/// Stateless sampling functions. Each method takes raw logits and returns a token ID.
/// </summary>
public static class Sampler
{
    // A per-thread cache of seeded RNGs. Sampler.Sample accepts an optional
    // Random so callers can drive one sequence across a whole generation, but
    // when none is supplied a seeded config must still advance a single stream
    // rather than re-creating `new Random(seed)` on every draw — which would
    // pick the same quantile each step.
    [System.ThreadStatic]
    private static Dictionary<int, Random>? _seededRngs;

    private static Random SeededRng(int seed)
    {
        _seededRngs ??= [];
        if (!_seededRngs.TryGetValue(seed, out var rng))
            _seededRngs[seed] = rng = new Random(seed);
        return rng;
    }

    // Entry point

    /// <summary>
    /// Selects the next token from <paramref name="logits"/> using the given config.
    /// Applies temperature → top-k on the logits → softmax over the survivors →
    /// min-p → top-p → sample (or argmax if greedy). Top-k and top-p therefore
    /// act on the renormalised candidate distribution, so top-p keeps the
    /// smallest set of candidates whose probabilities sum to at least p.
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingConfig config, Random? rng = null)
    {
        // Greedy shortcut — no allocation needed
        if (config.Temperature <= 0f)
            return Argmax(logits);

        // Candidate set: the top-k logits, or the whole vocabulary when unfiltered.
        int n = logits.Length;
        int k = config.TopK > 0 && config.TopK < n ? config.TopK : n;

        float[] valsBuf = MemoryHelpers.RentArray<float>(k);
        int[] idxBuf = MemoryHelpers.RentArray<int>(k);
        try
        {
            Span<float> probs = valsBuf.AsSpan(0, k);
            Span<int> idxs = idxBuf.AsSpan(0, k);

            // A non-finite logit collapses the distribution onto token 0.
            if (!SelectCandidates(logits, config.Temperature, probs, idxs))
                return 0;

            SoftmaxInPlace(probs);

            if (config.MinP > 0f) ApplyMinP(probs, config.MinP);
            if (config.TopP < 1.0f) ApplyTopP(probs, default, config.TopP);

            Normalise(probs);

            int pick = SampleFromProbs(probs, rng ?? (config.Seed.HasValue
                ? SeededRng(config.Seed.Value)
                : Random.Shared));
            return idxs[pick];
        }
        finally
        {
            MemoryHelpers.ReturnArray(valsBuf);
            MemoryHelpers.ReturnArray(idxBuf);
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

    // Candidate selection

    /// <summary>
    /// Writes the temperature-scaled logits of the candidates into
    /// <paramref name="vals"/> and their token ids into <paramref name="idxs"/>.
    /// When <paramref name="vals"/> is shorter than the vocabulary, it keeps the
    /// largest entries (min-heap select, O(n log k)). Returns false if any scaled
    /// logit is non-finite.
    /// </summary>
    private static bool SelectCandidates(ReadOnlySpan<float> logits, float temperature, Span<float> vals, Span<int> idxs)
    {
        float inv = 1f / temperature;
        int n = logits.Length;
        int k = vals.Length;

        if (k == n)
        {
            for (int i = 0; i < n; i++)
            {
                float v = logits[i] * inv;
                if (!float.IsFinite(v)) return false;
                vals[i] = v;
                idxs[i] = i;
            }
            return true;
        }

        for (int i = 0; i < k; i++)
        {
            float v = logits[i] * inv;
            if (!float.IsFinite(v)) return false;
            vals[i] = v;
            idxs[i] = i;
        }
        for (int i = k / 2 - 1; i >= 0; i--)
            SiftDown(vals, idxs, i, k);

        for (int i = k; i < n; i++)
        {
            float v = logits[i] * inv;
            if (!float.IsFinite(v)) return false;
            if (v > vals[0])
            {
                vals[0] = v;
                idxs[0] = i;
                SiftDown(vals, idxs, 0, k);
            }
        }
        return true;
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

    // Softmax

    /// <summary>
    /// Replaces scaled logits with their softmax, in place. The exp argument is
    /// clamped to [-80, 80] to avoid overflow, as before.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SoftmaxInPlace(Span<float> x)
    {
        float max = x[0];
        foreach (float v in x) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < x.Length; i++)
        {
            float expVal = x[i] - max;
            if (expVal > 80f) expVal = 80f;
            else if (expVal < -80f) expVal = -80f;
            x[i] = MathF.Exp(expVal);
            sum += x[i];
        }
        float inv = 1f / sum;
        for (int i = 0; i < x.Length; i++) x[i] *= inv;
    }

    // Top-p (nucleus)

    /// <summary>
    /// Zeroes tokens outside the nucleus: sorts the non-zero entries ascending,
    /// then keeps the largest until their running sum reaches p. Operates on
    /// whatever distribution it is given, so callers pass the renormalised
    /// candidates. No per-call GC allocations — uses ArrayPool.
    /// </summary>
    private static void ApplyTopP(Span<float> probs, ReadOnlySpan<int> survivors, float p)
    {
        if (p <= 0f || probs.Length == 0) return;

        int cap = survivors.IsEmpty ? probs.Length : survivors.Length;
        float[] rentedVals = MemoryHelpers.RentArray<float>(cap);
        int[] rentedIdxs = MemoryHelpers.RentArray<int>(cap);
        try
        {
            Span<float> vals = rentedVals.AsSpan(0, cap);
            Span<int> idxs = rentedIdxs.AsSpan(0, cap);
            int count;

            if (!survivors.IsEmpty)
            {
                count = survivors.Length;
                for (int i = 0; i < count; i++)
                {
                    vals[i] = probs[survivors[i]];
                    idxs[i] = survivors[i];
                }
            }
            else
            {
                count = 0;
                for (int i = 0; i < probs.Length; i++)
                {
                    float v = probs[i];
                    if (v > 0f) { vals[count] = v; idxs[count] = i; count++; }
                }
            }

            if (count == 0) return;

            // Sort ascending by value, then traverse descending (largest first)
            vals[..count].Sort(idxs[..count]);

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
            MemoryHelpers.ReturnArray(rentedVals);
            MemoryHelpers.ReturnArray(rentedIdxs);
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

    /// <summary>Returns a position in <paramref name="probs"/>, not a token id.</summary>
    private static int SampleFromProbs(Span<float> probs, Random rng)
    {
        float r = rng.NextSingle();
        float cumSum = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (r <= cumSum) return i;
        }
        // Floating point edge case — return last non-zero candidate
        for (int i = probs.Length - 1; i >= 0; i--)
            if (probs[i] > 0f) return i;
        return 0;
    }
}
