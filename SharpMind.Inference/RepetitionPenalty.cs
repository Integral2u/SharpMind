namespace SharpMind.Inference;

/// <summary>
/// Correct repetition penalty semantics shared by the streaming generators.
///
/// Matches HF/llama.cpp: scale each DISTINCT token id once per window/context, NOT
/// once per occurrence. Scaling per occurrence raises the effective penalty to
/// penalty^count for high-frequency words (signals, "the", "is", ...), wiping them
/// from the distribution as context grows and causing the late-sequence degeneration
/// seen across small models. Repeated idiom tokens that each occur only once were the
/// converse bug (under-penalized), so verbatim phrase loops were never suppressed.
/// </summary>
internal static class RepetitionPenalty
{
    public static void Apply(
        Span<float> logits,
        ReadOnlySpan<int> ids,
        float penalty,
        HashSet<int> seen)
    {
        foreach (int id in ids)
        {
            if (!seen.Add(id)) continue;
            if ((uint)id >= (uint)logits.Length) continue;
            // Positive logits -> divide (suppress); negative -> multiply (suppress via magnitude).
            logits[id] = logits[id] >= 0f ? logits[id] / penalty : logits[id] * penalty;
        }
    }
}