namespace SharpMind.Data.Batching;

/// <summary>
/// B5: one tokenised training example plus a per-token assistant mask. The mask
/// marks the positions whose token is assistant output — the only positions an
/// instruction-tuning loss should be computed on. It is the piece a plain
/// <c>Func&lt;string, int[]&gt;</c> tokeniser has nowhere to return, which is why
/// record-aware batching needs its own tokeniser shape.
/// </summary>
/// <param name="TokenIds">Token IDs of the rendered example, in order.</param>
/// <param name="AssistantMask">
/// Per-token assistant flag, same length as <paramref name="TokenIds"/>. Null or
/// shorter means "no masking" — every token is treated as assistant output, which
/// is the correct default for plain language-model continuation.
/// </param>
public sealed record TokenizedExample(IReadOnlyList<int> TokenIds, IReadOnlyList<bool>? AssistantMask = null)
{
    /// <summary>True when position <paramref name="index"/> is assistant output (loss-bearing).</summary>
    public bool IsAssistant(int index)
        => AssistantMask is null || (index >= 0 && index < AssistantMask.Count && AssistantMask[index]);
}
