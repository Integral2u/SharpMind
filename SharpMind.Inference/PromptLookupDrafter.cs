namespace SharpMind.Inference;

/// <summary>Prompt-lookup counts for one generation: forward passes run, tokens drafted, drafts accepted.</summary>
public readonly record struct PromptLookupStats(int Forwards, int Drafted, int Accepted);

/// <summary>
/// Prompt-lookup drafting: guesses the next tokens by finding the latest earlier place in the
/// context where the last few tokens already occurred, and proposing what followed there. Code edits,
/// quoting and rewriting repeat the context verbatim, so the guesses often hold; the generator verifies
/// every guess against the model, so a wrong one costs time, never output.
/// </summary>
public sealed class PromptLookupDrafter
{
    /// <summary>Key lengths tried in order: a longer match is a better predictor than a more recent short one.</summary>
    public const int MaxKey = 4, MinKey = 2;

    /// <summary>Drafting stops for the request once this many tokens were drafted and under 20% were accepted.</summary>
    public const int GateMinDrafted = 24;

    public int Drafted { get; private set; }
    public int Accepted { get; private set; }

    /// <summary>False once the acceptance rate shows drafting does not pay for this request (prose).</summary>
    public bool Enabled => Drafted < GateMinDrafted || Accepted * 5 >= Drafted;

    /// <summary>
    /// Writes up to <c>draft.Length</c> proposed tokens to follow <paramref name="context"/> and returns
    /// how many. Zero when disabled, when <paramref name="draft"/> is empty, or when nothing matches.
    /// </summary>
    public int Propose(IReadOnlyList<int> context, Span<int> draft)
    {
        if (!Enabled || draft.IsEmpty) return 0;
        for (int key = MaxKey; key >= MinKey; key--)
        {
            int m = Search(context, key, draft);
            if (m > 0) return m;
        }
        return 0;
    }

    /// <summary>Records how a proposed run resolved, for the gate.</summary>
    public void Record(int drafted, int accepted)
    {
        Drafted += drafted;
        Accepted += accepted;
    }

    // Latest earlier occurrence of the last `key` tokens; copies what followed it. When the match
    // overlaps the tail (a repeating pattern: "aaaa", "= = =", indentation), what follows runs into
    // the tokens being drafted, so the copy continues from the draft itself: the pattern repeats.
    private static int Search(IReadOnlyList<int> context, int key, Span<int> draft)
    {
        int end = context.Count;
        if (end <= key) return 0;
        for (int s = end - key - 1; s >= 0; s--)
        {
            int j = 0;
            while (j < key && context[s + j] == context[end - key + j]) j++;
            if (j < key) continue;
            for (int i = 0; i < draft.Length; i++)
            {
                int from = s + key + i;
                draft[i] = from < end ? context[from] : draft[from - end];
            }
            return draft.Length;
        }
        return 0;
    }
}
