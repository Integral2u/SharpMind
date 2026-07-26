using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Compacts conversation history by marking the oldest non-pinned messages as
/// <see cref="ChatMessage.Ignore"/> until the estimated token count fits within
/// the context budget.
/// </summary>
public sealed class TruncatingCompactor : IContextCompactor
{
    public string Name { get; init; } = "Truncate";
    /// <summary>Fraction of MaxTokens at which compaction is triggered (default 0.7).</summary>
    public float Threshold { get; init; } = 0.7f;
    /// <summary>Target fraction of MaxTokens to compact down to (default 0.5).</summary>
    public float Target { get; init; } = 0.5f;

    public Task<bool> ShouldCompactAsync(CompactionContext context, CancellationToken ct)
    {
        bool needed = context.CurrentTokenCount > context.MaxTokens * Threshold;
        return Task.FromResult(needed);
    }

    public Task<bool> CompactAsync(CompactionContext context, CancellationToken ct)
    {
        if (context.Tokenizer is null) return Task.FromResult(false);

        int targetTokens = (int)(context.MaxTokens * Target);
        int currentTokens = context.CurrentTokenCount;

        // Count tokens for each non-pinned, non-ignored message
        var candidates = new List<(int Index, int EstimatedTokens)>();
        for (int i = 0; i < context.History.Count; i++)
        {
            var msg = context.History[i];
            if (msg.Ignore || msg.IsPinned) continue;
            int est = EstimateTokens(context.Tokenizer, msg);
            candidates.Add((i, est));
        }

        if (candidates.Count == 0) return Task.FromResult(false);

        // Mark from oldest (lowest index) until we're under target
        int removed = 0;
        foreach (var (idx, est) in candidates)
        {
            if (currentTokens - removed <= targetTokens) break;
            removed += est;
            context.History[idx].Ignore = true;
        }

        bool didWork = removed > 0;
        return Task.FromResult(didWork);
    }

    private static int EstimateTokens(Tokenizer tokenizer, ChatMessage msg)
    {
        // Rough estimate: average ~2 tokens per word in English text
        // but use the actual encoder when available for better accuracy
        try
        {
            int role = msg.Role == ChatRole.System ? 0
                : msg.Role == ChatRole.User ? 1 : 2;
            string prefix = role switch
            {
                0 => "<|im_start|>system\n",
                1 => "<|im_start|>user\n",
                _ => "<|im_start|>assistant\n"
            };
            string suffix = "\n<|im_end|>";
            return tokenizer.Encode(prefix + msg.Content + suffix, false, false).Length;
        }
        catch
        {
            // Fallback: character-based estimate (~4 chars per token)
            return msg.Content.Length / 4 + 8;
        }
    }
}
