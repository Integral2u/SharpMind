namespace SharpMind.Inference.Chat;

/// <summary>
/// Compacts conversation history by summarizing older non-pinned messages using the session's model.
/// The summary replaces the original messages as a System message, and the originals are marked
/// <see cref="ChatMessage.Ignore"/> so the prompt formatter skips them.
/// </summary>
public sealed class SummarizingCompactor : IContextCompactor
{
    public string Name { get; init; } = "Summarize";
    /// <summary>Fraction of MaxTokens at which compaction is triggered (default 0.7).</summary>
    public float Threshold { get; init; } = 0.7f;

    public Task<bool> ShouldCompactAsync(CompactionContext context, CancellationToken ct)
    {
        bool needed = context.CurrentTokenCount > context.MaxTokens * Threshold;
        return Task.FromResult(needed);
    }

    public async Task<bool> CompactAsync(CompactionContext context, CancellationToken ct)
    {
        if (context.SummarizeAsync is null) return false;

        // Collect non-pinned, non-ignored messages eligible for summarization.
        // Preserve the first System message (agent instructions) and pinned messages.
        var toSummarize = new List<ChatMessage>();
        int firstSystemIdx = -1;
        for (int i = 0; i < context.History.Count; i++)
        {
            var msg = context.History[i];
            if (msg.Ignore) continue;
            if (msg.IsPinned) continue;
            if (msg.Role == ChatRole.System && firstSystemIdx == -1)
            {
                firstSystemIdx = i;
                continue;
            }
            toSummarize.Add(msg);
        }

        if (toSummarize.Count == 0) return false;

        // Build a summarization prompt from the eligible messages
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Summarize the following conversation concisely. Preserve all facts, instructions, deadlines, and context needed to continue naturally. Omit pleasantries and redundant meta-commentary.");
        sb.AppendLine();
        foreach (var msg in toSummarize)
        {
            string role = msg.Role switch
            {
                ChatRole.System => "system",
                ChatRole.User => "user",
                ChatRole.Agent => "assistant",
                _ => "unknown"
            };
            sb.AppendLine($"<|{role}|>");
            sb.AppendLine(msg.Content);
            sb.AppendLine($"<|/{role}|>");
        }
        sb.AppendLine();
        sb.AppendLine("Summary:");

        string summary = await context.SummarizeAsync(sb.ToString());

        // Mark original messages as ignored
        foreach (var msg in toSummarize)
            msg.Ignore = true;

        // Insert the summary as a System message right after the first System message (or at the start)
        int insertAt = firstSystemIdx >= 0 ? firstSystemIdx + 1 : 0;
        context.History.Insert(insertAt, new ChatMessage
        {
            Role = ChatRole.System,
            Content = $"Summary of earlier conversation: {summary}",
            IsPinned = true
        });

        return true;
    }
}
