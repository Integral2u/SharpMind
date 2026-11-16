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

        // The active turn starts at the latest user message. Everything after it
        // (the current question, its tool calls and their results) must stay in
        // the prompt; only older turns are summarized.
        int lastUserIndex = context.History.FindLastIndex(
            message => message.Role == ChatRole.User);
        int summarizeBefore = lastUserIndex >= 0 ? lastUserIndex : context.History.Count;

        // Collect older, non-pinned, non-ignored messages. Every System message
        // is instructions, not conversation, so all of them stay.
        var toSummarize = new List<ChatMessage>();
        int lastSystemIndex = -1;
        for (int i = 0; i < context.History.Count; i++)
        {
            ChatMessage message = context.History[i];
            if (message.Role == ChatRole.System)
            {
                lastSystemIndex = i;
                continue;
            }

            if (i >= summarizeBefore) continue;
            if (message.Ignore) continue;
            if (message.IsPinned) continue;
            toSummarize.Add(message);
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
                ChatRole.Tool => "tool",
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

        // Insert the summary after the last system message, but never after the
        // current turn's first message.
        int insertAt = lastSystemIndex >= 0
            ? Math.Min(lastSystemIndex + 1, summarizeBefore)
            : 0;
        context.History.Insert(insertAt, new ChatMessage
        {
            Role = ChatRole.System,
            Content = $"Summary of earlier conversation: {summary}",
            IsPinned = true
        });

        return true;
    }
}
