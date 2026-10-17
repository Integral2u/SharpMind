namespace SharpMind.Data.Records;

/// <summary>
/// One turn of a conversation: who spoke and what they said. Deliberately a plain
/// string role (not the inference layer's enum), so the data layer stays free of
/// an Inference reference — the role is interpreted later by a chat template.
/// </summary>
/// <param name="Role">Speaker role, e.g. <c>system</c>, <c>user</c>, <c>assistant</c>.</param>
/// <param name="Content">The message text.</param>
public sealed record ConversationTurn(string Role, string Content);

/// <summary>
/// B4: a typed training example, the record-shaped counterpart to a bare string
/// document. Every record has <see cref="Text"/>; the optional fields are what a
/// string cannot express — a multi-turn conversation, provenance, and tags used
/// by filters and curriculum. Existing string sources and stages bridge to this
/// type through the adapters in <see cref="RecordSourceAdapters"/>, so nothing
/// has to be rewritten to adopt it.
/// </summary>
public sealed record DataRecord
{
    /// <summary>The record's plain-text form. Always present; for a conversation this is the rendered transcript.</summary>
    public required string Text { get; init; }

    /// <summary>
    /// Structured conversation turns, when the record is a dialogue (used for
    /// role-aware masking). Null for an ordinary text document.
    /// </summary>
    public IReadOnlyList<ConversationTurn>? Turns { get; init; }

    /// <summary>Free-form labels for filtering, mixing or curriculum.</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Where the record came from (file, dataset, URI), for lineage and diagnostics.</summary>
    public string? Source { get; init; }
}
