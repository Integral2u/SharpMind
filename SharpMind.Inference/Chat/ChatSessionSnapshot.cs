using System.Text.Json.Serialization;

namespace SharpMind.Inference.Chat;

public sealed class ChatSessionSnapshot
{
    public required List<ChatMessage> History { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? PendingDraft { get; init; }

    /// <summary>
    /// Persisted KV-cache state plus the prompt-token hash that was
    /// prefilled. When present and the hash matches the freshly-built
    /// prompt on resume, <see cref="ChatSession.WarmupPrefillAsync"/>
    /// restores the cache from this blob and skips the expensive prefill.
    /// </summary>
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public KVCacheSnapshot? KVCache { get; init; }
}
