using System.Text.Json.Serialization;

namespace SharpMind.Inference.Chat;

public sealed class ChatSessionSnapshot
{
    public required List<ChatMessage> History { get; init; }
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? PendingDraft { get; init; }
}
