using SharpMind.CUI.App;

namespace SharpMind.CUI;

/// <summary>
/// One open chat session as tracked by MainWindow. DisplayName is mutable
/// and is what's renameable — separate from ChatView's own AgentName, which
/// stays whatever the launched agent is actually called even if the
/// session's tab-name changes, since those are conceptually different
/// things (who's talking vs. what this conversation is called).
/// </summary>
public sealed class ChatSessionState
{
    public required Guid Id { get; init; }
    public required string DisplayName { get; set; }
    public required SessionOptions Options { get; init; }
    public required IChatBridge Bridge { get; init; }
    public required ChatView View { get; init; }
}
