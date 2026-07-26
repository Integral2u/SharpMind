namespace SharpMind.Inference.Chat;

/// <summary>
/// Post-processes each completed agent response before the next turn starts
/// or the session ends. Receives the full agent <see cref="ChatMessage"/>
/// (with its response text and any artifacts already attached) and the
/// current history, and can modify the message in-place — add or remove
/// artifacts, rewrite content, etc. Runs once per turn, after
/// <see cref="ChatSession.GetResponseStreamAsync"/> has fully yielded and
/// the agent message has been appended to history.
/// </summary>
public interface IPromptPostProcessor
{
    string Name { get; }
    Task ProcessAsync(ChatMessage agentMessage, IReadOnlyList<ChatMessage> history, CancellationToken ct);
}
