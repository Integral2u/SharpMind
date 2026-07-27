namespace SharpMind.Inference.Chat;

public interface IPromptPreProcessor
{
    string Name { get; }
    string Description { get; }
    Task ProcessAsync(ChatMessage userInput, IReadOnlyList<ChatMessage> history, CancellationToken ct);
}
