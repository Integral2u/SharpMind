namespace SharpMind.Inference.Chat;

public interface IPromptPreProcessor
{
    string Name { get; }
    Task<string> ProcessAsync(string userInput, IReadOnlyList<ChatMessage> history, CancellationToken ct);
}
