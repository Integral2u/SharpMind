namespace SharpMind.Inference.Chat;

public interface IPromptPreProcessor
{
    Task<string> ProcessAsync(string userInput, IReadOnlyList<ChatMessage> history, CancellationToken ct);
}
