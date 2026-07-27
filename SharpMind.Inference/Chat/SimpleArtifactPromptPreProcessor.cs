using System.Text;

namespace SharpMind.Inference.Chat;

public sealed class SimpleArtifactPromptPreProcessor : IPromptPreProcessor
{
    public string Name => "Simple Artifact Injection";
    public string Description => "Inlines text artifacts into the user prompt; adds path hints for binary files";

    public Task ProcessAsync(ChatMessage userInput, IReadOnlyList<ChatMessage> history, CancellationToken ct)
    {
        if (userInput.Artifacts is not { Length: > 0 })
            return Task.CompletedTask;

        var sb = new StringBuilder();
        sb.Append(userInput.Content);
        foreach (var art in userInput.Artifacts)
        {
            if (art.Type is "text" or "code" or "json")
            {
                string text = Encoding.UTF8.GetString(art.Content);
                sb.Append($"\n\n--- {art.FileName} ---\n{text}");
            }
            else
            {
                string path = art.SourcePath ?? art.FileName ?? "unknown";
                sb.Append($"\n\n[File: {art.FileName} at {path} — use a file read tool to access it]");
            }
        }
        userInput.Content = sb.ToString();
        return Task.CompletedTask;
    }
}
