using Microsoft.Extensions.AI;
using SharpMind.Inference.Chat;

namespace SharpMind.Extensions.AI;

/// <summary>
/// Maps between Microsoft.Extensions.AI types and SharpMind's chat types.
/// </summary>
internal static class ChatMessageConverter
{
    /// <summary>
    /// Converts a MEAI <see cref="MeChatMessage"/> to a SharpMind
    /// <see cref="SmChatMessage"/>.
    /// </summary>
    public static SmChatMessage ToSharpMind(MeChatMessage source)
    {
        var role = source.Role.Value switch
        {
            "system" => SmChatRole.System,
            "assistant" => SmChatRole.Agent,
            "user" => SmChatRole.User,
            _ => SmChatRole.User
        };

        string content;
        if (!string.IsNullOrEmpty(source.Text))
        {
            content = source.Text;
        }
        else if (source.Contents is { Count: > 0 })
        {
            content = string.Concat(source.Contents.OfType<Microsoft.Extensions.AI.TextContent>().Select(c => c.Text));
        }
        else
        {
            content = string.Empty;
        }

        return new SmChatMessage { Role = role, Content = content, Name = source.AuthorName };
    }

    /// <summary>
    /// Converts a SharpMind <see cref="SmChatMessage"/> to a MEAI
    /// <see cref="MeChatMessage"/>.
    /// </summary>
    public static MeChatMessage ToMEAI(SmChatMessage source) => source.Role switch
    {
        SmChatRole.System => new MeChatMessage(MeChatRole.System, source.Content),
        SmChatRole.Agent => new MeChatMessage(MeChatRole.Assistant, source.Content),
        SmChatRole.User => new MeChatMessage(MeChatRole.User, source.Content),
        _ => new MeChatMessage(MeChatRole.User, source.Content)
    };

    /// <summary>
    /// Maps MEAI <see cref="ChatOptions"/> onto a SharpMind
    /// <see cref="IChatSession"/>. Only non-null properties are applied.
    /// </summary>
    public static void ApplyOptions(ChatOptions? options, IChatSession session)
    {
        if (options is null) return;

        if (options.Temperature is { } temp)
            session.Temperature = temp;

        if (options.TopP is { } topP)
            session.TopP = topP;

        if (options.TopK is { } topK)
            session.TopK = (int)topK;

        if (options.MaxOutputTokens is { } maxTokens)
            session.MaxNewTokens = maxTokens;
    }
}
