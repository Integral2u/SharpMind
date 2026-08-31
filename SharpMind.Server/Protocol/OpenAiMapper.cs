using System.Text.Json;
using SharpMind.Inference.Chat;
using SharpMind.Server.Protocol;

namespace SharpMind.Server.Protocol;

/// <summary>
/// Bidirectional mapping between OpenAI protocol types and SharpMind types.
/// </summary>
public static class OpenAiMapper
{
    /// <summary>
    /// Convert OpenAI messages to a system prompt + conversation history.
    /// System messages are concatenated into the system prompt.
    /// User/Assistant messages become the conversation history.
    /// </summary>
    public static (string systemPrompt, List<ChatMessage> history) ToChatHistory(List<ChatCompletionRequestMessage> messages)
    {
        var systemParts = new List<string>();
        var history = new List<ChatMessage>();

        foreach (var msg in messages)
        {
            var content = ExtractTextContent(msg.Content);
            if (string.IsNullOrEmpty(content)) continue;

            switch (msg.Role)
            {
                case "system":
                    systemParts.Add(content);
                    break;
                case "user":
                    history.Add(new ChatMessage { Role = ChatRole.User, Content = content });
                    break;
                case "assistant":
                    history.Add(new ChatMessage { Role = ChatRole.Agent, Content = content });
                    break;
                // tool messages ignored — SharpMind tools are CUI-side
            }
        }

        return (string.Join("\n\n", systemParts), history);
    }

    /// <summary>
    /// True when a stream entry carries model output that belongs in the completion.
    /// </summary>
    /// <remarks>
    /// Progress entries reuse <see cref="ChatStreamEntry.Token"/> for status text
    /// (<c>"Prefilling 50.25%"</c>) under <see cref="ChatStatus.Updating"/>, so a
    /// consumer that concatenates every non-empty Token emits that to the client as
    /// if the model had said it. The streaming and non-streaming paths each wrote
    /// this rule out separately and drifted — streaming filtered, non-streaming did
    /// not — so it lives in one place now.
    /// </remarks>
    public static bool IsContent(ChatStreamEntry entry)
        => entry is { Status: ChatStatus.Responding or ChatStatus.Thinking, Token: { Length: > 0 } };

    /// <summary>
    /// Map OpenAI request params to session properties.
    /// </summary>
    public static void ApplyToSession(CreateChatCompletionRequest request, IChatSession session)
    {
        if (request.Temperature is { } temp)
            session.Temperature = Math.Clamp(temp, 0f, 2f);

        if (request.TopP is { } topP)
            session.TopP = Math.Clamp(topP, 0f, 1f);

        if (request.MaxCompletionTokens is { } maxCompletion)
            session.MaxNewTokens = Math.Max(1, maxCompletion);
        else if (request.MaxTokens is { } maxTokens)
            session.MaxNewTokens = Math.Max(1, maxTokens);

        if (request.FrequencyPenalty is { } freqPenalty)
            session.RepetitionPenalty = 1.0f + Math.Clamp(freqPenalty, -2f, 2f);

        if (request.Stop?.Values is { Count: > 0 } stopValues)
            session.StopStrings = [.. stopValues];
    }

    /// <summary>
    /// Build a non-streaming chat completion response.
    /// </summary>
    public static CreateChatCompletionResponse ToResponse(
        string completionId,
        long created,
        string modelId,
        string content,
        CompletionUsage? usage)
    {
        return new CreateChatCompletionResponse
        {
            Id = completionId,
            Created = created,
            Model = modelId,
            Choices =
            [
                new ChatCompletionChoice
                {
                    Index = 0,
                    Message = new ChatCompletionResponseMessage { Content = content },
                    FinishReason = "stop"
                }
            ],
            Usage = usage
        };
    }

    /// <summary>
    /// Build a streaming chunk.
    /// </summary>
    public static CreateChatCompletionStreamResponse ToStreamChunk(
        string completionId,
        long created,
        string modelId,
        string? content,
        string? finishReason,
        CompletionUsage? usage)
    {
        return new CreateChatCompletionStreamResponse
        {
            Id = completionId,
            Created = created,
            Model = modelId,
            Choices =
            [
                new ChatCompletionStreamChoice
                {
                    Index = 0,
                    Delta = new ChatCompletionStreamDelta
                    {
                        Content = content
                    },
                    FinishReason = finishReason
                }
            ],
            Usage = usage
        };
    }

    /// <summary>
    /// Build the first streaming chunk (includes role).
    /// </summary>
    public static CreateChatCompletionStreamResponse ToStreamRoleChunk(
        string completionId,
        long created,
        string modelId)
    {
        return new CreateChatCompletionStreamResponse
        {
            Id = completionId,
            Created = created,
            Model = modelId,
            Choices =
            [
                new ChatCompletionStreamChoice
                {
                    Index = 0,
                    Delta = new ChatCompletionStreamDelta { Role = "assistant" },
                    FinishReason = null
                }
            ]
        };
    }

    /// <summary>
    /// Build a model info object for /v1/models.
    /// </summary>
    public static ModelObject ToModelInfo(string modelId, long createdUnix)
    {
        return new ModelObject
        {
            Id = modelId,
            Created = createdUnix
        };
    }

    public static string ExtractTextContent(object? content)
    {
        if (content is null) return "";
        if (content is not string s) return content.ToString() ?? "";
        if (s.Length == 0 || s[0] != '[') return s;

        // Multi-part content array: [{"type":"text","text":"hello"}, ...]
        try
        {
            using var doc = JsonDocument.Parse(s);
            if (doc.RootElement.ValueKind != JsonValueKind.Array) return s;

            var parts = new List<string>();
            foreach (var part in doc.RootElement.EnumerateArray())
            {
                if (part.TryGetProperty("type", out var t) && t.GetString() == "text"
                    && part.TryGetProperty("text", out var txt))
                {
                    parts.Add(txt.GetString() ?? "");
                }
            }
            return parts.Count > 0 ? string.Concat(parts) : s;
        }
        catch
        {
            return s;
        }
    }
}
