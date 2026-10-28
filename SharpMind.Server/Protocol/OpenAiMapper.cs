using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Grammar;
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

            switch (msg.Role)
            {
                case "system":
                    if (!string.IsNullOrEmpty(content)) systemParts.Add(content);
                    break;
                case "user":
                    if (!string.IsNullOrEmpty(content))
                        history.Add(new ChatMessage { Role = ChatRole.User, Content = content });
                    break;
                case "assistant":
                    if (msg is AssistantMessage { ToolCalls: { Count: > 0 } calls })
                    {
                        string callText = string.Join("\n", calls.Select(c =>
                            new JsonObject
                            {
                                ["name"] = c.Function.Name,
                                ["arguments"] = ParseArguments(c.Function.Arguments)
                            }.ToJsonString()));
                        string combined = string.IsNullOrEmpty(content) ? callText : content + "\n" + callText;
                        history.Add(new ChatMessage { Role = ChatRole.Agent, Content = combined });
                    }
                    else if (!string.IsNullOrEmpty(content))
                    {
                        history.Add(new ChatMessage { Role = ChatRole.Agent, Content = content });
                    }
                    break;
                case "tool":
                    history.Add(new ChatMessage { Role = ChatRole.Tool, Content = content });
                    break;
            }
        }

        return (string.Join("\n\n", systemParts), history);
    }

    private static JsonNode ParseArguments(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return new JsonObject();
        try
        {
            return JsonNode.Parse(json) ?? new JsonObject();
        }
        catch (JsonException)
        {
            return new JsonObject();
        }
    }

    /// <summary>
    /// Builds the prompt-side tool registry for an OpenAI request. The server
    /// never executes client tools: <c>ProcessToolRequest</c> returns each call
    /// to the HTTP caller, and the execute delegate is a placeholder.
    /// </summary>
    public static AgentBuilder? CreateToolAgent(List<object>? tools)
    {
        if (tools is not { Count: > 0 }) return null;

        var builder = new AgentBuilder("Server");
        foreach (var item in tools)
        {
            JsonElement element = ToJsonElement(item);
            if (element.ValueKind != JsonValueKind.Object) continue;
            if (!element.TryGetProperty("function", out var function)
                || function.ValueKind != JsonValueKind.Object)
                continue;
            if (!function.TryGetProperty("name", out var nameProp)) continue;
            string? name = nameProp.GetString();
            if (string.IsNullOrWhiteSpace(name)) continue;

            string description = function.TryGetProperty("description", out var descProp)
                ? descProp.GetString() ?? "" : "";
            JsonObject schema = function.TryGetProperty("parameters", out var parameters)
                ? JsonNode.Parse(parameters.GetRawText()) as JsonObject ?? new JsonObject()
                : new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() };

            builder.WithTool(name, description, schema, _ => Task.FromResult(""));
        }

        return builder.ToolDefinitions.Count > 0 ? builder : null;
    }

    /// <summary>
    /// Original OpenAI tool definitions as JSON, for native chat templates
    /// that expect <c>{type:"function", function:{...}}</c> entries.
    /// </summary>
    public static string? ToolsJson(List<object>? tools)
        => tools is { Count: > 0 } ? JsonSerializer.Serialize(tools) : null;

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
        => entry is { Status: ChatStatus.Responding or ChatStatus.Thinking, Token.Length: > 0 };

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

        session.Grammar = ResolveGrammar(request);
    }

    /// <summary>
    /// Resolve <paramref name="request"/>.ResponseFormat into a GBNF grammar
    /// that forces the completion to be well-formed JSON. Returns null when no
    /// response_format is present. Throws <see cref="JsonSchemaException"/> for
    /// a malformed or unsupported format so the caller can reply 400.
    /// </summary>
    /// <remarks>
    /// <c>{"type":"json_object"}</c> maps to
    /// <see cref="JsonSchemaGrammar.AnyJson"/>; <c>{"type":"json_schema"}</c>
    /// maps to <see cref="JsonSchemaGrammar.FromSchema(JsonElement)"/> using the
    /// schema under <c>json_schema.schema</c> (OpenAI structured outputs) or the
    /// <c>json_schema</c> object itself when it is inline.
    /// </remarks>
    public static string? ResolveGrammar(CreateChatCompletionRequest request)
    {
        if (request.ResponseFormat is null)
            return null;

        JsonElement format = ToJsonElement(request.ResponseFormat);
        if (format.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
            return null;
        if (format.ValueKind != JsonValueKind.Object)
            throw new JsonSchemaException("'response_format' must be an object");

        string? type = format.TryGetProperty("type", out var typeProp) ? typeProp.GetString() : null;
        switch (type)
        {
            case "json_object":
                return JsonSchemaGrammar.AnyJson;
            case "json_schema":
                if (!format.TryGetProperty("json_schema", out var jsonSchema) || jsonSchema.ValueKind != JsonValueKind.Object)
                    throw new JsonSchemaException("'response_format' of type 'json_schema' requires a 'json_schema' object");
                if (jsonSchema.TryGetProperty("schema", out var schema) && schema.ValueKind == JsonValueKind.Object)
                    return JsonSchemaGrammar.FromSchema(schema);
                return JsonSchemaGrammar.FromSchema(jsonSchema);
            case null:
                throw new JsonSchemaException("'response_format' requires a 'type' of 'json_object' or 'json_schema'");
            default:
                throw new JsonSchemaException($"unsupported 'response_format.type' '{type}'");
        }
    }

    private static JsonElement ToJsonElement(object value)
    {
        if (value is JsonElement element)
            return element;
        if (value is JsonDocument document)
            return document.RootElement.Clone();
        if (value is string text)
            return JsonSerializer.Deserialize<JsonElement>(text);
        return JsonSerializer.SerializeToElement(value);
    }

    /// <summary>
    /// Build a non-streaming chat completion response.
    /// </summary>
    public static CreateChatCompletionResponse ToResponse(
        string completionId,
        long created,
        string modelId,
        string content,
        CompletionUsage? usage,
        List<ChatCompletionToolCall>? toolCalls = null)
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
                    Message = new ChatCompletionResponseMessage
                    {
                        Content = toolCalls is { Count: > 0 } ? null : content,
                        ToolCalls = toolCalls is { Count: > 0 } ? toolCalls : null
                    },
                    FinishReason = toolCalls is { Count: > 0 } ? "tool_calls" : "stop"
                }
            ],
            Usage = usage
        };
    }

    /// <summary>Maps a session tool call to the OpenAI wire shape.</summary>
    public static ChatCompletionToolCall ToToolCall(JsonObject call, string id)
    {
        string name = call["tool"]?.GetValue<string>() ?? "";
        string arguments = call["arguments"]?.ToJsonString() ?? "{}";
        return new ChatCompletionToolCall
        {
            Id = id,
            Type = "function",
            Function = new ChatCompletionFunctionCall { Name = name, Arguments = arguments }
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
        CompletionUsage? usage,
        List<ChatCompletionToolCall>? toolCalls = null)
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
                        Content = content,
                        ToolCalls = toolCalls
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
