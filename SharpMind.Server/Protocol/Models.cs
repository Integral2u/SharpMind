using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpMind.Server.Protocol;

// ──────────────────────────────────────────────────────────────────────────
// OpenAI API v1 protocol types — exact spec field names only.
// Source: https://github.com/openai/openai-openapi (v2.3.0)
// ──────────────────────────────────────────────────────────────────────────

// ── GET /v1/models ───────────────────────────────────────────────────────

public sealed class ListModelsResponse
{
    [JsonPropertyName("object")]
    public string Object { get; set; } = "list";

    [JsonPropertyName("data")]
    public List<ModelObject> Data { get; set; } = [];
}

public sealed class ModelObject
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; set; } = "sharpmind";
}

public sealed class DeleteModelResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "model";

    [JsonPropertyName("deleted")]
    public bool Deleted { get; set; }
}

// ── POST /v1/chat/completions — Request ──────────────────────────────────

public sealed class CreateChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("messages")]
    public List<ChatCompletionRequestMessage> Messages { get; set; } = [];

    [JsonPropertyName("temperature")]
    public float? Temperature { get; set; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; set; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    [JsonPropertyName("max_completion_tokens")]
    public int? MaxCompletionTokens { get; set; }

    [JsonPropertyName("stop")]
    public OneOrMany<string>? Stop { get; set; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; set; }

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; set; }

    [JsonPropertyName("stream")]
    public bool? Stream { get; set; }

    [JsonPropertyName("stream_options")]
    public StreamOptions? StreamOptions { get; set; }

    [JsonPropertyName("n")]
    public int? N { get; set; }

    [JsonPropertyName("logprobs")]
    public bool? Logprobs { get; set; }

    [JsonPropertyName("tools")]
    public List<object>? Tools { get; set; }

    [JsonPropertyName("tool_choice")]
    public object? ToolChoice { get; set; }

    [JsonPropertyName("response_format")]
    public object? ResponseFormat { get; set; }

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }

    [JsonPropertyName("user")]
    public string? User { get; set; }
}

public sealed class StreamOptions
{
    [JsonPropertyName("include_usage")]
    public bool? IncludeUsage { get; set; }
}

// ── POST /v1/chat/completions — Response (non-streaming) ─────────────────

public sealed class CreateChatCompletionResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("choices")]
    public List<ChatCompletionChoice> Choices { get; set; } = [];

    [JsonPropertyName("usage")]
    public CompletionUsage? Usage { get; set; }
}

public sealed class ChatCompletionChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("message")]
    public ChatCompletionResponseMessage Message { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }

    [JsonPropertyName("logprobs")]
    public object? Logprobs { get; set; }
}

public sealed class ChatCompletionResponseMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "assistant";

    [JsonPropertyName("content")]
    public string? Content { get; set; }
}

// ── POST /v1/chat/completions — Streaming chunks ─────────────────────────

public sealed class CreateChatCompletionStreamResponse
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = "";

    [JsonPropertyName("object")]
    public string Object { get; set; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("model")]
    public string Model { get; set; } = "";

    [JsonPropertyName("choices")]
    public List<ChatCompletionStreamChoice> Choices { get; set; } = [];

    [JsonPropertyName("usage")]
    public CompletionUsage? Usage { get; set; }
}

public sealed class ChatCompletionStreamChoice
{
    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("delta")]
    public ChatCompletionStreamDelta Delta { get; set; } = new();

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; set; }

    [JsonPropertyName("logprobs")]
    public object? Logprobs { get; set; }
}

public sealed class ChatCompletionStreamDelta
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; set; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
}

// ── Shared types ─────────────────────────────────────────────────────────

public sealed class CompletionUsage
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; set; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; set; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; set; }
}

/// <summary>
/// OpenAI's "messages" field accepts either a single string or an array of strings.
/// Used for the "stop" parameter.
/// </summary>
[JsonConverter(typeof(OneOrManyJsonConverter))]
public sealed class OneOrMany<T>
{
    private readonly List<T> _values;

    public OneOrMany(T single) { _values = [single]; }
    public OneOrMany(List<T> multiple) { _values = multiple; }

    public IReadOnlyList<T> Values => _values;

    public static implicit operator OneOrMany<T>(T single) => new(single);
    public static implicit operator OneOrMany<T>(List<T> multiple) => new(multiple);
}

// ── Request message types (discriminated on "role") ──────────────────────

[JsonConverter(typeof(ChatCompletionRequestMessageConverter))]
public abstract class ChatCompletionRequestMessage
{
    [JsonPropertyName("role")]
    public abstract string Role { get; }

    [JsonPropertyName("content")]
    public virtual object? Content { get; set; }

    [JsonPropertyName("name")]
    public string? Name { get; set; }
}

public sealed class SystemMessage : ChatCompletionRequestMessage
{
    public override string Role => "system";
}

public sealed class UserMessage : ChatCompletionRequestMessage
{
    public override string Role => "user";
}

public sealed class AssistantMessage : ChatCompletionRequestMessage
{
    public override string Role => "assistant";
}

public sealed class ToolMessage : ChatCompletionRequestMessage
{
    public override string Role => "tool";

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; set; }
}

// ── JSON converters ──────────────────────────────────────────────────────

internal sealed class ChatCompletionRequestMessageConverter : JsonConverter<ChatCompletionRequestMessage>
{
    public override ChatCompletionRequestMessage? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions? options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        var role = root.GetProperty("role").GetString();

        ChatCompletionRequestMessage msg = role switch
        {
            "system" => new SystemMessage(),
            "user" => new UserMessage(),
            "assistant" => new AssistantMessage(),
            "tool" => new ToolMessage(),
            _ => new UserMessage()
        };

        if (root.TryGetProperty("content", out var content))
            msg.Content = content.ValueKind == JsonValueKind.String
                ? content.GetString()
                : content.GetRawText();

        if (root.TryGetProperty("name", out var name))
            msg.Name = name.GetString();

        if (msg is ToolMessage toolMsg && root.TryGetProperty("tool_call_id", out var tcid))
            toolMsg.ToolCallId = tcid.GetString();

        return msg;
    }

    public override void Write(Utf8JsonWriter writer, ChatCompletionRequestMessage value, JsonSerializerOptions? options)
    {
        JsonSerializer.Serialize(writer, value, value.GetType(), options ?? new());
    }
}

internal sealed class OneOrManyJsonConverter : JsonConverter<OneOrMany<string>>
{
    public override OneOrMany<string>? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions? options)
    {
        if (reader.TokenType == JsonTokenType.String)
            return new OneOrMany<string>(reader.GetString()!);

        if (reader.TokenType == JsonTokenType.StartArray)
        {
            var list = new List<string>();
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                list.Add(reader.GetString()!);
            return new OneOrMany<string>(list);
        }

        return null;
    }

    public override void Write(Utf8JsonWriter writer, OneOrMany<string> value, JsonSerializerOptions? options)
    {
        if (value.Values.Count == 1)
            writer.WriteStringValue(value.Values[0]);
        else
            JsonSerializer.Serialize(writer, value.Values, options);
    }
}
