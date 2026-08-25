using System.Text.Json;
using SharpMind.Server.Protocol;

namespace SharpMind.Tests.Server;

/// <summary>
/// Verifies that the protocol types serialize to JSON matching the exact
/// OpenAI API spec field names and structure.
/// </summary>
public class ProtocolSerializationTests
{
    private static readonly JsonSerializerOptions s_options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    [Fact]
    public void ListModelsResponse_SerializesCorrectly()
    {
        var response = new ListModelsResponse
        {
            Data =
            [
                new ModelObject { Id = "model.gguf", Created = 1686935002 }
            ]
        };

        var json = JsonSerializer.Serialize(response, s_options);
        var doc = JsonDocument.Parse(json);

        Assert.Equal("list", doc.RootElement.GetProperty("object").GetString());
        Assert.Equal(1, doc.RootElement.GetProperty("data").GetArrayLength());

        var model = doc.RootElement.GetProperty("data")[0];
        Assert.Equal("model.gguf", model.GetProperty("id").GetString());
        Assert.Equal("model", model.GetProperty("object").GetString());
        Assert.Equal(1686935002, model.GetProperty("created").GetInt64());
        Assert.Equal("sharpmind", model.GetProperty("owned_by").GetString());
    }

    [Fact]
    public void CreateChatCompletionResponse_SerializesCorrectly()
    {
        var response = new CreateChatCompletionResponse
        {
            Id = "chatcmpl-123",
            Created = 1694268190,
            Model = "gpt-4",
            Choices =
            [
                new ChatCompletionChoice
                {
                    Index = 0,
                    Message = new ChatCompletionResponseMessage { Content = "Hello" },
                    FinishReason = "stop"
                }
            ],
            Usage = new CompletionUsage
            {
                PromptTokens = 10,
                CompletionTokens = 5,
                TotalTokens = 15
            }
        };

        var json = JsonSerializer.Serialize(response, s_options);
        var doc = JsonDocument.Parse(json);

        Assert.Equal("chatcmpl-123", doc.RootElement.GetProperty("id").GetString());
        Assert.Equal("chat.completion", doc.RootElement.GetProperty("object").GetString());
        Assert.Equal(1694268190, doc.RootElement.GetProperty("created").GetInt64());
        Assert.Equal("gpt-4", doc.RootElement.GetProperty("model").GetString());

        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal(0, choice.GetProperty("index").GetInt32());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        Assert.Equal("Hello", choice.GetProperty("message").GetProperty("content").GetString());

        var usage = doc.RootElement.GetProperty("usage");
        Assert.Equal(10, usage.GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(5, usage.GetProperty("completion_tokens").GetInt32());
        Assert.Equal(15, usage.GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public void StreamChunk_SerializesCorrectly()
    {
        var chunk = new CreateChatCompletionStreamResponse
        {
            Id = "chatcmpl-123",
            Created = 1694268190,
            Model = "gpt-4",
            Choices =
            [
                new ChatCompletionStreamChoice
                {
                    Index = 0,
                    Delta = new ChatCompletionStreamDelta { Content = "Hello" },
                    FinishReason = null
                }
            ]
        };

        var json = JsonSerializer.Serialize(chunk, s_options);
        var doc = JsonDocument.Parse(json);

        Assert.Equal("chat.completion.chunk", doc.RootElement.GetProperty("object").GetString());
        var delta = doc.RootElement.GetProperty("choices")[0].GetProperty("delta");
        Assert.Equal("Hello", delta.GetProperty("content").GetString());
        Assert.False(doc.RootElement.GetProperty("choices")[0].TryGetProperty("finish_reason", out _));
    }

    [Fact]
    public void DeleteModelResponse_SerializesCorrectly()
    {
        var response = new DeleteModelResponse { Id = "model.gguf", Deleted = true };

        var json = JsonSerializer.Serialize(response, s_options);
        var doc = JsonDocument.Parse(json);

        Assert.Equal("model.gguf", doc.RootElement.GetProperty("id").GetString());
        Assert.True(doc.RootElement.GetProperty("deleted").GetBoolean());
    }

    [Fact]
    public void CreateChatCompletionRequest_DeserializesCorrectly()
    {
        var json = """
        {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "stream": true
        }
        """;

        var request = JsonSerializer.Deserialize<CreateChatCompletionRequest>(json, s_options);

        Assert.NotNull(request);
        Assert.Equal("gpt-4", request!.Model);
        Assert.Equal(2, request.Messages.Count);
        Assert.IsType<SystemMessage>(request.Messages[0]);
        Assert.IsType<UserMessage>(request.Messages[1]);
        Assert.Equal(0.7f, request.Temperature);
        Assert.Equal(0.9f, request.TopP);
        Assert.Equal(100, request.MaxTokens);
        Assert.True(request.Stream);
    }

    [Fact]
    public void StopParameter_DeserializesStringOrArray()
    {
        // Single string
        var json1 = """{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":"END"}""";
        var req1 = JsonSerializer.Deserialize<CreateChatCompletionRequest>(json1, s_options);
        Assert.NotNull(req1!.Stop);
        Assert.Single(req1.Stop!.Values);
        Assert.Equal("END", req1.Stop.Values[0]);

        // Array
        var json2 = """{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":["END","STOP"]}""";
        var req2 = JsonSerializer.Deserialize<CreateChatCompletionRequest>(json2, s_options);
        Assert.NotNull(req2!.Stop);
        Assert.Equal(2, req2.Stop!.Values.Count);
    }
}
