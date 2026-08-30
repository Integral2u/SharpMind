using System.Text.Json;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Server.Protocol;
using SharpMind.Tokenization;
using SharpMind.Model;

namespace SharpMind.Tests.Server;

public class OpenAiMapperTests
{
    // ── IsContent ────────────────────────────────────────────────────────

    /// <summary>
    /// Progress entries reuse Token for status text, so the non-streaming path
    /// concatenated "Prefilling 3.99%Prefilling 7.98%..." into the completion.
    /// Only visible on prompts long enough to prefill in more than one chunk —
    /// which is why it hid behind the context-window bug.
    /// </summary>
    [Fact]
    public void IsContent_RejectsPrefillProgressEntries()
    {
        var progress = new ChatStreamEntry { Status = ChatStatus.Updating, Token = "Prefilling 50.25%" };
        Assert.False(OpenAiMapper.IsContent(progress));
    }

    [Theory]
    [InlineData(ChatStatus.Responding)]
    [InlineData(ChatStatus.Thinking)]
    public void IsContent_AcceptsModelOutput(ChatStatus status)
    {
        Assert.True(OpenAiMapper.IsContent(new ChatStreamEntry { Status = status, Token = "hello" }));
    }

    [Theory]
    [InlineData(ChatStatus.Updating)]
    [InlineData(ChatStatus.Executing)]
    [InlineData(ChatStatus.Waiting)]
    [InlineData(ChatStatus.Researching)]
    [InlineData(ChatStatus.ToolCall)]
    public void IsContent_RejectsStatusOnlyEntries(ChatStatus status)
    {
        Assert.False(OpenAiMapper.IsContent(new ChatStreamEntry { Status = status, Token = "status text" }));
    }

    [Fact]
    public void IsContent_RejectsEmptyAndNullTokens()
    {
        Assert.False(OpenAiMapper.IsContent(new ChatStreamEntry { Status = ChatStatus.Responding, Token = "" }));
        Assert.False(OpenAiMapper.IsContent(new ChatStreamEntry { Status = ChatStatus.Responding, Token = null }));
    }

    // ── ToChatHistory ────────────────────────────────────────────────────

    [Fact]
    public void ToChatHistory_SystemMessagesBecomeSystemPrompt()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new SystemMessage { Content = "You are helpful." },
            new UserMessage { Content = "Hello" }
        };

        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Equal("You are helpful.", systemPrompt);
        Assert.Single(history);
        Assert.Equal("Hello", history[0].Content);
    }

    [Fact]
    public void ToChatHistory_MultipleSystemMessagesConcatenated()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new SystemMessage { Content = "Rule 1" },
            new SystemMessage { Content = "Rule 2" },
            new UserMessage { Content = "Hi" }
        };

        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Equal("Rule 1\n\nRule 2", systemPrompt);
        Assert.Single(history);
    }

    [Fact]
    public void ToChatHistory_UserAndAssistantAlternate()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new UserMessage { Content = "Q1" },
            new AssistantMessage { Content = "A1" },
            new UserMessage { Content = "Q2" }
        };

        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Equal("", systemPrompt);
        Assert.Equal(3, history.Count);
        Assert.Equal("Q1", history[0].Content);
        Assert.Equal("A1", history[1].Content);
        Assert.Equal("Q2", history[2].Content);
    }

    [Fact]
    public void ToChatHistory_ToolMessagesIgnored()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new UserMessage { Content = "Hello" },
            new ToolMessage { Content = "result", ToolCallId = "123" },
            new AssistantMessage { Content = "Done" }
        };

        var (_, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Equal(2, history.Count);
    }

    [Fact]
    public void ToChatHistory_EmptyContentIgnored()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new UserMessage { Content = "" },
            new SystemMessage { Content = null },
            new UserMessage { Content = "Real" }
        };

        var (_, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Single(history);
        Assert.Equal("Real", history[0].Content);
    }

    // ── ToResponse ───────────────────────────────────────────────────────

    [Fact]
    public void ToResponse_MatchesOpenAISpec()
    {
        var response = OpenAiMapper.ToResponse("chatcmpl-abc", 1234567890, "model.gguf", "Hello", new CompletionUsage
        {
            PromptTokens = 10,
            CompletionTokens = 5,
            TotalTokens = 15
        });

        Assert.Equal("chatcmpl-abc", response.Id);
        Assert.Equal("chat.completion", response.Object);
        Assert.Equal(1234567890, response.Created);
        Assert.Equal("model.gguf", response.Model);
        Assert.Single(response.Choices);
        Assert.Equal("stop", response.Choices[0].FinishReason);
        Assert.Equal("Hello", response.Choices[0].Message.Content);
        Assert.Equal(15, response.Usage!.TotalTokens);
    }

    // ── ToStreamChunk ────────────────────────────────────────────────────

    [Fact]
    public void ToStreamChunk_MatchesOpenAISpec()
    {
        var chunk = OpenAiMapper.ToStreamChunk("chatcmpl-abc", 1234567890, "model.gguf", "Hi", null, null);

        Assert.Equal("chatcmpl-abc", chunk.Id);
        Assert.Equal("chat.completion.chunk", chunk.Object);
        Assert.Equal("Hi", chunk.Choices[0].Delta.Content);
        Assert.Null(chunk.Choices[0].FinishReason);
    }

    [Fact]
    public void ToStreamRoleChunk_HasRole()
    {
        var chunk = OpenAiMapper.ToStreamRoleChunk("chatcmpl-abc", 1234567890, "model.gguf");

        Assert.Equal("assistant", chunk.Choices[0].Delta.Role);
        Assert.Null(chunk.Choices[0].Delta.Content);
    }

    // ── ModelObject ──────────────────────────────────────────────────────

    [Fact]
    public void ToModelInfo_MatchesOpenAISpec()
    {
        var model = OpenAiMapper.ToModelInfo("SmolLM.gguf", 1686935002);

        Assert.Equal("SmolLM.gguf", model.Id);
        Assert.Equal("model", model.Object);
        Assert.Equal(1686935002, model.Created);
        Assert.Equal("sharpmind", model.OwnedBy);
    }

    // ── ExtractTextContent ─────────────────────────────────────────────

    [Fact]
    public void ExtractTextContent_NullReturnsEmpty()
    {
        Assert.Equal("", OpenAiMapper.ExtractTextContent(null));
    }

    [Fact]
    public void ExtractTextContent_PlainStringPassesThrough()
    {
        Assert.Equal("hello", OpenAiMapper.ExtractTextContent("hello"));
    }

    [Fact]
    public void ExtractTextContent_MultiPartArrayExtractsText()
    {
        var json = """[{"type":"text","text":"hello"},{"type":"text","text":" world"}]""";
        Assert.Equal("hello world", OpenAiMapper.ExtractTextContent(json));
    }

    [Fact]
    public void ExtractTextContent_SinglePartArrayExtractsText()
    {
        var json = """[{"type":"text","text":"just this"}]""";
        Assert.Equal("just this", OpenAiMapper.ExtractTextContent(json));
    }

    [Fact]
    public void ExtractTextContent_IgnoresNonTextParts()
    {
        var json = """[{"type":"image_url","url":"https://example.com/img.png"},{"type":"text","text":"describe this"}]""";
        Assert.Equal("describe this", OpenAiMapper.ExtractTextContent(json));
    }

    [Fact]
    public void ExtractTextContent_MalformedJsonPassesThrough()
    {
        Assert.Equal("not json at all", OpenAiMapper.ExtractTextContent("not json at all"));
    }

    [Fact]
    public void ExtractTextContent_NonStringContentCallsToString()
    {
        Assert.Equal("42", OpenAiMapper.ExtractTextContent(42));
    }

    [Fact]
    public void ExtractTextContent_EmptyArrayReturnsEmpty()
    {
        Assert.Equal("[]", OpenAiMapper.ExtractTextContent("[]"));
    }

    // ── ToChatHistory with multi-part content ──────────────────────────

    [Fact]
    public void ToChatHistory_MultiPartContentExtractsText()
    {
        var messages = new List<ChatCompletionRequestMessage>
        {
            new UserMessage { Content = """[{"type":"text","text":"What is 2+2?"}]""" }
        };

        var (_, history) = OpenAiMapper.ToChatHistory(messages);

        Assert.Single(history);
        Assert.Equal("What is 2+2?", history[0].Content);
    }

    // ── ApplyToSession stop strings ───────────────────────────────────

    [Fact]
    public void ApplyToSession_SetsStopStrings()
    {
        var session = new FakeTestSession();
        var request = new CreateChatCompletionRequest
        {
            Stop = new OneOrMany<string>(new List<string> { "\n\n", "Human:" })
        };

        OpenAiMapper.ApplyToSession(request, session);

        Assert.NotNull(session.StopStrings);
        Assert.Equal(2, session.StopStrings!.Count);
        Assert.Contains("\n\n", session.StopStrings);
        Assert.Contains("Human:", session.StopStrings);
    }

    [Fact]
    public void ApplyToSession_SingleStopString()
    {
        var session = new FakeTestSession();
        var request = new CreateChatCompletionRequest
        {
            Stop = new OneOrMany<string>("STOP")
        };

        OpenAiMapper.ApplyToSession(request, session);

        Assert.NotNull(session.StopStrings);
        Assert.Single(session.StopStrings!);
        Assert.Equal("STOP", session.StopStrings![0]);
    }

    private sealed class FakeTestSession : IChatSession
    {
        public int MaxTokens { get; set; }
        public int MaxNewTokens { get; set; }
        public float Temperature { get; set; }
        public int TopK { get; set; }
        public float TopP { get; set; }
        public float RepetitionPenalty { get; set; }
        public int RepetitionWindow { get; set; }
        public IReadOnlyList<int>? StopTokenIds { get; set; }
        public IReadOnlyList<string>? StopStrings { get; set; }
        public bool ShowThinking { get; set; }
        public bool EnableThinking { get; set; }
        public string UserName { get; set; } = "User";
        public Func<string, System.Text.Json.Nodes.JsonObject, CancellationToken, Task<ToolRequestResult>>? ProcessToolRequest { get; set; }
        public float? TokensPerSecond => null;
        public float? TimeToFirstToken => null;
        public Tokenizer Tokenizer => throw new NotSupportedException();
        public Transformer Model => throw new NotSupportedException();
        public IReadOnlyList<ChatMessage> History => [];
        public IChatPromptFormatter? Formatter => null;
        public int LastPrefillTokenCount => 0;
        public void AddMessage(ChatRole role, string content) { }
        public void AddMessage(ChatMessage message) { }
        public string GetFormattedPrompt() => "";
        public void ClearHistory() { }
        public void ResetCaches() { }
        public void Interrupt() { }
        public void InitializeChat(IProgress<float>? progress = null) { }
        public ChatSessionSnapshot GetSnapshot() => new() { History = [] };
        public void LoadSnapshot(ChatSessionSnapshot snapshot) { }
        public IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(string userInput, ChatArtifact[]? artifacts = null, CancellationToken ct = default) => throw new NotSupportedException();
        public Task<ChatMessage[]> StartChatAsync(Func<Task<ChatMessage>> prompt, Action<ChatStreamEntry> response, CancellationToken token = default) => throw new NotSupportedException();
        public Task<ChatMessage[]> StartChatAsync(Func<ChatMessage> prompt, Action<ChatStreamEntry> response, CancellationToken token = default) => throw new NotSupportedException();
        public Task<ChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default) => throw new NotSupportedException();
        public Task<ChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default) => throw new NotSupportedException();
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
