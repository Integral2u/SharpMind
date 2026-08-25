using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;

namespace SharpMind.Tests.ExtensionsAI;

/// <summary>
/// Tests for <see cref="ChatMessageConverter"/>: bidirectional mapping between
/// Microsoft.Extensions.AI types and SharpMind chat types.
/// </summary>
public sealed class ChatMessageConverterTests
{
    [Theory]
    [InlineData("system", SmChatRole.System)]
    [InlineData("user", SmChatRole.User)]
    [InlineData("assistant", SmChatRole.Agent)]
    [InlineData("unknown", SmChatRole.User)]
    public void ToSharpMind_MapsRole(string meaiRole, SmChatRole expectedRole)
    {
        var meai = new MeChatMessage(new MeChatRole(meaiRole), "Hello");
        var sm = ChatMessageConverter.ToSharpMind(meai);
        Assert.Equal(expectedRole, sm.Role);
    }

    [Fact]
    public void ToSharpMind_PrefersTextOverContents()
    {
        var meai = new MeChatMessage(MeChatRole.User, "Direct text");
        var sm = ChatMessageConverter.ToSharpMind(meai);
        Assert.Equal("Direct text", sm.Content);
    }

    [Fact]
    public void ToSharpMind_FallsBackToTextContentList()
    {
        var meai = new MeChatMessage(MeChatRole.User, new List<AIContent>
        {
            new TextContent("part1"),
            new TextContent("part2")
        });
        var sm = ChatMessageConverter.ToSharpMind(meai);
        Assert.Equal("part1part2", sm.Content);
    }

    [Fact]
    public void ToSharpMind_EmptyContent()
    {
        var meai = new MeChatMessage(MeChatRole.User, (string?)null);
        var sm = ChatMessageConverter.ToSharpMind(meai);
        Assert.Equal(string.Empty, sm.Content);
    }

    [Fact]
    public void ToSharpMind_PreservesAuthorName()
    {
        var meai = new MeChatMessage(MeChatRole.User, "Hi") { AuthorName = "Alice" };
        var sm = ChatMessageConverter.ToSharpMind(meai);
        Assert.Equal("Alice", sm.Name);
    }

    [Theory]
    [InlineData(SmChatRole.System, "system")]
    [InlineData(SmChatRole.User, "user")]
    [InlineData(SmChatRole.Agent, "assistant")]
    public void ToMEAI_MapsRole(SmChatRole smRole, string expectedRole)
    {
        var sm = new SmChatMessage { Role = smRole, Content = "test" };
        var meai = ChatMessageConverter.ToMEAI(sm);
        Assert.Equal(expectedRole, meai.Role.Value);
    }

    [Fact]
    public void ToMEAI_PreservesContent()
    {
        var sm = new SmChatMessage { Role = SmChatRole.User, Content = "Hello MEAI" };
        var meai = ChatMessageConverter.ToMEAI(sm);
        Assert.Equal("Hello MEAI", meai.Text);
    }

    [Fact]
    public void ApplyOptions_NullOptions_NoOp()
    {
        var session = new FakeChatSession();
        ChatMessageConverter.ApplyOptions(null, session);
        Assert.Equal(0f, session.Temperature);
    }

    [Fact]
    public void ApplyOptions_SetsTemperature()
    {
        var session = new FakeChatSession();
        var opts = new ChatOptions { Temperature = 0.7f };
        ChatMessageConverter.ApplyOptions(opts, session);
        Assert.Equal(0.7f, session.Temperature);
    }

    [Fact]
    public void ApplyOptions_SetsTopP()
    {
        var session = new FakeChatSession();
        var opts = new ChatOptions { TopP = 0.9f };
        ChatMessageConverter.ApplyOptions(opts, session);
        Assert.Equal(0.9f, session.TopP);
    }

    [Fact]
    public void ApplyOptions_SetsMaxOutputTokens()
    {
        var session = new FakeChatSession();
        var opts = new ChatOptions { MaxOutputTokens = 512 };
        ChatMessageConverter.ApplyOptions(opts, session);
        Assert.Equal(512, session.MaxNewTokens);
    }

    [Fact]
    public void ApplyOptions_NullableFieldsNotApplied()
    {
        var session = new FakeChatSession();
        session.Temperature = 0.5f;
        var opts = new ChatOptions();
        ChatMessageConverter.ApplyOptions(opts, session);
        Assert.Equal(0.5f, session.Temperature);
    }

    private sealed class FakeChatSession : SmIChatSession
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
        public float? TokensPerSecond => null;
        public float? TimeToFirstToken => null;
        public SharpMind.Tokenization.Tokenizer Tokenizer => throw new NotSupportedException();
        public SharpMind.Model.Transformer Model => throw new NotSupportedException();
        public IReadOnlyList<SmChatMessage> History => Array.Empty<SmChatMessage>();
        public SmIChatPromptFormatter? Formatter => null;
        public int LastPrefillTokenCount => 0;
        public void AddMessage(SmChatRole role, string content) { }
        public void AddMessage(SmChatMessage message) { }
        public string GetFormattedPrompt() => "";
        public void ClearHistory() { }
        public void ResetCaches() { }
        public void Interrupt() { }
        public void InitializeChat(IProgress<float>? progress = null) { }
        public SmChatSessionSnapshot GetSnapshot() => new() { History = [] };
        public void LoadSnapshot(SmChatSessionSnapshot snapshot) { }
        public async IAsyncEnumerable<SmChatStreamEntry> GetResponseStreamAsync(string userInput, SmChatArtifact[]? artifacts = null, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.CompletedTask;
            yield break;
        }
        public Task<SmChatMessage[]> StartChatAsync(Func<Task<SmChatMessage>> prompt, Action<SmChatStreamEntry> response, CancellationToken token = default) => Task.FromResult(Array.Empty<SmChatMessage>());
        public Task<SmChatMessage[]> StartChatAsync(Func<SmChatMessage> prompt, Action<SmChatStreamEntry> response, CancellationToken token = default) => Task.FromResult(Array.Empty<SmChatMessage>());
        public Task<SmChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default) => Task.FromResult(Array.Empty<SmChatMessage>());
        public Task<SmChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default) => Task.FromResult(Array.Empty<SmChatMessage>());
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
