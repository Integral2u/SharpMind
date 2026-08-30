using Microsoft.Extensions.AI;
using SharpMind.Extensions.AI;

namespace SharpMind.Tests.ExtensionsAI;

/// <summary>
/// Tests for <see cref="SharpMindChatClient"/>: the
/// <see cref="IChatClient"/> adapter wrapping SharpMind sessions.
/// </summary>
public sealed class SharpMindChatClientTests
{
    [Fact]
    public async Task GetResponseAsync_SubmitsUserMessage()
    {
        var session = new FakeChatSession(responseText: "Hi there");
        using var client = new SharpMindChatClient(session);

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.User, "Hello")
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.Equal("Hi there", response.Text);
        Assert.Single(response.Messages);
        Assert.Equal(MeChatRole.Assistant, response.Messages[0].Role);
    }

    [Fact]
    public async Task GetResponseAsync_NullUserMessage_ReturnsEmpty()
    {
        var session = new FakeChatSession(responseText: "n/a");
        using var client = new SharpMindChatClient(session);

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.System, "You are helpful")
        };

        var response = await client.GetResponseAsync(messages);
        Assert.Empty(response.Messages);
    }

    [Fact]
    public async Task GetResponseAsync_AppliesOptions()
    {
        var session = new FakeChatSession(responseText: "ok");
        using var client = new SharpMindChatClient(session);

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.User, "test")
        };

        await client.GetResponseAsync(messages, new ChatOptions
        {
            Temperature = 0.8f,
            MaxOutputTokens = 128
        });

        Assert.Equal(0.8f, session.Temperature);
        Assert.Equal(128, session.MaxNewTokens);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_YieldsTokens()
    {
        var session = new FakeChatSession(responseText: "Hello World");
        using var client = new SharpMindChatClient(session);

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.User, "Say hi")
        };

        var chunks = new List<string>();
        await foreach (var update in client.GetStreamingResponseAsync(messages))
        {
            if (update.Text is { Length: > 0 } text)
                chunks.Add(text);
        }

        var combined = string.Concat(chunks);
        Assert.Equal("Hello World", combined);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_EmptyUserMessage_YieldsNothing()
    {
        var session = new FakeChatSession(responseText: "should not see this");
        using var client = new SharpMindChatClient(session);

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.System, "System only")
        };

        var count = 0;
        await foreach (var _ in client.GetStreamingResponseAsync(messages))
            count++;

        Assert.Equal(0, count);
    }

    [Fact]
    public void GetService_ReturnsSelf()
    {
        var session = new FakeChatSession(responseText: "");
        using var client = new SharpMindChatClient(session);

        Assert.Same(client, client.GetService(typeof(SharpMindChatClient)));
        Assert.Same(client, client.GetService(typeof(SmIChatSession)));
        Assert.Null(client.GetService(typeof(string)));
    }

    [Fact]
    public void Constructor_NullSession_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new SharpMindChatClient(null!));
    }

    [Fact]
    public void Dispose_CallsDisposeAsync()
    {
        var session = new FakeChatSession(responseText: "");
        var client = new SharpMindChatClient(session);
        client.Dispose();
        Assert.True(session.Disposed);
    }

    // ------------------------------------------------------------------
    // Tool calls. The session streams every generated fragment as it arrives
    // and only parses the completed buffer for a <tool_call> afterwards, so
    // the markup reaches this adapter as ordinary Responding text. It must
    // not reach the MEAI caller that way, and a call handed back by
    // ToolRequestOutcome.ReturnToCaller must surface as FunctionCallContent.
    // ------------------------------------------------------------------

    private static System.Text.Json.Nodes.JsonObject WeatherCall() => new()
    {
        ["tool"] = "get_weather",
        ["arguments"] = new System.Text.Json.Nodes.JsonObject { ["city"] = "Delft" }
    };

    private const string ToolCallMarkup =
        """<tool_call>{"tool":"get_weather","arguments":{"city":"Delft"}}</tool_call>""";

    [Fact]
    public async Task GetStreamingResponseAsync_DoesNotLeakToolCallMarkupAsText()
    {
        // The native loop dispatched the tool itself and carried on — the default
        // wiring. The caller should see the reply, never the call that produced it.
        var session = new FakeChatSession(responseText: ToolCallMarkup + "It is 19C in Delft.");
        using var client = new SharpMindChatClient(session);

        var text = new System.Text.StringBuilder();
        await foreach (var update in client.GetStreamingResponseAsync([new(MeChatRole.User, "weather?")]))
            if (update.Text is { Length: > 0 } t) text.Append(t);

        Assert.DoesNotContain("<tool_call>", text.ToString());
        Assert.DoesNotContain("get_weather", text.ToString());
        Assert.Contains("It is 19C in Delft.", text.ToString());
    }

    [Fact]
    public async Task GetResponseAsync_DoesNotLeakToolCallMarkupAsText()
    {
        var session = new FakeChatSession(responseText: ToolCallMarkup + "It is 19C in Delft.");
        using var client = new SharpMindChatClient(session);

        var response = await client.GetResponseAsync([new(MeChatRole.User, "weather?")]);

        Assert.DoesNotContain("<tool_call>", response.Text);
        Assert.Contains("It is 19C in Delft.", response.Text);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_ReturnedToolCall_YieldsFunctionCallContent()
    {
        var session = new FakeChatSession(responseText: ToolCallMarkup, toolCall: WeatherCall());
        using var client = new SharpMindChatClient(session);

        var contents = new List<AIContent>();
        await foreach (var update in client.GetStreamingResponseAsync([new(MeChatRole.User, "weather?")]))
            contents.AddRange(update.Contents);

        var call = Assert.Single(contents.OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.NotNull(call.Arguments);
        Assert.Equal("Delft", call.Arguments!["city"]?.ToString());
    }

    [Fact]
    public async Task GetStreamingResponseAsync_ReturnedToolCall_FinishesWithToolCalls()
    {
        var session = new FakeChatSession(responseText: ToolCallMarkup, toolCall: WeatherCall());
        using var client = new SharpMindChatClient(session);

        ChatFinishReason? finish = null;
        await foreach (var update in client.GetStreamingResponseAsync([new(MeChatRole.User, "weather?")]))
            finish = update.FinishReason ?? finish;

        Assert.Equal(ChatFinishReason.ToolCalls, finish);
    }

    [Fact]
    public async Task GetResponseAsync_ReturnedToolCall_YieldsFunctionCallContent()
    {
        var session = new FakeChatSession(responseText: ToolCallMarkup, toolCall: WeatherCall());
        using var client = new SharpMindChatClient(session);

        var response = await client.GetResponseAsync([new(MeChatRole.User, "weather?")]);

        var call = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_ReplyOpeningWithBrace_IsStillDelivered()
    {
        // Held-back text must be released once it can no longer be a tool call,
        // and flushed at end of turn — otherwise a reply that merely opens with
        // "{" would be swallowed.
        var session = new FakeChatSession(responseText: "{not a tool call after all}");
        using var client = new SharpMindChatClient(session);

        var text = new System.Text.StringBuilder();
        await foreach (var update in client.GetStreamingResponseAsync([new(MeChatRole.User, "hi")]))
            if (update.Text is { Length: > 0 } t) text.Append(t);

        Assert.Equal("{not a tool call after all}", text.ToString());
    }

    private sealed class FakeChatSession : SmIChatSession
    {
        private readonly string _responseText;
        private readonly System.Text.Json.Nodes.JsonObject? _toolCall;
        public bool Disposed;

        /// <param name="responseText">
        /// Streamed one character at a time as <see cref="SmChatStatus.Responding"/>
        /// entries, mirroring the real session: every generated fragment is yielded
        /// as it arrives, *before* the completed buffer is parsed for a tool call.
        /// </param>
        /// <param name="toolCall">
        /// When set, the turn ends with a <see cref="SmChatStatus.ToolCall"/> entry
        /// carrying this call and no <see cref="SmChatStatus.Complete"/> entry —
        /// exactly what the real session yields for
        /// <see cref="SharpMind.Inference.Chat.ToolRequestOutcome.ReturnToCaller"/>.
        /// </param>
        public FakeChatSession(string responseText, System.Text.Json.Nodes.JsonObject? toolCall = null)
        {
            _responseText = responseText;
            _toolCall = toolCall;
        }

        public float Temperature { get; set; }
        public int TopK { get; set; }
        public float TopP { get; set; }
        public int MaxNewTokens { get; set; }
        public int MaxTokens { get; set; }
        public float RepetitionPenalty { get; set; }
        public int RepetitionWindow { get; set; }
        public IReadOnlyList<int>? StopTokenIds { get; set; }
        public IReadOnlyList<string>? StopStrings { get; set; }
        public bool ShowThinking { get; set; }
        public bool EnableThinking { get; set; }
        public string UserName { get; set; } = "User";
        public Func<string, System.Text.Json.Nodes.JsonObject, CancellationToken, Task<SharpMind.Inference.Chat.ToolRequestResult>>? ProcessToolRequest { get; set; }
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

        public Task<SmChatMessage[]> StartChatAsync(
            Func<SmChatMessage> prompt,
            Action<SmChatStreamEntry> response,
            CancellationToken token = default)
        {
            var input = prompt();
            if (string.IsNullOrWhiteSpace(input.Content))
                return Task.FromResult(Array.Empty<SmChatMessage>());

            foreach (char c in _responseText)
            {
                response(new SmChatStreamEntry
                {
                    Status = SmChatStatus.Responding,
                    Token = c.ToString(),
                    IsComplete = false
                });
            }

            if (_toolCall is not null)
            {
                // The real session hands the call back and ends the turn without
                // a Complete entry (ChatSession.cs, ToolRequestOutcome.ReturnToCaller).
                response(new SmChatStreamEntry
                {
                    Status = SmChatStatus.ToolCall,
                    Token = _toolCall["tool"]!.GetValue<string>(),
                    ToolCall = _toolCall,
                    IsComplete = true
                });

                return Task.FromResult(new[]
                {
                    new SmChatMessage { Role = SmChatRole.User, Content = input.Content }
                });
            }

            response(new SmChatStreamEntry
            {
                Status = SmChatStatus.Complete,
                Token = null,
                IsComplete = true
            });

            return Task.FromResult(new[]
            {
                new SmChatMessage { Role = SmChatRole.User, Content = input.Content },
                new SmChatMessage { Role = SmChatRole.Agent, Content = _responseText }
            });
        }

        public Task<SmChatMessage[]> StartChatAsync(
            Func<Task<SmChatMessage>> prompt,
            Action<SmChatStreamEntry> response,
            CancellationToken token = default)
            => StartChatAsync(() => prompt().GetAwaiter().GetResult(), response, token);

        public Task<SmChatMessage[]> StartChatAsync(
            Func<string> prompt,
            Action<string> response,
            CancellationToken token = default)
            => StartChatAsync(
                () => new SmChatMessage { Content = prompt(), Role = SmChatRole.User, Name = "User" },
                e => { if (e.Token is { Length: > 0 } t) response(t); },
                token);

        public Task<SmChatMessage[]> StartChatAsync(
            Func<Task<string>> prompt,
            Action<string> response,
            CancellationToken token = default)
            => StartChatAsync(() => prompt().GetAwaiter().GetResult(), response, token);

        public ValueTask DisposeAsync()
        {
            Disposed = true;
            return ValueTask.CompletedTask;
        }
    }
}
