using System.Runtime.CompilerServices;
using System.Text;
using SharpMind.Data.Batching;
using SharpMind.Data.Records;
using SharpMind.Data.Sources;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;
using SharpMind.Training.Sft;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// B6: the role-aware tokeniser masks exactly the assistant's own tokens
/// (content plus end-of-turn) for the three main chat skeletons, and leaves
/// prompt, system, template scaffolding and the trailing generation prompt
/// unmasked.
/// </summary>
public sealed class ConversationTokenizerTests
{
    // A char-level tokenizer whose token i is exactly character i of the input,
    // so masked token indices can be read straight back as a substring. The
    // characters not present in the tiny corpus fall back to UNK, which is still
    // one token per character.
    private static async Task<Tokenizer> CharTokenizerAsync()
    {
        var docs = new ListSource(
        [
            "abcxyz ABCXYZ 0123 \n\t<>|/_-.:!?",
            "hello world hi pong q1 q2 a1 a2 s user assistant model system",
        ]);
        return await TokenizationPipeline.TrainCharacterAsync(docs);
    }

    private static string Masked(TokenizedExample example, string text)
    {
        Assert.NotNull(example.AssistantMask);
        Assert.Equal(text.Length, example.TokenIds.Count);
        var sb = new StringBuilder();
        for (int i = 0; i < example.AssistantMask!.Count; i++)
            if (example.AssistantMask[i])
                sb.Append(text[i]);
        return sb.ToString();
    }

    // ── ChatML (built-in formatter) ─────────────────────────────────────────

    private const string ChatMlTokens =
        "<|im_start|><|im_end|><|system|><|user|><|assistant|>";

    [Fact]
    public async Task ChatML_MasksAssistantContentAndEndOfTurn()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new ChatMLFormatter(ChatMlTokens), tok, addBos: false);

        var history = new[] { ChatMessage.User("hi"), ChatMessage.Agent("pong") };
        var example = conv.Encode(history);

        const string golden =
            "<|im_start|>user\nhi<|im_end|>\n" +
            "<|im_start|>assistant\npong<|im_end|>\n" +
            "<|im_start|>assistant\n";
        Assert.Equal(golden.Length, example.TokenIds.Count);
        Assert.Equal("pong<|im_end|>\n", Masked(example, golden));
    }

    [Fact]
    public async Task ChatML_MultiTurn_MasksEveryAssistantTurn()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new ChatMLFormatter(ChatMlTokens), tok, addBos: false);

        var history = new[]
        {
            ChatMessage.System("s"),
            ChatMessage.User("q1"),
            ChatMessage.Agent("a1"),
            ChatMessage.User("q2"),
            ChatMessage.Agent("a2"),
        };
        var example = conv.Encode(history);

        const string golden =
            "<|im_start|>system\ns<|im_end|>\n" +
            "<|im_start|>user\nq1<|im_end|>\n" +
            "<|im_start|>assistant\na1<|im_end|>\n" +
            "<|im_start|>user\nq2<|im_end|>\n" +
            "<|im_start|>assistant\na2<|im_end|>\n" +
            "<|im_start|>assistant\n";
        Assert.Equal("a1<|im_end|>\na2<|im_end|>\n", Masked(example, golden));
    }

    [Fact]
    public async Task NoAssistantTurns_MeansNoMask()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new ChatMLFormatter(ChatMlTokens), tok);

        var example = conv.Encode(new[] { ChatMessage.User("hi") });

        Assert.Null(example.AssistantMask);
        Assert.True(example.IsAssistant(0));
    }

    // ── Llama 3 (built-in formatter) ────────────────────────────────────────

    [Fact]
    public async Task Llama3_MasksAssistantContentAndEot()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new Llama3Formatter(), tok, addBos: false);

        var history = new[] { ChatMessage.User("hello"), ChatMessage.Agent("world") };
        var example = conv.Encode(history);

        const string golden =
            "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\nworld<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
        Assert.Equal("world<|eot_id|>", Masked(example, golden));
    }

    // ── Gemma 3 (Jinja template) ────────────────────────────────────────────

    private const string Gemma3Template =
        "{% for message in messages %}" +
        "{% if message['role'] == 'assistant' %}" +
        "{{ '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' }}" +
        "{% else %}" +
        "{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}" +
        "{% endif %}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";

    [Fact]
    public async Task Gemma3_MasksAssistantContentAndEndOfTurn()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new JinjaTemplateFormatter(Gemma3Template), tok);

        var history = new[] { ChatMessage.User("hi"), ChatMessage.Agent("pong") };
        var example = conv.Encode(history);

        const string golden =
            "<start_of_turn>user\nhi<end_of_turn>\n" +
            "<start_of_turn>model\npong<end_of_turn>\n" +
            "<start_of_turn>model\n";
        Assert.Equal("pong<end_of_turn>\n", Masked(example, golden));
    }

    // ── DataRecord adapters ─────────────────────────────────────────────────

    [Fact]
    public async Task RecordWithTurns_MasksAssistantTurns()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new ChatMLFormatter(ChatMlTokens), tok, addBos: false);

        var record = new DataRecord
        {
            Text = "unused",
            Turns =
            [
                new ConversationTurn("user", "hi"),
                new ConversationTurn("assistant", "pong"),
            ],
        };

        var example = conv.Encode(record);
        Assert.NotNull(example.AssistantMask);

        var text = "<|im_start|>user\nhi<|im_end|>\n" +
                   "<|im_start|>assistant\npong<|im_end|>\n" +
                   "<|im_start|>assistant\n";
        Assert.Equal("pong<|im_end|>\n", Masked(example, text));
    }

    [Fact]
    public async Task RawTextRecord_IsPlainLanguageModelExample()
    {
        var tok = await CharTokenizerAsync();
        var conv = new ConversationTokenizer(new ChatMLFormatter(ChatMlTokens), tok);

        var example = conv.Encode(new DataRecord { Text = "some pretraining text" });

        Assert.Null(example.AssistantMask);
    }

    [Theory]
    [InlineData("assistant", ChatRole.Agent)]
    [InlineData("model", ChatRole.Agent)]
    [InlineData("agent", ChatRole.Agent)]
    [InlineData("user", ChatRole.User)]
    [InlineData("system", ChatRole.System)]
    [InlineData("tool", ChatRole.Tool)]
    [InlineData("weird", ChatRole.User)]
    public void ParseRole_MapsKnownAndUnknownRoles(string role, ChatRole expected)
        => Assert.Equal(expected, ConversationTokenizer.ParseRole(role));

    // ── Minimal in-memory data source ───────────────────────────────────────

    private sealed class ListSource(IEnumerable<string> docs) : IDataSource
    {
        public long? EstimatedCount => null;
        public string Description => "test";

        public async IAsyncEnumerable<string> ReadAsync(
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var doc in docs)
            {
                await Task.Yield();
                yield return doc;
            }
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
