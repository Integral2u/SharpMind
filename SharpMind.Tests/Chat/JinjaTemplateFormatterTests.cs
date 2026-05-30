using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.Chat;

/// <summary>
/// Verifies <see cref="JinjaTemplateFormatter"/> against golden outputs produced
/// by Python Jinja2 for each template found in the GGUF test models.
/// </summary>
public sealed class JinjaTemplateFormatterTests
{
    // ── Template strings (trimmed to exact content from GgufMeta.KvPairs) ──

    private const string SmolLM2Template =
        "{% for message in messages %}" +
        "{% if loop.first and messages[0]['role'] != 'system' %}" +
        "{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n' }}" +
        "{% endif %}" +
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private const string Qwen2Template =
        "{% for message in messages %}" +
        "{% if loop.first and messages[0]['role'] != 'system' %}" +
        "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}" +
        "{% endif %}" +
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}" +
        "{% endfor %}" +
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

    private const string TinyLlamaTemplate =
        "{% for message in messages %}\n" +
        "{% if message['role'] == 'user' %}\n" +
        "{{ '<|user|>\\n' + message['content'] + eos_token }}\n" +
        "{% elif message['role'] == 'system' %}\n" +
        "{{ '<|system|>\\n' + message['content'] + eos_token }}\n" +
        "{% elif message['role'] == 'assistant' %}\n" +
        "{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n" +
        "{% endif %}\n" +
        "{% if loop.last and add_generation_prompt %}\n" +
        "{{ '<|assistant|>' }}\n" +
        "{% endif %}\n" +
        "{% endfor %}";

    private const string Llama3Template =
        "{% set loop_messages = messages %}" +
        "{% for message in loop_messages %}" +
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'" +
        " + message['content'] | trim + '<|eot_id|>' %}" +
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}" +
        "{{ content }}" +
        "{% endfor %}" +
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}";

    // ── Stub tokenizer ──────────────────────────────────────────────────

    /// <summary>
    /// Minimal tokenizer built via the public <see cref="Tokenizer.FromGguf"/> factory.
    /// Only <see cref="Tokenizer.BosId"/>, <see cref="Tokenizer.EosId"/>, and
    /// <see cref="Tokenizer.IdToToken"/> are exercised by the formatter tests.
    /// </summary>
    private static readonly Tokenizer Tok = Tokenizer.FromGguf(
        tokens: ["[UNK]", "<s>", "</s>"],
        merges: null,
        tokenTypes: null,
        bosId: 1,
        eosId: 2);

    // ── Helpers ─────────────────────────────────────────────────────────

    private static string Render(string template, IReadOnlyList<ChatMessage> messages,
                                  bool addBos = false)
    {
        var fmt = new JinjaTemplateFormatter(template);
        return fmt.Format(messages, Tok, addBos);
    }

    // ── SmolLM2 ──────────────────────────────────────────────────────────

    [Fact]
    public void SmolLM2_SingleUserMessage_InjectsDefaultSystemAndAssistantPrompt()
    {
        var msgs = new[] { ChatMessage.User("hello") };
        string result = Render(SmolLM2Template, msgs);

        // Golden (Python Jinja2):
        // '<|im_start|>system\nYou are a helpful AI assistant…<|im_end|>\n
        //  <|im_start|>user\nhello<|im_end|>\n
        //  <|im_start|>assistant\n'
        Assert.Contains("<|im_start|>system\nYou are a helpful AI assistant named SmolLM", result);
        Assert.Contains("<|im_start|>user\nhello<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    [Fact]
    public void SmolLM2_ExplicitSystemMessage_NoInjectedDefault()
    {
        var msgs = new[]
        {
            ChatMessage.System("Custom system."),
            ChatMessage.User("hello")
        };
        string result = Render(SmolLM2Template, msgs);

        Assert.DoesNotContain("SmolLM, trained by Hugging Face", result);
        Assert.Contains("<|im_start|>system\nCustom system.", result);
        Assert.Contains("<|im_start|>user\nhello<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    // ── Qwen2 ────────────────────────────────────────────────────────────

    [Fact]
    public void Qwen2_SingleUserMessage_MatchesGolden()
    {
        var msgs = new[] { ChatMessage.User("hello") };
        string result = Render(Qwen2Template, msgs);

        const string golden =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
            "<|im_start|>user\nhello<|im_end|>\n" +
            "<|im_start|>assistant\n";
        Assert.Equal(golden, result);
    }

    [Fact]
    public void Qwen2_MultiTurn_FormatsAllRoles()
    {
        var msgs = new[]
        {
            ChatMessage.System("Be concise."),
            ChatMessage.User("ping"),
            ChatMessage.Agent("pong")
        };
        string result = Render(Qwen2Template, msgs);

        Assert.Contains("<|im_start|>system\nBe concise.<|im_end|>", result);
        Assert.Contains("<|im_start|>user\nping<|im_end|>", result);
        Assert.Contains("<|im_start|>assistant\npong<|im_end|>", result);
        Assert.EndsWith("<|im_start|>assistant\n", result);
    }

    // ── TinyLlama / Zephyr ───────────────────────────────────────────────

    [Fact]
    public void TinyLlama_UserMessage_UsesZephyrTokens()
    {
        var msgs = new[] { ChatMessage.User("hello") };
        string result = Render(TinyLlamaTemplate, msgs);

        Assert.Contains("<|user|>\nhello</s>", result);
        Assert.Contains("<|assistant|>", result);
    }

    [Fact]
    public void TinyLlama_SystemAndUser_BothFormatted()
    {
        var msgs = new[]
        {
            ChatMessage.System("You are helpful."),
            ChatMessage.User("hi")
        };
        string result = Render(TinyLlamaTemplate, msgs);

        Assert.Contains("<|system|>\nYou are helpful.</s>", result);
        Assert.Contains("<|user|>\nhi</s>", result);
    }

    // ── Llama 3.x ────────────────────────────────────────────────────────

    [Fact]
    public void Llama3_SingleUserMessage_MatchesGolden()
    {
        var msgs = new[] { ChatMessage.User("hello") };
        // addBos=true so bos_token="<s>" is prepended
        string result = Render(Llama3Template, msgs, addBos: true);

        const string golden =
            "<s><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
        Assert.Equal(golden, result);
    }

    [Fact]
    public void Llama3_MultiTurn_AllMessagesFormatted()
    {
        var msgs = new[]
        {
            ChatMessage.System("Be helpful."),
            ChatMessage.User("question"),
            ChatMessage.Agent("answer")
        };
        string result = Render(Llama3Template, msgs, addBos: true);

        Assert.Contains("<|start_header_id|>system<|end_header_id|>\n\nBe helpful.<|eot_id|>", result);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nquestion<|eot_id|>", result);
        Assert.Contains("<|start_header_id|>assistant<|end_header_id|>\n\nanswer<|eot_id|>", result);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", result);
    }

    // ── Factory ──────────────────────────────────────────────────────────

    [Fact]
    public void Factory_WithTemplate_ReturnsJinjaFormatter()
    {
        var fmt = ChatPromptFormatterFactory.Create(Qwen2Template);
        Assert.IsType<JinjaTemplateFormatter>(fmt);
    }

    [Fact]
    public void Factory_NullTemplate_ReturnsFallback()
    {
        var fmt = ChatPromptFormatterFactory.Create((string?)null);
        Assert.IsType<SimpleFormatter>(fmt);
    }

    [Fact]
    public void Factory_EmptyTemplate_ReturnsFallback()
    {
        var fmt = ChatPromptFormatterFactory.Create("");
        Assert.IsType<SimpleFormatter>(fmt);
    }

    // ── Jinja expression evaluator edge cases ────────────────────────────

    [Fact]
    public void Eval_StringConcat_Works()
    {
        var env = new JinjaEnv();
        env.Set("a", (object)"Hello");
        env.Set("b", (object)" World");
        var result = JinjaTemplateFormatter.Eval("a + b", env);
        Assert.Equal("Hello World", result?.ToString());
    }

    [Fact]
    public void Eval_IsNone_True()
    {
        var env = new JinjaEnv();
        env.Set("x", null);
        Assert.True(JinjaTemplateFormatter.IsTruthy(JinjaTemplateFormatter.Eval("x is none", env)));
    }

    [Fact]
    public void Eval_IsNotNone_True()
    {
        var env = new JinjaEnv();
        env.Set("x", (object)"value");
        Assert.True(JinjaTemplateFormatter.IsTruthy(JinjaTemplateFormatter.Eval("x is not none", env)));
    }

    [Fact]
    public void Eval_InOperator_SubstringFound()
    {
        var env = new JinjaEnv();
        env.Set("content", (object)"hello </think> world");
        var result = JinjaTemplateFormatter.Eval("'</think>' in content", env);
        Assert.True(JinjaTemplateFormatter.IsTruthy(result));
    }

    [Fact]
    public void Eval_SplitLastSegment_Works()
    {
        var env = new JinjaEnv();
        env.Set("content", (object)"before </think> after");
        var result = JinjaTemplateFormatter.Eval("content.split('</think>')[-1]", env);
        Assert.Equal(" after", result?.ToString());
    }

    [Fact]
    public void Eval_Namespace_MutationAcrossScope()
    {
        const string tmpl =
            "{% set ns = namespace(val='a') %}" +
            "{% for message in messages %}" +
            "{% set ns.val = message['content'] %}" +
            "{% endfor %}" +
            "{{ ns.val }}";

        var msgs = new[] { ChatMessage.User("final") };
        var fmt = new JinjaTemplateFormatter(tmpl);
        string result = fmt.Format(msgs, Tok, false);
        Assert.Equal("final", result);
    }
}
