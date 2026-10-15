using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.Chat;

/// <summary>
/// End-to-end regression tests for the Gemma 4 chat template (the exact
/// <c>tokenizer.chat_template</c> shipped in the GGUF). Golden strings are the
/// byte-for-byte output of Python Jinja2 rendering the same template.
/// </summary>
public sealed class Gemma4ChatTemplateTests
{
    private static readonly Tokenizer Tok = Tokenizer.FromGguf(
        tokens: ["[UNK]", "<bos>", "<eos>"],
        merges: null,
        tokenTypes: null,
        bosId: 1,
        eosId: 2);

    private static string Template()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "DataRef", "gemma4_chat_template.jinja");
        Assert.True(File.Exists(path), $"missing test asset: {path}");
        return File.ReadAllText(path);
    }

    [Fact]
    public void SingleUserTurn_MatchesJinja2()
    {
        var fmt = new JinjaTemplateFormatter(Template());
        string result = fmt.Format(new[] { ChatMessage.User("What is the capital of France?") }, Tok, addBos: true);

        const string golden = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";
        Assert.Equal(golden, result);
        Assert.Null(fmt.LastErrors);
    }

    [Fact]
    public void MultiTurn_MatchesJinja2()
    {
        var fmt = new JinjaTemplateFormatter(Template());
        string result = fmt.Format(new[]
        {
            ChatMessage.System("You are a helpful assistant."),
            ChatMessage.User("Hi"),
            ChatMessage.Agent("Hello! How can I help?"),
            ChatMessage.User("What is 2+2?")
        }, Tok, addBos: true);

        const string golden =
            "<bos><|turn>system\nYou are a helpful assistant.<turn|>\n" +
            "<|turn>user\nHi<turn|>\n" +
            "<|turn>model\nHello! How can I help?<turn|>\n" +
            "<|turn>user\nWhat is 2+2?<turn|>\n" +
            "<|turn>model\n";
        Assert.Equal(golden, result);
        Assert.Null(fmt.LastErrors);
    }

    [Fact]
    public void ToolDeclaration_MatchesJinja2()
    {
        const string toolsJson =
            """
            [{"type":"function","function":{"name":"get_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city"},"unit":{"type":"string","enum":["c","f"],"description":"Unit"}},"required":["location"]}}}]
            """;

        var fmt = new JinjaTemplateFormatter(Template());
        string result = fmt.Format(new[] { ChatMessage.User("What's the weather in Paris?") },
            Tok, addBos: true, enableThinking: false, toolsJson: toolsJson);

        const string golden =
            "<bos><|turn>system\n" +
            "<|tool>declaration:get_weather{description:<|\"|>Get the current weather<|\"|>," +
            "parameters:{properties:{location:{description:<|\"|>The city<|\"|>,type:<|\"|>STRING<|\"|>}," +
            "unit:{description:<|\"|>Unit<|\"|>,enum:[<|\"|>c<|\"|>,<|\"|>f<|\"|>],type:<|\"|>STRING<|\"|>}}," +
            "required:[<|\"|>location<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n" +
            "<|turn>user\nWhat's the weather in Paris?<turn|>\n" +
            "<|turn>model\n";
        Assert.Equal(golden, result);
        Assert.Null(fmt.LastErrors);
    }
}
