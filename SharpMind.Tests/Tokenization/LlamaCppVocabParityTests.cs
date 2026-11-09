using SharpMind.Model.Format;
using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Token-id parity with llama.cpp on its own tokenizer fixtures (DataRef/llama-vocab, see README there):
/// every input must encode to exactly the ids llama.cpp's test-tokenizer-0 expects. Loads through the
/// real GGUF path (GgufLoader.LoadMeta -> LoadTokenizerFromMeta), so pre-tokenizer selection and the
/// merge loop are both covered.
/// </summary>
public sealed class LlamaCppVocabParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "DataRef", "llama-vocab");
    private static readonly Dictionary<string, Tokenizer> Tokenizers = [];

    public static IEnumerable<object[]> Cases()
    {
        foreach (var vocab in new[] { "qwen2", "llama-bpe", "gpt-2", "phi-3", "deepseek-coder", "deepseek-llm" })
        {
            // .inp ends with a trailing separator, so it splits into one more chunk than .out has lines.
            int count = File.ReadAllLines(Path.Combine(Dir, $"ggml-vocab-{vocab}.gguf.out")).Length;
            for (int i = 0; i < count; i++)
                yield return [vocab, i];
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Encode_MatchesLlamaCpp(string vocab, int index)
    {
        string path = Path.Combine(Dir, $"ggml-vocab-{vocab}.gguf");
        string input = File.ReadAllText(path + ".inp").Split("\n__ggml_vocab_test__\n")[index];
        int[] expected = [.. File.ReadAllLines(path + ".out")[index]
            .Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)];

        int[] actual = Load(vocab).Encode(input, addBos: false, addEos: false);

        Assert.True(expected.SequenceEqual(actual),
            $"{vocab} #{index} {System.Text.Json.JsonSerializer.Serialize(input)}\n" +
            $"  llama.cpp [{string.Join(" ", expected)}]\n  sharpmind [{string.Join(" ", actual)}]");
    }

    [Theory]
    [InlineData("qwen2", "qwen2")]
    [InlineData("llama-bpe", "llama-bpe")]
    [InlineData("gpt-2", "gpt-2")]
    [InlineData("phi-3", "default")]
    [InlineData("deepseek-coder", "deepseek-coder")]
    [InlineData("deepseek-llm", "deepseek-llm")]
    public void Loader_PassesTokenizerGgmlPre(string vocab, string pre)
        // The fixtures' architectures would pick the same patterns on their own, so this is what
        // proves GgufLoader hands tokenizer.ggml.pre to the tokenizer at all.
        => Assert.Equal(pre, Load(vocab).GgufPreTokenizer);

    private static Tokenizer Load(string vocab)
    {
        lock (Tokenizers)
        {
            if (!Tokenizers.TryGetValue(vocab, out var tok))
            {
                var meta = GgufLoader.LoadMeta(Path.Combine(Dir, $"ggml-vocab-{vocab}.gguf"));
                tok = GgufLoader.LoadTokenizerFromMeta(meta)
                      ?? throw new InvalidOperationException($"no tokenizer in {vocab} fixture");
                Tokenizers[vocab] = tok;
            }
            return tok;
        }
    }
}
