using System.Text.Json;
using SharpMind.Tokenizer.Bpe;
using SharpMind.Tokenizer.PreTokeniser;
using SharpMind.Tokenizer.Vocab;

namespace SharpMind.Tokenizer.Serialisation;

/// <summary>
/// Converts Mistral tokenizer files to a SharpMind <see cref="BpeModel"/>.
///
/// Mistral 7B v0.1 uses the same vocabulary as LLaMA 2 (32,000 tokens, same
/// merge rules, same SentencePiece-derived special tokens). Later Mistral
/// models (v0.3+, Mixtral) extend the vocabulary with additional tokens.
///
/// The converter detects the vocab size and routes accordingly:
///   32,000 tokens → LLaMA 2 compatible, use <see cref="LlamaConverter"/>
///   32,768+ tokens → extended Mistral vocab with instruction tokens
///
/// Files available from:
///   https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.json
///   https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.json
/// </summary>
public static class MistralConverter
{
    // Mistral v0.3+ adds [INST], [/INST], [AVAILABLE_TOOLS] etc.
    private static readonly HashSet<string> KnownInstructTokens = new(StringComparer.Ordinal)
    {
        "[INST]", "[/INST]", "[TOOL_CALLS]", "[AVAILABLE_TOOLS]",
        "[TOOL_RESULTS]", "[/TOOL_RESULTS]",
    };

    /// <param name="tokenizerJsonPath">Path to HuggingFace <c>tokenizer.json</c>.</param>
    public static BpeModel Convert(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"Mistral tokenizer file not found: {tokenizerJsonPath}");

        using var doc  = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root       = doc.RootElement;
        int vocabSize  = root.GetProperty("model").GetProperty("vocab")
                             .EnumerateObject().Count();

        // v0.1 / v0.2 — identical to LLaMA 2 vocab; delegate directly
        if (vocabSize <= 32_000)
            return LlamaConverter.Convert(tokenizerJsonPath);

        // v0.3+ — extended vocab with instruction tokens
        return ConvertExtended(root);
    }

    private static BpeModel ConvertExtended(JsonElement root)
    {
        string unk = "<unk>";
        string bos = "<s>";
        string eos = "</s>";
        string pad = SpecialTokens.DefaultPad;
        var    additional = new List<string>();

        if (root.TryGetProperty("added_tokens", out var tokens))
        {
            foreach (var t in tokens.EnumerateArray())
            {
                string? content = t.TryGetProperty("content", out var c) ? c.GetString() : null;
                bool    special = t.TryGetProperty("special", out var s) && s.GetBoolean();
                if (content is null || !special) continue;

                if (content == "<unk>") unk = content;
                else if (content == "<s>") bos = content;
                else if (content == "</s>") eos = content;
                else if (content == "<pad>") pad = content;
                else if (KnownInstructTokens.Contains(content) || content.StartsWith('['))
                    additional.Add(content);
                else additional.Add(content);
            }
        }

        var specials = new SpecialTokens(unk, bos, eos, pad, additional);

        var ordered = root.GetProperty("model")
                          .GetProperty("vocab")
                          .EnumerateObject()
                          .OrderBy(p => p.Value.GetInt32())
                          .Select(p => p.Name)
                          .ToList();
        var vocab = new Vocabulary(ordered, specials);

        var merges = root.GetProperty("model")
                         .GetProperty("merges")
                         .EnumerateArray()
                         .Select((el, rank) =>
                         {
                             string[] parts = el.GetString()!.Split(' ', 2);
                             return new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank);
                         })
                         .ToList();

        return new BpeModel(vocab, merges, new Gpt2PreTokeniser());
    }
}
