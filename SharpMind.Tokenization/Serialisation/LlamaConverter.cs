using System.Text.Json;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Serialisation;

/// <summary>
/// Converts LLaMA 2 / LLaMA 3 tokenizer files to a SharpMind <see cref="BpeModel"/>.
///
/// Both LLaMA 2 and LLaMA 3 distribute as a HuggingFace <c>tokenizer.json</c>
/// with a BPE model block. The converter handles the differences between them:
///
/// LLaMA 2 special tokens (SentencePiece convention):
///   0 = &lt;unk&gt;   1 = &lt;s&gt; (BOS)   2 = &lt;/s&gt; (EOS)
///
/// LLaMA 3 special tokens (extended set):
///   128000 = &lt;|begin_of_text|&gt;   128001 = &lt;|end_of_text|&gt;
///   Plus many additional reserved tokens.
///
/// Files available from:
///   https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json
///   https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/tokenizer.json
/// </summary>
public static class LlamaConverter
{
    /// <param name="tokenizerJsonPath">Path to HuggingFace <c>tokenizer.json</c>.</param>
    public static BpeModel Convert(string tokenizerJsonPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"LLaMA tokenizer file not found: {tokenizerJsonPath}");

        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root      = doc.RootElement;

        AssertBpeModel(root, tokenizerJsonPath);

        var (unk, bos, eos, pad, additional) = ExtractSpecials(root);
        var specials = new SpecialTokens(unk, bos, eos, pad, additional);

        var vocab = BuildVocab(root, specials);
        var merges = BuildMerges(root);

        // Llama 3+ uses the cl100k pattern; Llama 2 is the classic GPT-2 pattern.
        bool llama3 = UsesCl100k(root);
        return new BpeModel(vocab, merges, llama3 ? new Cl100kPreTokeniser() : new Gpt2PreTokeniser());
    }

    private static bool UsesCl100k(JsonElement root)
    {
        if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
        {
            foreach (var token in added.EnumerateArray())
            {
                if (token.TryGetProperty("content", out var content) &&
                    content.GetString() is "<|begin_of_text|>" or "<|start_header_id|>")
                    return true;
            }
        }

        if (root.TryGetProperty("model", out var model) &&
            model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Object)
        {
            foreach (var token in vocab.EnumerateObject())
                if (token.Name is "<|begin_of_text|>" or "<|start_header_id|>")
                    return true;
        }

        return false;
    }

    // Helpers
    private static void AssertBpeModel(JsonElement root, string path)
    {
        if (root.TryGetProperty("model", out var model) &&
            model.TryGetProperty("type", out var type) &&
            !string.Equals(type.GetString(), "BPE", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException(
                $"Expected BPE model in {path}, got: {type.GetString()}");
    }

    private static (string unk, string bos, string eos, string pad,
                    IReadOnlyList<string> additional)
        ExtractSpecials(JsonElement root)
    {
        string unk = SpecialTokens.DefaultUnk;
        string bos = SpecialTokens.DefaultBos;
        string eos = SpecialTokens.DefaultEos;
        string pad = SpecialTokens.DefaultPad;
        var    additional = new List<string>();

        if (!root.TryGetProperty("added_tokens", out var tokens))
            return (unk, bos, eos, pad, additional);

        foreach (var t in tokens.EnumerateArray())
        {
            string? content = t.TryGetProperty("content", out var c) ? c.GetString() : null;
            bool    special = t.TryGetProperty("special", out var s) && s.GetBoolean();
            if (content is null || !special) continue;

            // Match both LLaMA 2 SentencePiece format, LLaMA 3 format, and Qwen format
            if (content is "<unk>" or "[UNK]" or "<|unk|>") unk = content;
            else if (content is "<s>" or "[BOS]" or "<|begin_of_text|>" or "<|im_start|>") bos = content;
            else if (content is "</s>" or "[EOS]" or "<|end_of_text|>" or "<|im_end|>") eos = content;
            else if (content is "<pad>" or "[PAD]" or "<|endoftext|>") pad = content;
            else additional.Add(content);
        }

        return (unk, bos, eos, pad, additional);
    }

    private static Vocabulary BuildVocab(JsonElement root, SpecialTokens specials)
    {
        var ordered = root.GetProperty("model")
                          .GetProperty("vocab")
                          .EnumerateObject()
                          .OrderBy(p => p.Value.GetInt32())
                          .Select(p => p.Name)
                          .ToList();

        // Add special tokens to vocabulary if they're not already there
        foreach (string token in specials.All)
        {
            if (!ordered.Contains(token))
                ordered.Add(token);
        }

        return new Vocabulary(ordered, specials);
    }

    private static List<MergeRule> BuildMerges(JsonElement root)
        => [.. root.GetProperty("model")
               .GetProperty("merges")
               .EnumerateArray()
               .Select((el, rank) =>
               {
                   string[] parts = el.GetString()!.Split(' ', 2);
                   return new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank);
               })];
}
