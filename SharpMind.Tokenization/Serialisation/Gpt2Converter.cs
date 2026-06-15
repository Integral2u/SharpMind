using System.Text.Json;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Serialisation;

/// <summary>
/// Converts GPT-2 tokenizer files to a SharpMind <see cref="BpeModel"/>.
///
/// GPT-2 distributes its tokenizer as two separate files:
///   <c>encoder.json</c>  — token → id mapping
///   <c>vocab.bpe</c>     — BPE merge rules, one per line as "left right"
///
/// These files are available from:
///   https://huggingface.co/gpt2/resolve/main/encoder.json
///   https://huggingface.co/gpt2/resolve/main/vocab.bpe
///
/// GPT-2 special tokens:
///   50256 = &lt;|endoftext|&gt; — used as both BOS and EOS
///   No explicit PAD or UNK — <see cref="SpecialTokens.DefaultPad"/> and
///   <see cref="SpecialTokens.DefaultUnk"/> are added as SharpMind conventions.
/// </summary>
public static class Gpt2Converter
{
    /// <param name="encoderJsonPath">Path to <c>encoder.json</c>.</param>
    /// <param name="vocabBpePath">Path to <c>vocab.bpe</c>.</param>
    public static BpeModel Convert(string encoderJsonPath, string vocabBpePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(encoderJsonPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(vocabBpePath);

        // Vocab
        using var doc = JsonDocument.Parse(File.ReadAllText(encoderJsonPath));
        var ordered   = doc.RootElement
                           .EnumerateObject()
                           .OrderBy(p => p.Value.GetInt32())
                           .Select(p => p.Name)
                           .ToList();

        // GPT-2's only special token
        const string eot = "<|endoftext|>";
        var specials = new SpecialTokens(new SpecialTokensConfig
        {
            Unk        = SpecialTokens.DefaultUnk,
            Bos        = eot,
            Eos        = eot,
            Pad        = SpecialTokens.DefaultPad,
            Additional = [],
        });

        // Build vocab from the ordered token list — GPT-2's vocab already
        // contains its special token at id 50256 so we use it as-is
        var vocab = new Vocabulary(ordered, specials);

        // Merges
        // vocab.bpe format:
        //   Line 0: "#version: 0.2"  (header — skip)
        //   Lines 1+: "Ġhello Ġworld"  (one merge per line, rank = line index - 1)
        var merges = File.ReadLines(vocabBpePath)
                         .Skip(1)               // skip version header
                         .Where(l => l.Contains(' '))
                         .Select((line, rank) =>
                         {
                             string[] parts = line.Split(' ', 2);
                             return new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank);
                         })
                         .ToList();

        return new BpeModel(vocab, merges, new Gpt2PreTokeniser());
    }
}
