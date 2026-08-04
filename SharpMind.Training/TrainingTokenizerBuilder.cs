using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Tokenization;
using System.Text.Json.Nodes;

namespace SharpMind.Training;

/// <summary>
/// Builds a SharpMind <see cref="Tokenizer"/> for a freshly trained model whose
/// vocab size is fixed at construction time.
///
/// The four specials (&lt;unk&gt;, &lt;s&gt;, &lt;/s&gt;, &lt;pad&gt;) are placed at the
/// end of the model's vocabulary, <em>inside</em> <c>[0, vocabSize)</c>, so every
/// token id the tokenizer can emit maps to a valid embedding row. Filler tokens
/// fill the gap between the generator's words and the specials so decode is
/// well-defined for any greedy argmax.
/// </summary>
public static class TrainingTokenizerBuilder
{
    /// <summary>
    /// Builds a tokenizer from an ordered word list and a model vocab size.
    /// </summary>
    /// <param name="words">Words in generator-ID order.</param>
    /// <param name="vocabSize">Model vocabulary size — must be at least <c>words.Count + 4</c>.</param>
    /// <param name="unknownTokens">Optional naming hook for the filler rows (defaults to <c>"&lt;unk&gt;"</c>).</param>
    public static Tokenizer BuildForVocab(IReadOnlyList<string> words, int vocabSize, Func<int, string>? unknownTokens = null)
    {
        ArgumentNullException.ThrowIfNull(words);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        if (words.Count + 4 > vocabSize)
            throw new ArgumentException(
                $"Vocabulary of {words.Count} words plus 4 specials exceeds model vocab size {vocabSize}. " +
                "Increase the model VocabSize or reduce the word list.");

        var unknown = unknownTokens ?? (_ => "<unk>");
        var vocabObj = new JsonObject();
        for (int i = 0; i < words.Count; i++)
            vocabObj[words[i]] = i;

        // Filler rows between the words and the specials.
        for (int i = words.Count; i < vocabSize - 4; i++)
            vocabObj[unknown(i)] = i;

        vocabObj["<unk>"] = vocabSize - 4;
        vocabObj["<s>"] = vocabSize - 3;
        vocabObj["</s>"] = vocabSize - 2;
        vocabObj["<pad>"] = vocabSize - 1;

        var root = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokeniser"] = "whitespace",
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = "<unk>",
                ["bos"] = "<s>",
                ["eos"] = "</s>",
                ["pad"] = "<pad>",
                ["additional"] = new JsonArray(),
            },
            ["vocab"] = vocabObj,
            ["merges"] = new JsonArray(),
        };
        return Tokenizer.FromJson(root.ToJsonString());
    }

    /// <summary>Builds a tokenizer covering a <see cref="LearnableGenerator"/>'s vocabulary.</summary>
    public static Tokenizer BuildForVocab(LearnableGenerator generator, int vocabSize, Func<int, string>? unknownTokens = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        return BuildForVocab(generator.Vocabulary, vocabSize, unknownTokens);
    }
}
