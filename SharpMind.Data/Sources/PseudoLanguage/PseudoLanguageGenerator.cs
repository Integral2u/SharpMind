namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class PseudoLanguageGenerator : IDisposable
{
    private readonly VocabConfig _config;
    private readonly MorphemeDictionary _morphemes;
    private readonly VocabularyBuilder _vocabBuilder;
    private readonly Random _random;
    private bool _disposed;

    public IReadOnlyList<PseudoWord> Vocabulary => _vocabBuilder.Words;
    public VocabConfig Config => _config;
    public int VocabSize => _vocabBuilder.Words.Count;

    public PseudoLanguageGenerator(VocabConfig config, Random? random = null)
    {
        _config = config;
        _random = random ?? Random.Shared;
        _morphemes = new MorphemeDictionary();
        _vocabBuilder = new VocabularyBuilder(config, random).Build();
    }

    public IEnumerable<GeneratedSequence> GenerateSyntactic(int count, ComplexityLevel level)
    {
        return level switch
        {
            ComplexityLevel.Options => GenerateOptions(count),
            ComplexityLevel.Patterns => GeneratePatterns(count),
            ComplexityLevel.Syntactic => GenerateSyntacticSequences(count),
            _ => GenerateSyntacticSequences(count),
        };
    }

    public IEnumerable<GeneratedSequence> GenerateOptions(int count)
    {
        for (int i = 0; i < count; i++)
        {
            var subject = GetRandomWord([MorphemeCategory.Agent, MorphemeCategory.Root]);
            var options = new List<(string text, int id)>();

            var correctWord = GetRandomWordInFamily(subject.Text);
            options.Add((correctWord.Text, correctWord.TokenId));

            for (int j = 0; j < 3; j++)
            {
                var wrong = GetRandomWord();
                if (wrong.TokenId != subject.TokenId && !options.Any(o => o.id == wrong.TokenId))
                    options.Add((wrong.Text, wrong.TokenId));
            }

            var shuffle = options.OrderBy(_ => _random.Next()).ToList();
            var correctIndex = shuffle.FindIndex(o => o.text == correctWord.Text);

            yield return new GeneratedSequence
            {
                TokenIds = [subject.TokenId],
                RawText = subject.Text + ": " + string.Join(", ", shuffle.Select(o => o.text)),
                GroundTruthIds = [shuffle[correctIndex].id],
                GroundTruthText = shuffle[correctIndex].text,
                Complexity = ComplexityLevel.Options,
            };
        }
    }

    public IEnumerable<GeneratedSequence> GeneratePatterns(int count)
    {
        for (int i = 0; i < count; i++)
        {
            var root = GetRandomWord(MorphemeCategory.Root);
            var variants = new List<(string text, int id)>();

            var family = GetWordFamily(root.Text);
            foreach (var word in family.Take(4))
            {
                variants.Add((word.Text, word.TokenId));
            }

            if (variants.Count < 2) continue;

            var ordered = variants.OrderBy(_ => _random.Next()).ToList();
            var nextIndex = _random.Next(1, ordered.Count);
            var sequence = ordered.Take(nextIndex).Select(v => v.id).ToArray();
            var (text, id) = ordered[nextIndex];

            yield return new GeneratedSequence
            {
                TokenIds = sequence,
                RawText = string.Join(" ", sequence.Select(IdToText)),
                GroundTruthIds = [id],
                GroundTruthText = text,
                Complexity = ComplexityLevel.Patterns,
            };
        }
    }

    public IEnumerable<GeneratedSequence> GenerateSyntacticSequences(int count)
    {
        for (int i = 0; i < count; i++)
        {
            var subject = GetRandomWord([MorphemeCategory.Agent, MorphemeCategory.Root]);
            var verb = GetRandomWord([MorphemeCategory.Verb, MorphemeCategory.Root]);
            var obj = GetRandomWord([MorphemeCategory.Noun, MorphemeCategory.Adjective]);

            var tokenIds = new[] { subject.TokenId, verb.TokenId, obj.TokenId };
            var continuation = GetRandomWord(MorphemeCategory.Adjective);

            yield return new GeneratedSequence
            {
                TokenIds = tokenIds,
                RawText = $"{IdToText(subject.TokenId)} {IdToText(verb.TokenId)} {IdToText(obj.TokenId)}",
                GroundTruthIds = [continuation.TokenId],
                GroundTruthText = IdToText(continuation.TokenId),
                Complexity = ComplexityLevel.Syntactic,
            };
        }
    }

    private PseudoWord? GetRandomWord(MorphemeCategory? category = null)
    {
        var words = (category.HasValue
            ? _vocabBuilder.Words.Where(w => w.BaseCategory == category.Value)
            : _vocabBuilder.Words).ToList();

        if (words.Count == 0) return null;
        return words[_random.Next(words.Count)];
    }

    private PseudoWord? GetRandomWord(MorphemeCategory[] categories)
    {
        var words = _vocabBuilder.Words
            .Where(w => categories.Contains(w.BaseCategory))
            .ToList();
        if (words.Count == 0) return null;
        return words[_random.Next(words.Count)];
    }

    private PseudoWord? GetRandomWordInFamily(string baseText)
    {
        var family = GetWordFamily(baseText);
        if (family.Count == 0) return GetRandomWord();

        var filtered = family.Where(f => f.Text != baseText).ToList();
        return filtered.Count > 0
            ? filtered[_random.Next(filtered.Count)]
            : family[0];
    }

    private List<PseudoWord> GetWordFamily(string baseText)
    {
        var root = _vocabBuilder.Words.FirstOrDefault(w => w.Text == baseText);
        if (root == null) return [];

        var family = new List<PseudoWord> { root };
        foreach (var w in _vocabBuilder.Words)
        {
            if (w.WordFamily.Any(f => f.BaseWord == baseText))
                family.Add(w);
        }
        return family;
    }

    public string IdToText(int tokenId)
    {
        var word = _vocabBuilder.Words.FirstOrDefault(w => w.TokenId == tokenId);
        return word?.Text ?? $"<UNK:{tokenId}>";
    }

    public int TextToId(string text)
    {
        var word = _vocabBuilder.Words.FirstOrDefault(w => w.Text == text);
        return word?.TokenId ?? -1;
    }

    public ModelSizeRecommendation GetModelSizeRecommendation()
    {
        var vocab = VocabSize;

        return new ModelSizeRecommendation
        {
            VocabSize = vocab,
            EmbeddingDim = (int)Math.Ceiling(Math.Sqrt(vocab)),
            HiddenDim = vocab / 8,
            NumLayers = vocab switch
            {
                < 1000 => 4,
                < 5000 => 6,
                < 20000 => 8,
                < 50000 => 12,
                _ => 16,
            },
            HeadDim = 64,
            NumHeads = 8,
            FfnDim = vocab / 4,
            EstimatedParams = ComputeParameterCount(vocab),
        };
    }

    private static long ComputeParameterCount(int vocab)
    {
        var emb = vocab * (int)Math.Ceiling(Math.Sqrt(vocab));
        var ffn = (vocab / 8) * 4 * (vocab / 4);
        var attn = 3 * (vocab / 8) * (vocab / 8);
        var layers = vocab switch { < 1000 => 4, < 5000 => 6, < 20000 => 8, _ => 12 };
        return (emb + ffn + attn) * layers + vocab;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}
