namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class VocabularyBuilder
{
    private readonly MorphemeDictionary _morphemes;
    private readonly VocabConfig _config;
    private readonly Random _random;
    private readonly HashSet<string> _usedWords;
    private readonly List<PseudoWord> _words;

    public IReadOnlyList<PseudoWord> Words => _words;

    public VocabularyBuilder(VocabConfig config, Random? random = null)
    {
        _config = config;
        _random = random ?? Random.Shared;
        _morphemes = new MorphemeDictionary();
        _usedWords = new HashSet<string>();
        _words = new List<PseudoWord>(config.VocabSize);
    }

    public VocabularyBuilder Build()
    {
        foreach (var root in _morphemes.Roots)
        {
            if (_words.Count >= _config.VocabSize) break;
            var family = new List<(string, MorphemeCategory)>();
            family.Add((root, MorphemeCategory.Root));
            AddWord(root, MorphemeCategory.Root, family);
        }

        GenerateDerivedWords();
        AddNegations();

        while (_words.Count < _config.VocabSize)
        {
            var rootIndex = _random.Next(_morphemes.Roots.Count);
            var root = _morphemes.Roots[rootIndex];
            var suffixes = _morphemes.GetSuffixes().ToList();
            var suffix = _config.Affixes > 0 && suffixes.Count > 0
                ? suffixes[_random.Next(suffixes.Count)]
                : "";
            if (!string.IsNullOrEmpty(suffix))
            {
                var word = root + suffix;
                var family = new List<(string, MorphemeCategory)>();
                family.Add((root, MorphemeCategory.Root));
                _morphemes.TryGetSuffixCategory(suffix, out var cat);
                AddWord(word, cat, family);
            }
        }

        return this;
    }

    private void GenerateDerivedWords()
    {
        foreach (var root in _morphemes.Roots)
        {
            if (_words.Count >= _config.VocabSize) break;

            var commonSuffixes = new List<string> { "er", "ed", "ing", "able", "ful", "less", "ly" };
            foreach (var suffix in commonSuffixes)
            {
                if (_words.Count >= _config.VocabSize) break;
                var word = root + suffix;
                var family = new List<(string, MorphemeCategory)>();
                family.Add((root, MorphemeCategory.Root));
                _morphemes.TryGetSuffixCategory(suffix, out var cat);
                AddWord(word, cat, family);
            }

            var commonPrefixes = new List<string> { "un", "re", "pre", "over", "under", "dis" };
            foreach (var prefix in commonPrefixes)
            {
                if (_words.Count >= _config.VocabSize) break;
                var word = prefix + root;
                var family = new List<(string, MorphemeCategory)>();
                family.Add((root, MorphemeCategory.Root));
                _morphemes.TryGetPrefixCategory(prefix, out var cat);
                AddWord(word, cat, family);
            }

            if (_config.Affixes >= 5)
            {
                var combos = new List<string> { "un" + root + "able", "re" + root + "ed", "over" + root + "er" };
                foreach (var word in combos)
                {
                    if (_words.Count >= _config.VocabSize) break;
                    var family = new List<(string, MorphemeCategory)>();
                    family.Add((root, MorphemeCategory.Root));
                    AddWord(word, MorphemeCategory.Negation, family);
                }
            }
        }
    }

    private void AddNegations()
    {
        var negPrefixes = new List<string> { "un", "non", "im", "dis" };
        var halfCount = _words.Count / 2;
        for (int i = 0; i < halfCount; i++)
        {
            var wordItem = _words[i];
            foreach (var prefix in negPrefixes)
            {
                if (_words.Count >= _config.VocabSize) break;
                var negWord = prefix + wordItem.Text;
                var family = new List<(string, MorphemeCategory)>();
                family.Add((wordItem.Text, wordItem.BaseCategory));
                AddWord(negWord, MorphemeCategory.Negation, family);
            }
        }
    }

    private void AddWord(string text, MorphemeCategory category, List<(string, MorphemeCategory)> family)
    {
        if (string.IsNullOrEmpty(text)) return;
        if (text.Length < _config.MinWordLength || text.Length > _config.MaxWordLength) return;
        bool added = _usedWords.Add(text);
        if (!added) return;

        var pseudoWord = new PseudoWord
        {
            Text = text,
            TokenId = _words.Count,
            BaseCategory = category,
            WordFamily = family.ToArray(),
        };
        _words.Add(pseudoWord);
    }
}
