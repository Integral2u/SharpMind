namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class VocabularyBuilder(VocabConfig config, Random? random = null)
{
    private readonly MorphemeDictionary _morphemes = new(config.VocabSize);
    private readonly VocabConfig _config = config;
    private readonly Random _random = random ?? Random.Shared;
    private readonly HashSet<string> _usedWords = [];
    private readonly List<PseudoWord> _words = new(config.VocabSize);

    public IReadOnlyList<PseudoWord> Words => _words;

    public VocabularyBuilder Build()
    {
        foreach (var root in _morphemes.Roots)
        {
            if (_words.Count >= _config.VocabSize) break;
            var family = new List<(string, MorphemeCategory)>
            {
                (root, MorphemeCategory.Root)
            };
            AddWord(root, MorphemeCategory.Root, family);
        }

        GenerateDerivedWords();
        AddNegations();

        if (_words.Count >= _config.VocabSize) return this;
        if (_config.Affixes <= 0 || _morphemes.Roots.Count == 0)
            throw new InvalidOperationException(
                $"VocabularyBuilder cannot reach {_config.VocabSize} words with Affixes={_config.Affixes}.");

        var suffixes = _morphemes.GetSuffixes().ToList();
        if (suffixes.Count == 0)
            throw new InvalidOperationException(
                $"VocabularyBuilder cannot reach {_config.VocabSize} words: no suffixes are configured.");

        // Random retries can revisit the same root+suffix forever once the finite
        // word space is exhausted. Bound the attempts so a misconfigured size
        // fails loudly instead of hanging the build.
        long maxAttempts = Math.Max(1_000L, (long)_morphemes.Roots.Count * suffixes.Count * 4);
        long attempts = 0;
        while (_words.Count < _config.VocabSize && attempts++ < maxAttempts)
        {
            var rootIndex = _random.Next(_morphemes.Roots.Count);
            var root = _morphemes.Roots[rootIndex];
            var suffix = suffixes[_random.Next(suffixes.Count)];
            var word = root + suffix;
            var family = new List<(string, MorphemeCategory)>
            {
                (root, MorphemeCategory.Root)
            };
            _morphemes.TryGetSuffixCategory(suffix, out var cat);
            AddWord(word, cat, family);
        }

        if (_words.Count < _config.VocabSize)
            throw new InvalidOperationException(
                $"VocabularyBuilder exhausted {_morphemes.Roots.Count} roots x {suffixes.Count} suffixes " +
                $"after {attempts} attempts and only produced {_words.Count}/{_config.VocabSize} words.");

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
                var family = new List<(string, MorphemeCategory)>
                {
                    (root, MorphemeCategory.Root)
                };
                _morphemes.TryGetSuffixCategory(suffix, out var cat);
                AddWord(word, cat, family);
            }

            var commonPrefixes = new List<string> { "un", "re", "pre", "over", "under", "dis" };
            foreach (var prefix in commonPrefixes)
            {
                if (_words.Count >= _config.VocabSize) break;
                var word = prefix + root;
                var family = new List<(string, MorphemeCategory)>
                {
                    (root, MorphemeCategory.Root)
                };
                _morphemes.TryGetPrefixCategory(prefix, out var cat);
                AddWord(word, cat, family);
            }

            if (_config.Affixes >= 5)
            {
                var combos = new List<string> { "un" + root + "able", "re" + root + "ed", "over" + root + "er" };
                foreach (var word in combos)
                {
                    if (_words.Count >= _config.VocabSize) break;
                    var family = new List<(string, MorphemeCategory)>
                    {
                        (root, MorphemeCategory.Root)
                    };
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
                var family = new List<(string, MorphemeCategory)>
                {
                    (wordItem.Text, wordItem.BaseCategory)
                };
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
            WordFamily = [.. family],
        };
        _words.Add(pseudoWord);
    }
}
