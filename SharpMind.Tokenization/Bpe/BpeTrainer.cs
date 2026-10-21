using System.Collections.Concurrent;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Bpe;

/// <summary>
/// Trains a BPE vocabulary from a text corpus.
/// Optimized with parallel processing and efficient merge tracking.
/// </summary>
public sealed class BpeTrainer
{
    private readonly int _targetVocabSize;
    private readonly int _minFrequency;
    private readonly IPreTokeniser _preTokeniser;
    private readonly SpecialTokens _specials;
    private readonly bool _byteLevel;

    public BpeTrainer(
        int targetVocabSize,
        int minFrequency = 2,
        IPreTokeniser? preTokeniser = null,
        SpecialTokens? specials = null,
        bool byteLevel = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetVocabSize);
        _targetVocabSize = targetVocabSize;
        _minFrequency = minFrequency;
        _preTokeniser = preTokeniser ?? new Gpt2PreTokeniser();
        _specials = specials ?? new SpecialTokens();
        _byteLevel = byteLevel;
    }

    public async Task<BpeModel> TrainAsync(
        IAsyncEnumerable<string> documents,
        CancellationToken cancellationToken = default)
    {
        var wordFreqs = await CountWordsAsync(documents, cancellationToken);
        return RunBpe(wordFreqs);
    }

    /// <summary>
    /// Grows an existing BPE model by continuing its greedy merge process over a
    /// (possibly larger) corpus, appending only NEW merges up to
    /// <paramref name="newTargetVocabSize"/>.
    ///
    /// The existing model's base vocabulary (specials + byte tokens) and merge
    /// rules are preserved verbatim — their IDs are NOT renumbered — so a model
    /// trained against the old vocabulary stays valid: every old token keeps its
    /// ID and new tokens occupy the previously-unused IDs above the old vocab
    /// size. This is what makes it safe to expand a checkpoint's embedding table
    /// (old rows keep their meaning, only new rows are added).
    ///
    /// Returns the grown model, or the original when no growth is possible
    /// (already at/above the target, or no merge clears <c>minFrequency</c>).
    /// </summary>
    public async Task<BpeModel> GrowAsync(
        BpeModel existing,
        IAsyncEnumerable<string> documents,
        int newTargetVocabSize,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(existing);
        if (newTargetVocabSize <= existing.Vocab.Size) return existing;
        if (existing.IsCharMode)
            throw new InvalidOperationException(
                "Cannot grow a character-level tokenizer — it has no merge rules.");

        var wordFreqs = await CountWordsAsync(documents, cancellationToken);

        // Start from an exact copy of the existing vocabulary. Its layout (specials,
        // optional byte tokens, then learned merges) is preserved verbatim, so every
        // old token keeps its ID no matter how this trainer was configured — growing
        // a non-byte-level tokenizer with a default (byte-level) trainer must not
        // inject 256 tokens ahead of the old ones and renumber them.
        var vocab = new Vocabulary(existing.Vocab.AllTokens, existing.Vocab.Specials);
        var merges = new List<MergeRule>();

        var wordTokens = BuildInitialWordTokens(wordFreqs);

        // Replay the existing merges in rank order. This must reproduce the old
        // segmentation exactly; if a rule references a token the base vocab can't
        // hold, abort rather than corrupt the mapping.
        foreach (var rule in existing.Merges.OrderBy(m => m.Rank))
        {
            if (!vocab.Contains(rule.Merged))
                throw new InvalidOperationException(
                    $"Cannot grow tokenizer: existing merge '{rule.Left} {rule.Right}' " +
                    $"produces unknown token '{rule.Merged}'. The tokenizer is not growable.");
            merges.Add(new MergeRule(rule.Left, rule.Right, rule.Merged, rule.Rank));
            ApplyMergeParallel(wordTokens, rule.Left, rule.Right, rule.Merged);
        }

        // Continue the greedy loop, appending new merges only. Loop on actual size
        // (not a fixed merge count) so a pair that resolves to an already-known
        // token does not eat into the growth budget.
        while (vocab.Size < newTargetVocabSize)
        {
            var pairCounts = CountPairsParallel(wordTokens, wordFreqs);
            if (pairCounts.Count == 0) break;

            var (left, right, count) = FindBestPair(pairCounts);
            if (count < _minFrequency) break;

            string merged = left + right;
            // A pair whose merged token already exists in the (loaded) vocab must
            // not be re-added; resolve the pair and try the next-best one instead
            // of stalling.
            if (vocab.Contains(merged))
            {
                ApplyMergeParallel(wordTokens, left, right, merged);
                continue;
            }

            vocab.AddToken(merged);
            merges.Add(new MergeRule(left, right, merged, merges.Count));
            ApplyMergeParallel(wordTokens, left, right, merged);
        }

        return new BpeModel(vocab, merges, existing.PreTokeniser);
    }

    /// <summary>
    /// Runs the greedy BPE merge process over the FULL corpus and reports how
    /// many tokens each additional merge saves, so a caller can pick the smallest
    /// vocabulary that captures the useful compression. Nothing is persisted; the
    /// returned analysis is advisory.
    /// </summary>
    public async Task<BpeGrowthAnalysis> AnalyseAsync(
        IAsyncEnumerable<string> documents,
        int maxProbeVocabSize,
        CancellationToken cancellationToken = default)
    {
        var wordFreqs = await CountWordsAsync(documents, cancellationToken);

        var vocab = new Vocabulary(_specials, addByteTokens: _byteLevel);
        var wordTokens = BuildInitialWordTokens(wordFreqs);

        int totalWordOccurrences = wordFreqs.Values.Sum();

        long TokensAtBaseline()
        {
            long n = 0;
            foreach (var (word, tokens) in wordTokens)
                n += (long)tokens.Count * wordFreqs[word];
            return n;
        }

        long tokensNow = TokensAtBaseline();
        long baseline = tokensNow;

        var steps = new List<MergeSaving>(Math.Max(0, maxProbeVocabSize - vocab.Size));
        int mergesNeeded = maxProbeVocabSize - vocab.Size;
        for (int step = 0; step < mergesNeeded; step++)
        {
            var pairCounts = CountPairsParallel(wordTokens, wordFreqs);
            if (pairCounts.Count == 0) break;

            var (left, right, count) = FindBestPair(pairCounts);
            if (count < _minFrequency) break;

            string merged = left + right;
            if (vocab.Contains(merged))
            {
                ApplyMergeParallel(wordTokens, left, right, merged);
                continue;
            }

            vocab.AddToken(merged);
            ApplyMergeParallel(wordTokens, left, right, merged);

            long tokensAfter = TokensAtBaseline();
            long saved = tokensNow - tokensAfter;
            tokensNow = tokensAfter;
            steps.Add(new MergeSaving(vocab.Size - 1, merged, count, saved, tokensNow));
        }

        return new BpeGrowthAnalysis(
            baseVocabSize: vocab.Size - steps.Count,
            finalVocabSize: vocab.Size,
            totalWordOccurrences: totalWordOccurrences,
            baselineTokens: baseline,
            finalTokens: tokensNow,
            steps: steps);
    }

    private BpeModel RunBpe(Dictionary<string, int> wordFreqs)
    {
        var vocab = new Vocabulary(_specials, addByteTokens: _byteLevel);
        var merges = new List<MergeRule>();

        // Represent each word as a list of tokens. 
        // Using a simple list here; for massive datasets, a linked structure is better.
        var wordTokens = BuildInitialWordTokens(wordFreqs);

        int mergesNeeded = _targetVocabSize - vocab.Size;
        if (mergesNeeded <= 0)
            return new BpeModel(vocab, merges, _preTokeniser);

        for (int step = 0; step < mergesNeeded; step++)
        {
            var pairCounts = CountPairsParallel(wordTokens, wordFreqs);
            if (pairCounts.Count == 0) break;

            var (left, right, count) = FindBestPair(pairCounts);
            if (count < _minFrequency) break;

            string merged = left + right;
            vocab.AddToken(merged);
            merges.Add(new MergeRule(left, right, merged, step));

            ApplyMergeParallel(wordTokens, left, right, merged);
        }
        return new BpeModel(vocab, merges, _preTokeniser);
    }

    private async Task<Dictionary<string, int>> CountWordsAsync(
        IAsyncEnumerable<string> documents,
        CancellationToken cancellationToken)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        await foreach (string doc in documents.WithCancellation(cancellationToken))
        {
            foreach (string word in _preTokeniser.PreTokenise(doc))
            {
                counts[word] = counts.GetValueOrDefault(word) + 1;
            }
        }
        return counts;
    }

    private Dictionary<string, List<string>> BuildInitialWordTokens(Dictionary<string, int> wordFreqs)
    {
        var result = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (string word in wordFreqs.Keys)
        {
            var tokens = new List<string>(_byteLevel
                ? Vocabulary.ByteTokenise(word)
                : word.Select(c => c.ToString()));
            result[word] = tokens;
        }
        return result;
    }

    private static Dictionary<(string, string), int> CountPairsParallel(
        Dictionary<string, List<string>> wordTokens,
        Dictionary<string, int> wordFreqs)
    {
        var globalCounts = new ConcurrentDictionary<(string, string), long>();
        
        Parallel.ForEach(wordTokens, kvp =>
        {
            var word = kvp.Key;
            var tokens = kvp.Value;
            int freq = wordFreqs[word];

            for (int i = 0; i < tokens.Count - 1; i++)
            {
                var pair = (tokens[i], tokens[i + 1]);
                globalCounts.AddOrUpdate(pair, freq, (_, existing) => existing + freq);
            }
        });

        return globalCounts.ToDictionary(kvp => kvp.Key, kvp => (int)kvp.Value);
    }

    private static (string left, string right, int count) FindBestPair(
        Dictionary<(string, string), int> pairCounts)
    {
        string bestLeft = string.Empty;
        string bestRight = string.Empty;
        int bestCount = 0;

        foreach (var ((left, right), count) in pairCounts)
        {
            if (count > bestCount ||
               (count == bestCount &&
                string.Compare(left + right, bestLeft + bestRight, StringComparison.Ordinal) < 0))
            {
                bestLeft = left;
                bestRight = right;
                bestCount = count;
            }
        }
        return (bestLeft, bestRight, bestCount);
    }

    private static void ApplyMergeParallel(
        Dictionary<string, List<string>> wordTokens,
        string left, string right, string merged)
    {
        Parallel.ForEach(wordTokens.Values, tokens =>
        {
            for (int i = 0; i < tokens.Count - 1; )
            {
                if (tokens[i] == left && tokens[i + 1] == right)
                {
                    tokens[i] = merged;
                    tokens.RemoveAt(i + 1);
                }
                else
                {
                    i++;
                }
            }
        });
    }
}
