using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
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
