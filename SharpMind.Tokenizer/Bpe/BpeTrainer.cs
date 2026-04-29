using SharpMind.Tokenizer.PreTokeniser;
using SharpMind.Tokenizer.Vocab;

namespace SharpMind.Tokenizer.Bpe;

/// <summary>
/// Trains a BPE vocabulary from a text corpus.
///
/// Algorithm (Sennrich et al., 2015):
///   1. Pre-tokenise all text into words.
///   2. Represent each word as a sequence of character/byte tokens.
///   3. Count all adjacent token pairs across the corpus.
///   4. Merge the most frequent pair into a new token.
///   5. Repeat until <see cref="TargetVocabSize"/> is reached.
///
/// Supports both <see cref="IEnumerable{String}"/> (synchronous, for small corpora)
/// and <see cref="IAsyncEnumerable{String}"/> (streaming, for large corpora that
/// plug directly into <c>SharpMind.Data</c> pipelines).
/// </summary>
public sealed class BpeTrainer
{
    private readonly int _targetVocabSize;
    private readonly int _minFrequency;
    private readonly IPreTokeniser _preTokeniser;
    private readonly SpecialTokens _specials;
    private readonly bool _byteLevel;
    private readonly Action<string>? _progressCallback;

    // ── Construction ──────────────────────────────────────────────────────

    /// <param name="targetVocabSize">Total vocabulary size including special and byte tokens.</param>
    /// <param name="minFrequency">
    /// Minimum pair frequency to be merged. Pairs below this threshold are skipped.
    /// Default 2 — pair must appear at least twice.
    /// </param>
    /// <param name="preTokeniser">
    /// How to split raw text before BPE. Defaults to <see cref="Gpt2PreTokeniser"/>.
    /// </param>
    /// <param name="specials">Special tokens. Defaults to standard BOS/EOS/PAD/UNK.</param>
    /// <param name="byteLevel">
    /// When true (default), each byte 0x00–0xFF is added to the vocab before merges.
    /// Guarantees any unicode text can be encoded without [UNK].
    /// </param>
    /// <param name="progressCallback">
    /// Called after each merge with a status message. Useful for long training runs.
    /// </param>
    public BpeTrainer(
        int targetVocabSize,
        int minFrequency = 2,
        IPreTokeniser? preTokeniser = null,
        SpecialTokens? specials = null,
        bool byteLevel = true,
        Action<string>? progressCallback = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetVocabSize);
        _targetVocabSize = targetVocabSize;
        _minFrequency = minFrequency;
        _preTokeniser = preTokeniser ?? new Gpt2PreTokeniser();
        _specials = specials ?? new SpecialTokens();
        _byteLevel = byteLevel;
        _progressCallback = progressCallback;
    }

    // ── Training ──────────────────────────────────────────────────────────

    /// <summary>
    /// Trains from an async document stream.
    /// Plugs directly into a <c>SharpMind.Data</c> pipeline:
    /// <code>
    /// var model = await trainer.TrainAsync(pipeline.ReadAsync());
    /// </code>
    /// </summary>
    public async Task<BpeModel> TrainAsync(
        IAsyncEnumerable<string> documents,
        CancellationToken cancellationToken = default)
    {
        var wordFreqs = await CountWordsAsync(documents, cancellationToken);
        return RunBpe(wordFreqs);
    }

    // ── Core BPE algorithm ────────────────────────────────────────────────

    private BpeModel RunBpe(Dictionary<string, int> wordFreqs)
    {
        var vocab = new Vocabulary(_specials, addByteTokens: _byteLevel);
        var merges = new List<MergeRule>();

        // Represent each word as a list of byte-level tokens
        var wordTokens = BuildInitialWordTokens(wordFreqs, vocab);

        int mergesNeeded = _targetVocabSize - vocab.Size;
        if (mergesNeeded <= 0)
            return new BpeModel(vocab, merges, _preTokeniser);

        _progressCallback?.Invoke(
            $"Starting BPE: {vocab.Size} base tokens, {mergesNeeded} merges needed.");

        for (int step = 0; step < mergesNeeded; step++)
        {
            // Count all adjacent pairs
            var pairCounts = CountPairs(wordTokens, wordFreqs);
            if (pairCounts.Count == 0) break;

            // Find the best pair (highest frequency, ties broken by lexicographic order)
            var (left, right, count) = FindBestPair(pairCounts);
            if (count < _minFrequency) break;

            // Create the merged token
            string merged = left + right;
            int mergeId = vocab.AddToken(merged);
            merges.Add(new MergeRule(left, right, merged, step));

            // Apply the merge to all word sequences
            ApplyMerge(wordTokens, left, right, merged);

            if (step % 1000 == 0)
                _progressCallback?.Invoke(
                    $"Merge {step + 1}/{mergesNeeded}: '{left}' + '{right}' → '{merged}' (freq={count})");
        }

        _progressCallback?.Invoke($"Training complete. Vocab size: {vocab.Size}");
        return new BpeModel(vocab, merges, _preTokeniser);
    }

    // ── Word frequency counting ───────────────────────────────────────────

    private async Task<Dictionary<string, int>> CountWordsAsync(
        IAsyncEnumerable<string> documents,
        CancellationToken cancellationToken)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        await foreach (string doc in documents.WithCancellation(cancellationToken))
            foreach (string word in _preTokeniser.PreTokenise(doc))
                counts[word] = counts.GetValueOrDefault(word) + 1;
        return counts;
    }

    // ── Initial word → byte token representation ──────────────────────────

    private Dictionary<string, List<string>> BuildInitialWordTokens(
        Dictionary<string, int> wordFreqs,
        Vocabulary vocab)
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

    // ── Pair counting ─────────────────────────────────────────────────────

    private static Dictionary<(string, string), int> CountPairs(
        Dictionary<string, List<string>> wordTokens,
        Dictionary<string, int> wordFreqs)
    {
        var counts = new Dictionary<(string, string), int>();
        foreach (var (word, tokens) in wordTokens)
        {
            int freq = wordFreqs[word];
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                var pair = (tokens[i], tokens[i + 1]);
                counts[pair] = counts.GetValueOrDefault(pair) + freq;
            }
        }
        return counts;
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
                string.Compare(left + right, bestLeft + bestRight,
                    StringComparison.Ordinal) < 0))
            {
                bestLeft = left;
                bestRight = right;
                bestCount = count;
            }
        }
        return (bestLeft, bestRight, bestCount);
    }

    // ── Merge application ─────────────────────────────────────────────────

    private static void ApplyMerge(
        Dictionary<string, List<string>> wordTokens,
        string left, string right, string merged)
    {
        foreach (var tokens in wordTokens.Values)
        {
            int i = 0;
            while (i < tokens.Count - 1)
            {
                if (tokens[i] == left && tokens[i + 1] == right)
                {
                    tokens[i] = merged;
                    tokens.RemoveAt(i + 1);
                }
                else i++;
            }
        }
    }
}
