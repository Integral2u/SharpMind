namespace SharpMind.Tokenization.Bpe;

/// <summary>One greedy BPE merge and how many corpus tokens it saves.</summary>
/// <param name="VocabSize">The vocabulary size AFTER this merge is applied.</param>
/// <param name="Merged">The merged token string.</param>
/// <param name="Frequency">How often the underlying pair occurred in the corpus.</param>
/// <param name="TokensSaved">Tokens removed from the corpus encoding by this merge.</param>
/// <param name="CumulativeTokens">Total corpus tokens remaining after this merge.</param>
public readonly record struct MergeSaving(
    int VocabSize,
    string Merged,
    int Frequency,
    long TokensSaved,
    long CumulativeTokens);

/// <summary>
/// The result of running greedy BPE over a full corpus purely for analysis:
/// the per-merge savings curve plus the endpoints needed to pick the smallest
/// worthwhile vocabulary.
/// </summary>
public sealed class BpeGrowthAnalysis
{
    public BpeGrowthAnalysis(
        int baseVocabSize,
        int finalVocabSize,
        int totalWordOccurrences,
        long baselineTokens,
        long finalTokens,
        IReadOnlyList<MergeSaving> steps)
    {
        BaseVocabSize = baseVocabSize;
        FinalVocabSize = finalVocabSize;
        TotalWordOccurrences = totalWordOccurrences;
        BaselineTokens = baselineTokens;
        FinalTokens = finalTokens;
        Steps = steps;
    }

    /// <summary>Vocabulary size before any merge (specials + byte tokens).</summary>
    public int BaseVocabSize { get; }

    /// <summary>Largest vocabulary reached before the probe stopped.</summary>
    public int FinalVocabSize { get; }

    /// <summary>Total pre-tokenised word occurrences in the corpus.</summary>
    public int TotalWordOccurrences { get; }

    /// <summary>Corpus tokens with zero merges applied (all byte tokens).</summary>
    public long BaselineTokens { get; }

    /// <summary>Corpus tokens with every learned merge applied.</summary>
    public long FinalTokens { get; }

    /// <summary>Every merge learned, in rank order, with its savings.</summary>
    public IReadOnlyList<MergeSaving> Steps { get; }

    /// <summary>Total tokens removable by the very best (fully merged) vocabulary.</summary>
    public long MaxSavings => BaselineTokens - FinalTokens;

    /// <summary>Average tokens per word at the given vocabulary size.</summary>
    public double TokensPerWord(int vocabSize)
    {
        long tokens = TokensAt(vocabSize);
        return TotalWordOccurrences == 0 ? 0 : (double)tokens / TotalWordOccurrences;
    }

    /// <summary>Corpus token count when trained to <paramref name="vocabSize"/>.</summary>
    public long TokensAt(int vocabSize)
    {
        if (vocabSize <= BaseVocabSize) return BaselineTokens;
        long tokens = BaselineTokens;
        foreach (var s in Steps)
        {
            if (s.VocabSize > vocabSize) break;
            tokens = s.CumulativeTokens;
        }
        return tokens;
    }

    /// <summary>
    /// Smallest vocabulary size that reaches <paramref name="coverage"/> (0..1) of
    /// the total achievable savings. Returns <see cref="BaseVocabSize"/> when even
    /// the full vocabulary cannot reach the fraction (too little repetition).
    /// </summary>
    public int SmallestVocabForCoverage(double coverage)
    {
        if (coverage <= 0 || Steps.Count == 0 || MaxSavings <= 0) return BaseVocabSize;
        long target = (long)Math.Ceiling(MaxSavings * Math.Min(1.0, coverage));

        long best = 0;
        int size = BaseVocabSize;
        foreach (var s in Steps)
        {
            best += s.TokensSaved;
            size = s.VocabSize;
            if (best >= target) return size;
        }
        return FinalVocabSize;
    }

    /// <summary>
    /// Smallest vocabulary whose marginal merge still saves at least
    /// <paramref name="minSavedPerMerge"/> tokens — stops at the point extra
    /// merges stop paying for themselves (the "each word is already one token"
    /// plateau). Returns <see cref="BaseVocabSize"/> when nothing is worthwhile.
    /// </summary>
    public int SmallestVocabForMarginal(long minSavedPerMerge)
    {
        int size = BaseVocabSize;
        foreach (var s in Steps)
        {
            if (s.TokensSaved < minSavedPerMerge) break;
            size = s.VocabSize;
        }
        return size;
    }
}
