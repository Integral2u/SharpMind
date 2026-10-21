using SharpMind.Data.Sources;
using SharpMind.Tokenization.Bpe;

namespace SharpMind.Training;

/// <summary>The suggested BPE vocabulary size for a corpus.</summary>
/// <param name="CurrentVocabSize">The vocabulary the job already has (0 = not yet trained).</param>
/// <param name="SuggestedVocabSize">The smallest size that is actually worthwhile.</param>
/// <param name="MaxProbeVocabSize">The largest size the analysis explored.</param>
/// <param name="GrowthWorthwhile">False when the current vocab already covers the corpus well.</param>
/// <param name="Analysis">The full savings curve for display.</param>
public sealed record VocabSuggestion(
    int CurrentVocabSize,
    int SuggestedVocabSize,
    int MaxProbeVocabSize,
    bool GrowthWorthwhile,
    BpeGrowthAnalysis Analysis)
{
    /// <summary>Tokens-per-word now versus at the suggested size.</summary>
    public double TokensPerWordNow => Analysis.TokensPerWord(CurrentVocabSize);

    /// <summary>Tokens-per-word if the suggested size is adopted.</summary>
    public double TokensPerWordSuggested => Analysis.TokensPerWord(SuggestedVocabSize);

    /// <summary>Total corpus tokens now versus at the suggested size.</summary>
    public long CorpusTokensNow => Analysis.TokensAt(CurrentVocabSize);

    /// <summary>Total corpus tokens at the suggested size.</summary>
    public long CorpusTokensSuggested => Analysis.TokensAt(SuggestedVocabSize);
}

/// <summary>
/// Suggests the smallest BPE vocabulary that meaningfully compresses a corpus.
///
/// Runs the ordinary greedy BPE over the FULL corpus (nothing is persisted) and
/// looks for the point where extra merges stop paying for themselves — either
/// because they no longer add tokens to the vocabulary (every distinct word is
/// already a single token) or because the marginal saving is negligible. That
/// avoids over-sizing the vocabulary (bigger embedding, no real benefit) while
/// still growing a base vocabulary when the new corpus repeats itself.
///
/// Always advisory: the caller decides whether to apply the suggestion and grow
/// the tokenizer (see <see cref="BpeTrainer.GrowAsync"/>).
/// </summary>
public static class VocabSuggester
{
    /// <summary>Default coverage fraction used when no explicit target is given.</summary>
    public const double DefaultCoverage = 0.98;

    /// <summary>
    /// Analyses <paramref name="source"/> in full and returns the smallest
    /// worthwhile vocabulary. <paramref name="currentVocabSize"/> lets the result
    /// report whether growth is worthwhile; 0 means "unknown / fresh".
    /// </summary>
    public static async Task<VocabSuggestion> SuggestAsync(
        IDataSource source,
        int currentVocabSize,
        int maxProbeVocabSize = 8192,
        double coverage = DefaultCoverage,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (maxProbeVocabSize < 256) maxProbeVocabSize = 256;

        var trainer = new BpeTrainer(maxProbeVocabSize);
        var analysis = await trainer.AnalyseAsync(source.ReadAsync(), maxProbeVocabSize, cancellationToken);

        // Candidate sizes: the coverage knee, and the marginal-savings plateau.
        // The plateau is the stronger signal ("each distinct word already one
        // token" => adding more merges buys nothing), so prefer the smaller of
        // the two when it is still above the base vocabulary.
        int coverageSize = analysis.SmallestVocabForCoverage(coverage);
        int plateauSize = analysis.SmallestVocabForMarginal(minSavedPerMerge: 1);
        int suggested = Math.Min(Math.Max(coverageSize, plateauSize), analysis.FinalVocabSize);

        // Never suggest below what the model already has, and treat "no gain over
        // the current vocab" as not worthwhile.
        bool worthwhile = suggested > currentVocabSize
            && analysis.MaxSavings > 0
            && analysis.TokensAt(currentVocabSize) > analysis.TokensAt(suggested);

        return new VocabSuggestion(
            CurrentVocabSize: currentVocabSize,
            SuggestedVocabSize: worthwhile ? suggested : Math.Max(currentVocabSize, analysis.BaseVocabSize),
            MaxProbeVocabSize: analysis.FinalVocabSize,
            GrowthWorthwhile: worthwhile,
            Analysis: analysis);
    }
}
