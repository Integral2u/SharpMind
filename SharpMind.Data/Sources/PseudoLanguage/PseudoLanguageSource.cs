using System.Runtime.CompilerServices;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Sources.PseudoLanguage;

/// <summary>
/// A synthetic corpus of generated pseudo-language documents, exposed as an
/// <see cref="IDataSource"/> so it participates in the training wizard's source
/// picker and the normal tokenise → size → clean pipeline.
///
/// The vocabulary is built from morphemes and affixes per <see cref="VocabConfig"/>;
/// documents are produced at the selected <see cref="ComplexityLevel"/> (options,
/// morphological patterns, or full syntactic sequences). Because the corpus is
/// generated on construction it is fully reproducible only with a fixed seed.
/// </summary>
[ComponentKind("Pseudo Language", "Synthetic pseudo-language corpus.")]
public sealed class PseudoLanguageSource : IDataSource
{
    private readonly VocabConfig _config;
    private readonly ComplexityLevel _level;
    private readonly int _sequenceCount;
    private readonly PseudoLanguagePipeline _pipeline;
    private readonly PseudoLanguageGenerator _generator;

    public PseudoLanguageSource(
        [MinMaxDefault(1_000, 1_000_000, 5_000, 1_000)]
        [DefaultValue("5000")]
        [Tooltip("Target vocab size")]
        int vocabSize,
        [MinMaxDefault(10, 50_000, 300, 10)]
        [DefaultValue("300")]
        [Tooltip("Root morpheme count")]
        int rootMorphemes,
        [MinMaxDefault(0, 500, 20, 1)]
        [DefaultValue("20")]
        [Tooltip("Affix count")]
        int affixes,
        [MinMaxDefault(1, 10_000_000, 10_000, 100)]
        [DefaultValue("10000")]
        [Tooltip("Number of sequences")]
        int sequenceCount,
        [Tooltip("Syntactic level")]
        ComplexityLevel level = ComplexityLevel.Syntactic)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegative(rootMorphemes);
        ArgumentOutOfRangeException.ThrowIfNegative(affixes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequenceCount);

        _config = new VocabConfig
        {
            VocabSize = vocabSize,
            RootMorphemes = rootMorphemes,
            Affixes = affixes,
        };
        _level = level;
        _sequenceCount = sequenceCount;
        _pipeline = new PseudoLanguagePipeline(_config, level, sequenceCount);
        _generator = _pipeline.Generator;
    }

    public long? EstimatedCount => _sequenceCount;

    public string Description =>
        $"PseudoLanguage({_level}, vocab={_generator.VocabSize}, n={_sequenceCount})";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var text in _pipeline.ReadAsync(cancellationToken))
            yield return text;
    }

    public ValueTask DisposeAsync()
    {
        _generator.Dispose();
        return ValueTask.CompletedTask;
    }
}