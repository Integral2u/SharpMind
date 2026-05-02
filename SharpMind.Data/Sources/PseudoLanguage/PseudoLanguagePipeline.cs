using System.Runtime.CompilerServices;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Batching;

namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class PseudoLanguagePipeline : PipelineNode
{
    private readonly PseudoLanguageGenerator _generator;
    private readonly ComplexityLevel _level;
    private readonly int _sequenceCount;
    private readonly List<GeneratedSequence> _sequences;
    private int _index;

    public PseudoLanguagePipeline(
        VocabConfig config,
        ComplexityLevel level,
        int sequenceCount)
    {
        _generator = new PseudoLanguageGenerator(config);
        _level = level;
        _sequenceCount = sequenceCount;

        _sequences = level switch
        {
            ComplexityLevel.Options => _generator.GenerateOptions(sequenceCount).ToList(),
            ComplexityLevel.Patterns => _generator.GeneratePatterns(sequenceCount).ToList(),
            ComplexityLevel.Syntactic => _generator.GenerateSyntacticSequences(sequenceCount).ToList(),
            _ => _generator.GenerateSyntacticSequences(sequenceCount).ToList(),
        };
    }

    public PseudoLanguageGenerator Generator => _generator;

    public override async IAsyncEnumerable<string> ReadAsync(CancellationToken cancellationToken = default)
    {
        foreach (var seq in _sequences)
        {
            if (cancellationToken.IsCancellationRequested) yield break;
            _index++;
            yield return seq.RawText;
        }
    }

    public override string Describe(int depth = 0)
        => new string(' ', depth * 2) + $"PseudoLanguage({_level}, vocab={_generator.VocabSize}, n={_sequenceCount})";
}

public static class PseudoLanguageExtensions
{
    public static DataLoader ToDataLoader(
        this PseudoLanguagePipeline pipeline,
        int batchSize,
        int maxSeqLen,
        int eosTokenId = 2,
        int padTokenId = 0)
    {
        var generator = pipeline.Generator;

        int[] Tokenize(string text)
        {
            return text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(word => generator.TextToId(word))
                .Where(id => id >= 0)
                .ToArray();
        }

        var batcher = new PackingBatcher(batchSize, maxSeqLen, eosTokenId, padTokenId);
        return new DataLoader(pipeline, Tokenize, batcher);
    }
}