using SharpMind.Data.Pipeline;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
            ComplexityLevel.Options => [.. _generator.GenerateOptions(sequenceCount)],
            ComplexityLevel.Patterns => [.. _generator.GeneratePatterns(sequenceCount)],
            ComplexityLevel.Syntactic => [.. _generator.GenerateSyntacticSequences(sequenceCount)],
            _ => [.. _generator.GenerateSyntacticSequences(sequenceCount)],
        };
    }

    public PseudoLanguageGenerator Generator => _generator;

    public override async IAsyncEnumerable<string> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        foreach (var seq in _sequences)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _index++;
            yield return seq.RawText;
        }
    }

    public override string Describe(int depth = 0)
        => new string(' ', depth * 2) + $"PseudoLanguage({_level}, vocab={_generator.VocabSize}, n={_sequenceCount})";
}