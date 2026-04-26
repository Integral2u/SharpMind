using SharpMind.Data.Sources;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;
namespace SharpMind.Data.Pipeline;

internal sealed class SourceNode : PipelineNode
{
    private readonly IDataSource _source;
    internal SourceNode(IDataSource source) => _source = source;

    public override IAsyncEnumerable<string> ReadAsync(CancellationToken ct = default)
        => _source.ReadAsync(ct);

    public override string Describe(int depth = 0)
        => $"{Indent(depth)}Source: {_source.Description}";
}

