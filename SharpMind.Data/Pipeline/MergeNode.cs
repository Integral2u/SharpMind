using System.Runtime.CompilerServices;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;

namespace SharpMind.Data.Pipeline;

internal sealed class MergeNode : PipelineNode
{
    private readonly PipelineNode[] _parents;

    internal MergeNode(IEnumerable<PipelineNode> parents)
        => _parents = [.. parents];

    public override async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        foreach (var parent in _parents)
            await foreach (string doc in parent.ReadAsync(ct))
                yield return doc;
    }

    public override string Describe(int depth = 0)
    {
        var branches = string.Join("\n", _parents.Select(p => p.Describe(depth + 1)));
        return $"{Indent(depth)}Merge:\n{branches}";
    }
}