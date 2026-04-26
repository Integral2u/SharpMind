using System.Runtime.CompilerServices;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;
namespace SharpMind.Data.Pipeline;
internal sealed class BranchNode : PipelineNode
{
    private readonly PipelineNode _parent;
    private readonly string _name;
    private readonly Func<string, bool> _predicate;

    internal BranchOutputNode MatchOutput { get; }
    internal BranchOutputNode OtherOutput { get; }

    internal BranchNode(PipelineNode parent, string name, Func<string, bool> predicate)
    {
        _parent = parent;
        _name = name;
        _predicate = predicate;
        MatchOutput = new BranchOutputNode(this, isMatch: true);
        OtherOutput = new BranchOutputNode(this, isMatch: false);
    }

    // BranchNode itself is not readable — only its outputs are
    public override IAsyncEnumerable<string> ReadAsync(CancellationToken ct = default)
        => throw new InvalidOperationException(
            "Read from BranchNode.MatchOutput or BranchNode.OtherOutput, not the branch itself.");

    public override string Describe(int depth = 0)
        => $"{_parent.Describe(depth)}\n{Indent(depth)}Branch: {_name}";

    internal async IAsyncEnumerable<string> ReadMatchAsync(
        bool isMatch,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (string doc in _parent.ReadAsync(ct))
            if (_predicate(doc) == isMatch)
                yield return doc;
    }
}

