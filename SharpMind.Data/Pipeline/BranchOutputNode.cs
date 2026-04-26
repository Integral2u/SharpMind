namespace SharpMind.Data.Pipeline;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;
internal sealed class BranchOutputNode : PipelineNode
{
    private readonly BranchNode _branch;
    private readonly bool _isMatch;

    internal BranchOutputNode(BranchNode branch, bool isMatch)
    {
        _branch = branch;
        _isMatch = isMatch;
    }

    public override IAsyncEnumerable<string> ReadAsync(CancellationToken ct = default)
        => _branch.ReadMatchAsync(_isMatch, ct);

    public override string Describe(int depth = 0)
        => $"{_branch.Describe(depth)}\n{Indent(depth)}  → {(_isMatch ? "match" : "other")}";
}