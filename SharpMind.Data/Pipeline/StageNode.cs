using System.Runtime.CompilerServices;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;
namespace SharpMind.Data.Pipeline;
internal sealed class StageNode : PipelineNode
{
    private readonly PipelineNode _parent;
    private readonly ICleaningStage _stage;

    internal StageNode(PipelineNode parent, ICleaningStage stage)
    {
        _parent = parent;
        _stage = stage;
    }

    public override async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (string doc in _parent.ReadAsync(ct))
        {
            string? result = _stage.Process(doc);
            if (result is not null)
                yield return result;
        }
    }

    public override string Describe(int depth = 0)
        => $"{_parent.Describe(depth)}\n{Indent(depth)}Stage: {_stage.Name}";
}