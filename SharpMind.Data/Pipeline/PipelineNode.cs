using SharpMind.Data.Sources;

namespace SharpMind.Data.Pipeline;
/// <summary>
/// A node in the cleaning DAG. Produced by <see cref="CleaningPipeline"/>.
/// Chain further stages via <see cref="Pipe"/>, <see cref="Branch"/>,
/// or merge multiple nodes via <see cref="CleaningPipeline.Merge"/>.
/// </summary>
public abstract class PipelineNode
{
    /// <summary>Starts a new cleaning pipeline from a raw data source.</summary>
    public static PipelineNode From(IDataSource source)
        => new SourceNode(source);

    // Fluent builders


    /// <summary>Appends a stage that transforms every document.</summary>
    public PipelineNode Pipe(ICleaningStage stage)
        => new StageNode(this, stage);

    /// <summary>Appends a stage built from a lambda — no need to create a class.</summary>
    public PipelineNode Pipe(string name, Func<string, string?> transform)
        => Pipe(new LambdaStage(name, transform));

    /// <summary>
    /// Splits the stream: documents matching <paramref name="predicate"/> flow to
    /// the <paramref name="matchBranch"/>; all others flow to the node returned
    /// by this call. Both branches can be piped further and merged later.
    /// </summary>
    public (PipelineNode Match, PipelineNode Other) Branch(
        string name, Func<string, bool> predicate)
    {
        var node = new BranchNode(this, name, predicate);
        return (node.MatchOutput, node.OtherOutput);
    }

    // Document streaming

    /// <summary>
    /// Materialises the pipeline and streams processed documents.
    /// This is the terminal operation — call it on the last node in the graph.
    /// </summary>
    public abstract IAsyncEnumerable<string> ReadAsync(
        CancellationToken cancellationToken = default);

    /// <summary>Human-readable description of this node and its ancestors.</summary>
    public abstract string Describe(int depth = 0);
}
