using SharpMind.Data.Sources;

namespace SharpMind.Data.Pipeline;
/// <summary>
/// Entry point for building cleaning DAGs.
///
/// <code>
/// var node = CleaningPipeline
///     .From(new JsonlSource("data.jsonl"))
///     .Pipe(new NormaliseWhitespace())
///     .Pipe(new MinLengthFilter(50));
///
/// var (code, prose) = node.Branch("is-code", d => d.StartsWith("```"));
/// var cleaned = CleaningPipeline.Merge(
///     code.Pipe(new StripHtml()),
///     prose.Pipe(new LowerCase()));
/// </code>
/// </summary>
public static class CleaningPipeline
{
    /// <summary>Creates the root node of a pipeline from a data source.</summary>
    public static PipelineNode From(IDataSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return new SourceNode(source);
    }

    /// <summary>
    /// Merges multiple pipeline nodes into one stream (concatenation order).
    /// Use after <see cref="PipelineNode.Branch"/> to re-join split streams.
    /// </summary>
    public static PipelineNode Merge(params PipelineNode[] nodes)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        if (nodes.Length < 2)
            throw new ArgumentException("Merge requires at least two nodes.", nameof(nodes));
        return new MergeNode(nodes);
    }

    /// <summary>
    /// Merges multiple pipeline nodes into one stream (concatenation order).
    /// </summary>
    public static PipelineNode Merge(IEnumerable<PipelineNode> nodes)
        => Merge([.. nodes]);
}