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

    /// <summary>
    /// Merges multiple pipeline nodes by interleaving their documents in
    /// proportion to the supplied weights - the replay-mixing form of
    /// <see cref="Merge(PipelineNode[])"/>. Weights are relative (they need not
    /// sum to one); a weight of zero excludes that node entirely.
    /// </summary>
    /// <example>
    /// <code>
    /// var mixed = CleaningPipeline.MergeWeighted(
    ///     (instructions, 3.0),   // 75%
    ///     (replay, 1.0));        // 25%
    /// </code>
    /// </example>
    public static PipelineNode MergeWeighted(params (PipelineNode Node, double Weight)[] parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (parts.Length < 2)
            throw new ArgumentException("MergeWeighted requires at least two nodes.", nameof(parts));

        foreach (var (node, weight) in parts)
        {
            ArgumentNullException.ThrowIfNull(node);
            if (double.IsNaN(weight) || double.IsInfinity(weight) || weight < 0)
                throw new ArgumentOutOfRangeException(nameof(parts), weight,
                    "Weights must be finite and non-negative.");
        }

        if (parts.Sum(p => p.Weight) <= 0)
            throw new ArgumentException("At least one weight must be positive.", nameof(parts));

        return new WeightedMergeNode(parts);
    }
}