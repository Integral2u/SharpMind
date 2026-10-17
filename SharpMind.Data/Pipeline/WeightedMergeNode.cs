using System.Runtime.CompilerServices;
using static SharpMind.Data.Pipeline.PipelineNodeExtensions;

namespace SharpMind.Data.Pipeline;

/// <summary>
/// B7: interleaves several pipeline nodes in proportion to their weights, for
/// replay mixing (e.g. 3 parts new instruction data to 1 part replay corpus).
/// Unlike <see cref="MergeNode"/>, which concatenates whole streams, this pulls
/// documents one at a time from the node that is furthest below its target
/// share (smooth weighted round-robin), so the mix stays on-ratio across the
/// whole run rather than being front-loaded by source.
///
/// The schedule is deterministic: the same weights and the same upstream order
/// always produce the same mixed order, which is what makes a resumed training
/// run see the same stream again.
/// </summary>
internal sealed class WeightedMergeNode : PipelineNode
{
    private readonly PipelineNode[] _parents;
    private readonly double[] _weights;
    private readonly double _totalWeight;

    internal WeightedMergeNode(IReadOnlyList<(PipelineNode Node, double Weight)> parts)
    {
        _parents = [.. parts.Select(p => p.Node)];
        _weights = [.. parts.Select(p => p.Weight)];
        _totalWeight = _weights.Sum();
    }

    public override async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var enumerators = new IAsyncEnumerator<string>[_parents.Length];
        for (int i = 0; i < _parents.Length; i++)
            enumerators[i] = _parents[i].ReadAsync(ct).GetAsyncEnumerator(ct);

        var current = new double[_parents.Length];
        var exhausted = new bool[_parents.Length];
        int remaining = _parents.Length;

        try
        {
            while (remaining > 0)
            {
                int chosen = -1;
                double best = double.NegativeInfinity;

                for (int i = 0; i < _parents.Length; i++)
                {
                    if (exhausted[i] || _weights[i] <= 0)
                        continue;

                    current[i] += _weights[i];
                    if (current[i] > best)
                    {
                        best = current[i];
                        chosen = i;
                    }
                }

                if (chosen < 0)
                {
                    // Every remaining node is weight-zero; nothing left to emit.
                    break;
                }

                current[chosen] -= _totalWeight;

                if (await enumerators[chosen].MoveNextAsync())
                {
                    ct.ThrowIfCancellationRequested();
                    yield return enumerators[chosen].Current;
                }
                else
                {
                    exhausted[chosen] = true;
                    remaining--;
                }
            }
        }
        finally
        {
            foreach (var enumerator in enumerators)
                await enumerator.DisposeAsync();
        }
    }

    public override string Describe(int depth = 0)
    {
        var lines = _parents
            .Select((p, i) => $"{Indent(depth + 1)}[{_weights[i]:0.###}] {p.Describe(depth + 2).Trim()}");
        return $"{Indent(depth)}WeightedMerge:\n{string.Join("\n", lines)}";
    }
}
