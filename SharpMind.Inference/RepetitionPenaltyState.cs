namespace SharpMind.Inference;

/// <summary>
/// Per-generation repetition-penalty bookkeeping. The penalty itself is one
/// line per token id; what costs is deciding which ids to apply it to, and the
/// two modes need opposite strategies:
///
///   window &gt; 0: scan the last N prompt and generated ids every step (cheap,
///   because N is small).
///   window == 0: the full context. The prompt is scanned once into a distinct
///   id list and the generated ids are folded into a set incrementally, so the
///   cost is one pass over the context per generation rather than one per token.
///
/// Shared by <see cref="StandardGenerator{T}"/> and <see cref="BatchedGenerator"/>
/// so a batched sequence and a standalone one apply exactly the same penalty.
/// </summary>
internal sealed class RepetitionPenaltyState
{
    /// <summary>Pooled distinct-token tracking; cleared each step.</summary>
    private HashSet<int>? _seen;
    /// <summary>Distinct prompt ids, built once per generation for full-context penalties.</summary>
    private List<int>? _promptDistinct;
    /// <summary>Growing distinct generated ids for full-context penalties.</summary>
    private HashSet<int>? _generatedDistinct;
    private int _generatedDistinctThrough;

    /// <summary>Drops the cached prompt/generated views; call when a new generation starts.</summary>
    public void Reset()
    {
        _promptDistinct = null;
        _generatedDistinct = null;
        _generatedDistinctThrough = 0;
    }

    public void Apply(
        Span<float> logits,
        ReadOnlySpan<int> promptIds,
        List<int> generatedIds,
        float penalty,
        int window)
    {
        // Penalize the top of the vector in place, once per DISTINCT token id
        // across the window/context (matching HF/llama.cpp), NOT once per
        // occurrence. Scaling per occurrence would raise the penalty to
        // penalty^count for common words, wiping them from the distribution as
        // generation (and context) grows.
        _seen ??= new HashSet<int>(512);
        _seen.Clear();
        var seen = _seen;

        if (window > 0)
        {
            int promptStart = Math.Max(0, promptIds.Length - window);
            RepetitionPenalty.Apply(logits, promptIds[promptStart..], penalty, seen);
            int genStart = Math.Max(0, generatedIds.Count - window);
            RepetitionPenalty.Apply(logits, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generatedIds)[genStart..], penalty, seen);
            return;
        }

        // Full-context penalty: only distinct ids matter, so scan the prompt
        // once per generation and the generated list incrementally instead of
        // rescanning every occurrence on every token.
        _promptDistinct ??= Distinct(promptIds);
        _generatedDistinct ??= new HashSet<int>(generatedIds.Count);
        for (; _generatedDistinctThrough < generatedIds.Count; _generatedDistinctThrough++)
            _generatedDistinct.Add(generatedIds[_generatedDistinctThrough]);

        RepetitionPenalty.Apply(logits, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_promptDistinct), penalty, seen);
        foreach (int id in _generatedDistinct)
            RepetitionPenalty.ApplyToken(logits, id, penalty, seen);

        static List<int> Distinct(ReadOnlySpan<int> ids)
        {
            var set = new HashSet<int>(ids.Length);
            var list = new List<int>(ids.Length);
            foreach (int id in ids)
                if (set.Add(id)) list.Add(id);
            return list;
        }
    }
}
