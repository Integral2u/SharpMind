using System.Text;

namespace SharpMind.Data.Pipeline;

/// <summary>
/// B2: a deterministic train/validation split built on <see cref="PipelineNode.Branch"/>.
/// A document's side is decided by a stable content hash (FNV-1a over its UTF-8
/// bytes, mixed with <paramref name="seed"/>) rather than its position, so the
/// same corpus always splits identically, and appending new files — the normal
/// incremental-training case — never moves an existing document to the other
/// side. Nothing is random at split time, so the split is reproducible across
/// runs and processes.
/// </summary>
public static class DeterministicSplit
{
    /// <summary>Number of hash buckets a document can land in; the split threshold is a fraction of these.</summary>
    public const int Buckets = 10_000;

    /// <summary>
    /// Splits <paramref name="pipeline"/> into a training node and a validation
    /// node. Roughly <paramref name="valFraction"/> of documents (by content
    /// hash) go to validation; the rest to training. Each returned node re-reads
    /// the pipeline independently, so use one, the other, or both.
    /// </summary>
    /// <param name="pipeline">The pipeline to split; both outputs inherit it.</param>
    /// <param name="valFraction">Fraction of documents for validation, in [0, 1).</param>
    /// <param name="seed">Mix-in so a different seed reshuffles the split without changing the corpus.</param>
    public static (PipelineNode Train, PipelineNode Val) TrainVal(
        this PipelineNode pipeline, double valFraction, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        if (valFraction < 0 || valFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(valFraction), valFraction,
                "valFraction must be in [0, 1).");

        int threshold = (int)Math.Round(valFraction * Buckets);
        var (val, train) = pipeline.Branch("train-val", doc => Bucket(doc, seed) < threshold);
        return (train, val);
    }

    /// <summary>
    /// True when <paramref name="document"/> belongs to the validation side for
    /// the given fraction and seed — the exact predicate the branch uses, exposed
    /// so callers can label a document without materialising both streams.
    /// </summary>
    public static bool IsValidation(string document, double valFraction, int seed = 0)
    {
        if (valFraction < 0 || valFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(valFraction), valFraction,
                "valFraction must be in [0, 1).");

        return Bucket(document, seed) < (int)Math.Round(valFraction * Buckets);
    }

    /// <summary>Stable bucket in [0, <see cref="Buckets"/>) for a document, mixed with <paramref name="seed"/>.</summary>
    public static int Bucket(string document, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(document);

        // FNV-1a over the UTF-8 bytes — deliberately not string.GetHashCode(),
        // which is randomised per process and would make the split irreproducible.
        uint hash = (2166136261u ^ (uint)seed);
        foreach (byte b in Encoding.UTF8.GetBytes(document))
        {
            hash ^= b;
            hash *= 16777619u;
        }
        return (int)(hash % Buckets);
    }
}
