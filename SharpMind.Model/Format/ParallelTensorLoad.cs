namespace SharpMind.Model.Format;

/// <summary>
/// Shared helpers for the opt-in parallel full-weight load in
/// <see cref="GgufLoader"/> / <see cref="SmmLoader"/>.
///
/// The parallel path exists because a full load is dominated by per-tensor
/// seeks + dequantization, both of which are independent per tensor, so it is
/// the default (degree 0 = one per core). Degree 1 keeps the original
/// sequential loop, byte-for-byte, for a memory-starved host or for reproducing
/// the old ordering exactly; peak transient memory scales with the degree.
/// </summary>
internal static class ParallelTensorLoad
{
    /// <summary>
    /// Clamps a caller-requested degree of parallelism to something that can
    /// actually help, or 0 when the load must stay sequential.
    ///
    /// A <paramref name="requested"/> of 0 — the default — means "let the library
    /// decide": one worker per core. 1 explicitly asks for the original
    /// sequential loop.
    ///
    /// Returns 0 (not 1) so callers can branch on a single "no fan-out" test
    /// and keep their original loop verbatim.
    /// </summary>
    public static int ResolveDegree(int requested, int workItemCount, bool safeIo)
    {
        if (requested < 0)
            throw new ArgumentOutOfRangeException(nameof(requested), requested, "Degree of parallelism cannot be negative.");

        // A single-processor host would pay the extra streams and the
        // partitioning for no overlap at all.
        int cores = Environment.ProcessorCount;
        if (cores <= 1) return 0;

        // Safe-IO is the WASM/browser path (WeightStreamFactory falls back to
        // plain FileStream). Browser workers have no shared memory and no real
        // threads to overlap with, and Environment.ProcessorCount there reports
        // the host's core count rather than anything actionable — so the flag is
        // the only honest signal that this platform cannot fan out.
        if (safeIo) return 0;

        int wanted = requested == 0 ? cores : requested;

        // Nothing to split: one work item can't be shared without two threads
        // writing the same block's dictionaries.
        if (wanted <= 1 || workItemCount <= 1) return 0;

        // Never spin up more workers than there are work items: idle workers
        // would each still open their own file view for nothing.
        return Math.Min(wanted, Math.Min(cores, workItemCount));
    }

    /// <summary>
    /// Spreads whole work items over <paramref name="degree"/> buckets by
    /// descending weight, always handing the next item to the currently
    /// lightest bucket. Without this a model whose blocks vary in size would
    /// leave one worker holding nearly all the bytes while the rest idle, and
    /// the load would run no faster than sequential.
    ///
    /// Items stay whole — a bucket is a block, and splitting one across workers
    /// would mean two threads writing the same block's dictionaries.
    ///
    /// Fewer buckets than requested simply come back empty (when there are
    /// fewer blocks than workers); the caller skips those.
    /// </summary>
    public static List<List<T>> PartitionByWeight<T>(List<T> items, int degree, Func<T, long> weightOf)
    {
        // Weight each item once: the sort comparator would otherwise recompute
        // it O(n log n) times, and the weight is a byte-count walk over a shape.
        var weighted = new List<(T Item, long Weight)>(items.Count);
        foreach (var item in items) weighted.Add((item, weightOf(item)));
        weighted.Sort(static (a, b) => b.Weight.CompareTo(a.Weight));

        var buckets = new List<List<T>>(degree);
        var loads = new long[degree];
        for (int i = 0; i < degree; i++) buckets.Add([]);

        foreach (var (item, weight) in weighted)
        {
            int lightest = 0;
            for (int i = 1; i < degree; i++)
                if (loads[i] < loads[lightest]) lightest = i;
            buckets[lightest].Add(item);
            loads[lightest] += weight;
        }
        return buckets;
    }

    /// <summary>
    /// Turns per-tensor completion counts into a progress callback that only
    /// ever moves forward.
    ///
    /// The sequential loop reported <c>loaded / total</c> per tensor in file
    /// order, so callers are entitled to a non-decreasing sequence. Workers
    /// finish out of order — thread A can increment to 5, thread B to 6, and B
    /// reach the callback first — which would rewind the bar. Reporting under
    /// the lock (rather than merely updating the maximum under it) is what
    /// restores the ordering guarantee, because only the lock holder invokes
    /// the callback and it is released in the same order the values were taken.
    ///
    /// The lock is held across the callback on purpose; it is a cheap hand-off
    /// (the CUI wraps this in <see cref="Progress{T}"/>, which just posts to the
    /// sync context), and dropping it would trade a cosmetic wart for the
    /// possibility of a deadlock in caller code.
    /// </summary>
    internal sealed class ProgressReporter(IProgress<float>? inner, int total)
    {
        private readonly Lock _gate = new();
        private float _last = -1f;

        public void ReportLoaded(int loaded)
        {
            if (inner is null) return;
            float fraction = total > 0 ? (float)loaded / total : 1f;
            lock (_gate)
            {
                if (fraction <= _last) return;
                _last = fraction;
                inner.Report(fraction);
            }
        }

        /// <summary>Final 100%, whether or not the last tensor reported it.</summary>
        public void ReportComplete()
        {
            if (inner is null) return;
            lock (_gate)
            {
                if (_last >= 1f) return;
                _last = 1f;
                inner.Report(1f);
            }
        }
    }
}
