namespace SharpMind.Model;

/// <summary>
/// Memory-driven sizing for streaming MoE expert residency. The caller (streaming weights or
/// the CUI) tells it how much memory is usable and how costly one expert is; it answers how
/// many experts per layer can be kept resident without overflowing the budget, and clamps a
/// desired count down to whatever memory actually permits.
/// </summary>
public static class MoEResidencyPlanner
{
    /// <summary>Fraction of the usable budget kept as wriggle room so pinned experts plus the
    /// layer window, activations and debug copies never trip an OOM.</summary>
    public const double Headroom = 0.20;

    /// <summary>
    /// Largest per-layer resident-expert count that fits in the budget when one expert costs
    /// <paramref name="bytesPerExpertPerLayer"/> across <paramref name="numLayers"/> identical
    /// MoE layers, after reserving <paramref name="headroom"/>. Clamped to
    /// <paramref name="numExperts"/>. Returns 0 when the budget cannot hold even one expert per
    /// layer (everything then streams on demand).
    /// </summary>
    public static int SuggestResidentExperts(
        long usableBudgetBytes,
        long bytesPerExpertPerLayer,
        int numLayers,
        int numExperts,
        double headroom = Headroom)
    {
        if (usableBudgetBytes <= 0 || bytesPerExpertPerLayer <= 0 || numLayers <= 0 || numExperts <= 0)
            return 0;

        double budget = usableBudgetBytes * (1.0 - headroom);
        long perExpertAcrossAllLayers = bytesPerExpertPerLayer * numLayers;
        double fit = Math.Floor(budget / Math.Max(1, perExpertAcrossAllLayers));
        return (int)Math.Clamp(fit, 0, numExperts);
    }

    /// <summary>Reduces a desired count to what the budget permits ("evict if we have to"):
    /// never pins more experts than memory can hold.</summary>
    public static int ClampToBudget(int desired, int budgetAllows) => Math.Min(desired, budgetAllows);
}