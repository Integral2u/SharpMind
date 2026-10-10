namespace SharpMind.Model.Layers.Ffn;

/// <summary>
/// Streaming MoE expert residency hook. A MoE layer calls
/// <see cref="EnsureExpertsLoaded"/> with the union of experts its router selected for the
/// current token; the implementor (wired by the streaming weights when
/// <c>DesiredResidentExperts &gt; 0</c>) loads any of those experts that are not resident and
/// keeps the most-used ones pinned across tokens. Models whose loader cannot slice experts, or
/// loads with residency disabled, never wire this hook and evaluate every expert from the
/// fully-loaded layer exactly as before.
/// </summary>
public interface IExpertResidencyHost
{
    /// <summary>
    /// Ensures the given routed experts have raw quantized weights pushed into the layer's
    /// expert projections before the expert matmuls run. Spans the whole batch's selection.
    /// </summary>
    void EnsureExpertsLoaded(ReadOnlySpan<int> expertIndices);
}