using SharpMind.Inference.Grammar;

namespace SharpMind.Inference;

/// <summary>
/// Shared sampling entry point that threads a grammar constraint through the
/// sampler. The constraint masks the caller's mutable logits in place, so
/// callers must own the span (the model output itself must not be passed when a
/// constraint is active).
/// </summary>
internal static class SamplingHelpers
{
    /// <summary>
    /// Applies <see cref="SamplingConfig.Constraint"/> (when present), samples
    /// one token, then commits the choice so the next step's mask reflects it.
    /// Returns false when the grammar has no legal continuation; the caller
    /// should stop without emitting a token.
    /// </summary>
    public static bool TrySampleConstrained(
        Span<float> logits, SamplingConfig config, Random rng, out int token)
    {
        IGrammarConstraint? constraint = config.Constraint;
        if (constraint is null)
        {
            token = Sampler.Sample(logits, config, rng);
            return true;
        }

        constraint.Apply(logits);
        if (constraint.IsDead)
        {
            token = -1;
            return false;
        }

        token = Sampler.Sample(logits, config, rng);
        constraint.Accept(token);
        return true;
    }
}