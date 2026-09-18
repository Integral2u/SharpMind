namespace SharpMind.Inference.Grammar;

/// <summary>
/// Masks token logits so generation can only emit tokens that keep the decoded
/// output inside a grammar, and tracks the grammar state as tokens are
/// committed. One instance is threaded through an entire generation: call
/// <see cref="Apply"/> immediately before sampling each token, then
/// <see cref="Accept"/> the token that was sampled.
///
/// This works uniformly for ordinary, speculative, and Medusa decoding without
/// touching the samplers: a draft token that the grammar forbids is simply
/// rejected by the greedy-agreement check, because its logit is masked.
/// </summary>
public interface IGrammarConstraint
{
    /// <summary>True when the grammar has reached a state where it may stop.</summary>
    bool CanStop { get; }

    /// <summary>True when no ordinary token can continue the grammar (a dead end).</summary>
    bool IsDead { get; }

    /// <summary>
    /// True when <paramref name="tokenId"/> was left unmasked by the most recent
    /// <see cref="Apply"/>.
    /// </summary>
    bool IsAllowed(int tokenId);

    /// <summary>Masks every disallowed logit in <paramref name="logits"/> so sampling can never pick it.</summary>
    void Apply(Span<float> logits);

    /// <summary>Commits a sampled token, advancing the grammar state.</summary>
    void Accept(int tokenId);

    /// <summary>Returns the constraint to its initial state for a fresh generation.</summary>
    void Reset();
}