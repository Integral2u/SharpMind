namespace SharpMind.Tokenization.Bpe;

/// <summary>
/// A single BPE merge rule: the pair (<see cref="Left"/>, <see cref="Right"/>)
/// merges into the token <see cref="Merged"/>.
///
/// Merge rules are applied in priority order — lower <see cref="Rank"/> = higher priority.
/// This matches the original BPE paper and all HuggingFace tokenizer implementations.
/// </summary>
public readonly struct MergeRule(string left, string right, string merged, int rank) : IEquatable<MergeRule>
{
    public string Left { get; } = left;
    public string Right { get; } = right;
    public string Merged { get; } = merged;

    /// <summary>
    /// Priority rank — the order in which this rule was learned.
    /// Lower rank = learned earlier = higher frequency pair = applied first.
    /// </summary>
    public int Rank { get; } = rank;

    public bool Equals(MergeRule other) =>
        Left == other.Left && Right == other.Right;

    public override bool Equals(object? obj) => obj is MergeRule r && Equals(r);
    public override int GetHashCode() => HashCode.Combine(Left, Right);
    public override string ToString() => $"({Left}, {Right}) → {Merged} [{Rank}]";
    public static bool operator ==(MergeRule left, MergeRule right) => left.Equals(right);

    public static bool operator !=(MergeRule left, MergeRule right) => !(left == right);
}
