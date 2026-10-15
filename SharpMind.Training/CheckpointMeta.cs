namespace SharpMind.Training;

/// <summary>Metadata returned when loading a checkpoint.</summary>
public sealed record CheckpointMeta
{
    public int      Step     { get; init; }
    public float    Loss     { get; init; } = float.NaN;
    public string?  Note     { get; init; }
    public DateTime SavedUtc { get; init; }

    /// <summary>
    /// Free-form key/value metadata an interrupted run can carry forward and a
    /// host can use to identify or resume it — for example a base fingerprint,
    /// recipe hash, dataset hash, tokenizer hash, <c>valLoss</c> or
    /// <c>tokensSeen</c>. Empty for checkpoints written before this field
    /// existed.
    /// </summary>
    public IReadOnlyDictionary<string, string> Extra { get; init; } =
        new Dictionary<string, string>();
}
