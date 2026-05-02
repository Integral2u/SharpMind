namespace SharpMind.Training;

/// <summary>Metadata returned when loading a checkpoint.</summary>
public sealed record CheckpointMeta
{
    public int      Step     { get; init; }
    public float    Loss     { get; init; } = float.NaN;
    public string?  Note     { get; init; }
    public DateTime SavedUtc { get; init; }
}
