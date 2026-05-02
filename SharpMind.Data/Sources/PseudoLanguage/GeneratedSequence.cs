namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed record GeneratedSequence
{
    public required int[] TokenIds { get; init; }
    public required string RawText { get; init; }
    public required int[] GroundTruthIds { get; init; }
    public required string GroundTruthText { get; init; }
    public required ComplexityLevel Complexity { get; init; }
}
