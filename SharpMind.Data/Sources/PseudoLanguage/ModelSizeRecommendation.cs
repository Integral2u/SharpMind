namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed record ModelSizeRecommendation
{
    public int VocabSize { get; init; }
    public int EmbeddingDim { get; init; }
    public int HiddenDim { get; init; }
    public int NumLayers { get; init; }
    public int HeadDim { get; init; }
    public int NumHeads { get; init; }
    public int FfnDim { get; init; }
    public long EstimatedParams { get; init; }
}