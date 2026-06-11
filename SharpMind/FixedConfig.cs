namespace SharpMind;

/// <summary>
/// Fixed configuration - must match trained model weights.
/// Dimensions and activation functions trained into the model.
/// </summary>
public record FixedConfig
{
    public int VocabSize { get; init; } = 32000;
    public int HiddenDim { get; init; } = 2048;
    public int NumLayers { get; init; } = 1;
    public int NumHeads { get; init; } = 32;
    public int NumKvHeads { get; init; } = 4;
    public int FfnDim { get; init; } = 5632;
    public int MaxSeqLen { get; init; } = 2048;
    public float RopeTheta { get; init; } = 10000f;
    
    public ActivationKind Activation { get; init; } = ActivationKind.SiLU;
    public GateKind Gate { get; init; } = GateKind.SwiGLU;
    public FfnKind Ffn { get; init; } = FfnKind.Gated;
    
    public static FixedConfig Default => new();
}
