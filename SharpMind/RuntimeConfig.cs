namespace SharpMind;

/// <summary>
/// Runtime configuration - can be changed without affecting model.
/// Performance and execution settings.
/// </summary>
public record RuntimeConfig
{
    public AttentionKind Attention { get; init; } = AttentionKind.GQA;
    public NormKind Norm { get; init; } = NormKind.RMSNorm;
    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;
    
    public static RuntimeConfig Default => new();
}