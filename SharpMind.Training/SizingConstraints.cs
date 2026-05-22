namespace SharpMind.Training;

public record SizingConstraints(
    int MinHiddenDim = 16,
    int MaxHiddenDim = 256,
    int MinLayers = 1,
    int MaxLayers = 8,
    int HiddenStep = 16,
    int LayerStep = 1);
