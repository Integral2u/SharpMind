namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Discards documents shorter than <see cref="MinLength"/> characters.</summary>
public sealed class MinLengthFilter : ICleaningStage
{
    private readonly int _minLength;

    public MinLengthFilter(int minLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minLength);
        _minLength = minLength;
    }

    public string Name => $"MinLength({_minLength})";
    public string? Process(string document) => document.Length >= _minLength ? document : null;
}