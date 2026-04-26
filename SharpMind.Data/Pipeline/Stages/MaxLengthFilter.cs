namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Discards documents longer than <see cref="MaxLength"/> characters.</summary>
public sealed class MaxLengthFilter : ICleaningStage
{
    private readonly int _maxLength;

    public MaxLengthFilter(int maxLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        _maxLength = maxLength;
    }

    public string Name => $"MaxLength({_maxLength})";
    public string? Process(string document) => document.Length <= _maxLength ? document : null;
}
