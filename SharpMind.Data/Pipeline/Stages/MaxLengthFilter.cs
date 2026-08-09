using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Discards documents longer than <see cref="MaxLength"/> characters.</summary>
[ComponentKind("Max Length Filter", "Drops documents longer than a fixed length.")]
public sealed class MaxLengthFilter : ICleaningStage
{
    private readonly int _maxLength;

    public MaxLengthFilter(
        [MinMaxDefault(1, 65536, 2048, 1), Tooltip("Maximum document length in characters.")] int maxLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        _maxLength = maxLength;
    }

    public string Name => $"MaxLength({_maxLength})";
    public string? Process(string document) => document.Length <= _maxLength ? document : null;
}
