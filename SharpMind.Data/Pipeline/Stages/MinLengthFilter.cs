using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Discards documents shorter than <see cref="MinLength"/> characters.</summary>
[ComponentKind("Min Length Filter", "Drops documents shorter than a fixed length.")]
public sealed class MinLengthFilter : ICleaningStage
{
    private readonly int _minLength;

    public MinLengthFilter(
        [MinMaxDefault(1, 65536, 8, 1), Tooltip("Minimum document length in characters.")] int minLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minLength);
        _minLength = minLength;
    }

    public string Name => $"MinLength({_minLength})";
    public string? Process(string document) => document.Length >= _minLength ? document : null;
}