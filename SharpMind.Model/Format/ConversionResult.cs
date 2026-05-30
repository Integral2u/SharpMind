using SharpMind.Core.Training;

namespace SharpMind.Model.Format;
public static partial class ModelConverter
{
    /// <summary>Conversion result with parameters and config.</summary>
    public sealed class ConversionResult
    {
        public required List<Parameter> Parameters { get; init; }
        public required SharpMindModelConfig Config { get; init; }
        public string? Warning { get; init; }
    }
}