using SharpMind.Data.Metadata;
namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Lowercases all characters.</summary>
[ComponentKind("Lower Case", "Lowercases every document.")]
public sealed class LowerCase : ICleaningStage
{
    public string Name => "LowerCase";
    public string? Process(string document) => document.ToLowerInvariant();
}