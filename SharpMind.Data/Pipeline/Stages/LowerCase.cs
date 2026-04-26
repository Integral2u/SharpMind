namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Lowercases all characters.</summary>
public sealed class LowerCase : ICleaningStage
{
    public string Name => "LowerCase";
    public string? Process(string document) => document.ToLowerInvariant();
}