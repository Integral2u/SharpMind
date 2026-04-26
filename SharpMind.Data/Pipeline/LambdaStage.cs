namespace SharpMind.Data.Pipeline;
// ─────────────────────────────────────────────────────────────────────────────
// Lambda stage — lets callers write inline transforms without a class
// ─────────────────────────────────────────────────────────────────────────────
internal sealed class LambdaStage : ICleaningStage
{
    private readonly Func<string, string?> _fn;
    public string Name { get; }
    internal LambdaStage(string name, Func<string, string?> fn) { Name = name; _fn = fn; }
    public string? Process(string doc) => _fn(doc);
}