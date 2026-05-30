namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Jinja <c>namespace()</c> object — allows cross-scope variable mutation via
/// <c>set ns.field = value</c>.
/// </summary>
public sealed class JinjaNamespace
{
    private readonly Dictionary<string, object?> _fields = [];
    public void Set(string k, object? v) => _fields[k] = v;
    public object? Get(string k) => _fields.TryGetValue(k, out var v) ? v : null;
}