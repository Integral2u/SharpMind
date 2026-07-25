namespace SharpMind.Inference.Chat.PromptFormatters;

/// <summary>
/// Scoped variable environment. Child scopes see parent variables but cannot
/// pollute the parent (matching Jinja2 for-loop scoping rules).
/// </summary>
public sealed class JinjaEnv
{
    private readonly JinjaEnv? _parent;
    private readonly Dictionary<string, object?> _vars = [];

    public JinjaEnv() { _parent = null; }
    private JinjaEnv(JinjaEnv p) { _parent = p; }

    public void Set(string name, object? value) => _vars[name] = value;

    public object? Get(string name)
    {
        if (_vars.TryGetValue(name, out var v)) return v;
        return _parent?.Get(name);
    }

    public bool ContainsKey(string name)
    {
        if (_vars.ContainsKey(name)) return true;
        return _parent?.ContainsKey(name) ?? false;
    }

    /// <summary>Creates a child scope (used for for-loop iterations).</summary>
    public JinjaEnv Push() => new(this);
}