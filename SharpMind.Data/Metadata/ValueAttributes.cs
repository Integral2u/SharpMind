namespace SharpMind.Data.Metadata;

/// <summary>
/// Describes a numeric constructor parameter (int or float) as a bounded,
/// stepped value the training UI can render as a spinner or slider.
/// The attribute itself is metadata only — validation that the supplied value
/// falls inside the declared range is the component's own responsibility at
/// construction time.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class MinMaxDefaultAttribute(double min, double max, double defaultValue = 0, double step = 1) : Attribute
{
    public double Min { get; } = min;
    public double Max { get; } = max;
    public double Default { get; } = defaultValue;
    public double Step { get; } = step;
}

/// <summary>
/// Supplies the wizard's default value for a constructor parameter when the
/// parameter declaration itself cannot carry one (e.g. an <c>int</c> that must
/// default to a non-zero production value, or any value that differs from the
/// C# default).
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class DefaultValueAttribute(string value) : Attribute
{
    public string Value { get; } = value;
}

/// <summary>Human-readable per-parameter help text in the wizard.</summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class TooltipAttribute(string text) : Attribute
{
    public string Text { get; } = text;
}

/// <summary>
/// Restricts a string parameter to one of a fixed set of choices, e.g. text
/// column names or enum-style option strings. The value of <see cref="Choices"/>
/// is a comma-separated list; the wizard renders a radio-group.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class ChoicesAttribute(string choices) : Attribute
{
    public string[] Choices { get; } = choices.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}