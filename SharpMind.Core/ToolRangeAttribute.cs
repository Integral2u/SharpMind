namespace SharpMind.Core;

/// <summary>
/// Optional numeric bounds for a tool parameter. They are written into the schema as
/// <c>minimum</c> and <c>maximum</c>, so the model sees the limits. The binder does not
/// enforce them; the tool should validate its own input.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class ToolRangeAttribute : Attribute
{
    /// <summary>Inclusive lower bound. Used only when <see cref="HasMinimum"/> is true.</summary>
    public double Minimum { get; init; }
    /// <summary>Inclusive upper bound. Used only when <see cref="HasMaximum"/> is true.</summary>
    public double Maximum { get; init; }
    public bool HasMinimum { get; init; }
    public bool HasMaximum { get; init; }
}
