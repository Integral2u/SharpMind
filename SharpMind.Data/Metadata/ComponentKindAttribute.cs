namespace SharpMind.Data.Metadata;

/// <summary>
/// Marks a class as a user-selectable pipeline component — either an
/// <see cref="Sources.IDataSource"/> (corpus origin) or a
/// <see cref="Pipeline.ICleaningStage"/> (document transform). Components
/// decorated with this attribute are discovered by
/// <see cref="ComponentRegistry"/> and surfaced in the training wizard's
/// source and stage-chain pickers.
/// </summary>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
public sealed class ComponentKindAttribute(string name, string description) : Attribute
{
    /// <summary>Display name shown in the picker.</summary>
    public string Name { get; } = name;

    /// <summary>One-line description shown as picker help.</summary>
    public string Description { get; } = description;
}