using SharpMind.Data.Metadata;

namespace SharpMind.CUI.App;

/// <summary>
/// One persisted pipeline component (a source or a stage): the full assembly
/// qualified type name plus the wizard-supplied constructor values. Rebuilds an
/// instance via <see cref="ComponentRegistry.Build{T}"/> from
/// <see cref="ComponentDescriptor"/> lookup.
/// </summary>
public sealed class JobComponent
{
    /// <summary>Display name from metadata, for the chain editor list.</summary>
    public string? DisplayName { get; set; }

    /// <summary>Assembly-qualified type name used to resolve the component descriptor.</summary>
    public required string TypeName { get; set; }

    /// <summary>Parameter name → string value as entered/reflected by the wizard.</summary>
    public Dictionary<string, string> Args { get; set; } = new(StringComparer.Ordinal);
}

/// <summary>
/// A configured data source plus the cleaning stages applied to its own stream,
/// before the streams are merged and the global stages run.
/// </summary>
public sealed class JobSource
{
    public required JobComponent Component { get; set; }

    /// <summary>Per-source stages run on this source's documents before merging.</summary>
    public List<JobComponent> Stages { get; set; } = [];

    public string? DisplayName => Component.DisplayName;
}