namespace SharpMind.Model.Format;

/// <summary>
/// Mapper from external weight names to SharpMind parameter names.
/// Architecture-specific implementations provide the mapping rules.
/// </summary>
public abstract class WeightMapper
{
    /// <summary>Map external weight name to SharpMind parameter name. Returns null if skipped.</summary>
    public abstract string? MapWeight(string externalName, int[] shape);
}
