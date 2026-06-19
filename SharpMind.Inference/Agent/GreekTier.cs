namespace SharpMind.Inference.Agent;

/// <summary>
/// Maps temperature ranges to Greek tiers and deity pools for auto-naming sub-agents.
/// Format: <c>{Deity}-{Tier}</c> — e.g. <c>Athena-Alpha</c>, <c>Hermes-Gamma</c>.
/// </summary>
public static class GreekTier
{
    private static readonly string[] Tiers = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"];

    private static readonly Dictionary<string, string[]> Deities = new()
    {
        ["Alpha"]   = ["Athena", "Astraea", "Artemis", "Andromeda", "Ariadne"],
        ["Beta"]    = ["Apollo", "Ares", "Achilles", "Ajax", "Actaeon"],
        ["Gamma"]   = ["Hermes", "Hephaestus", "Hercules", "Hector", "Helios"],
        ["Delta"]   = ["Dionysus", "Demeter", "Daedalus", "Danae", "Doris"],
        ["Epsilon"] = ["Prometheus", "Perseus", "Pandora", "Paris", "Plato"],
    };

    private static readonly Dictionary<string, float> DefaultTemps = new()
    {
        ["Alpha"]   = 0.15f,
        ["Beta"]    = 0.4f,
        ["Gamma"]   = 0.65f,
        ["Delta"]   = 0.95f,
        ["Epsilon"] = 1.2f,
    };

    /// <summary>Returns the tier letter for a given temperature.</summary>
    public static string TierForTemperature(float temperature) => temperature switch
    {
        <= 0.3f => "Alpha",
        <= 0.5f => "Beta",
        <= 0.8f => "Gamma",
        <= 1.1f => "Delta",
        _       => "Epsilon"
    };

    /// <summary>Default temperature for a given tier letter.</summary>
    public static float DefaultTemperature(string tier) =>
        DefaultTemps.TryGetValue(tier, out var t) ? t : 0.65f;

    /// <summary>
    /// Generates an auto-name in the format <c>{Deity}-{Tier}</c>.
    /// When <paramref name="temperature"/> is null, cycles through tier letters
    /// (Alpha → Beta → Gamma → Delta → Epsilon → Alpha-2 → ...).
    /// When <paramref name="temperature"/> is set, picks the corresponding tier
    /// and the first unused deity from that tier's pool.
    /// </summary>
    public static string AutoName(float? temperature, ref int unnamedCounter, HashSet<string> usedNames)
    {
        string tier;
        if (temperature.HasValue)
        {
            tier = TierForTemperature(temperature.Value);
        }
        else
        {
            tier = Tiers[unnamedCounter % Tiers.Length];
            unnamedCounter++;
        }

        var pool = Deities[tier];
        foreach (var deity in pool)
        {
            var full = $"{deity}-{tier}";
            if (!usedNames.Contains(full))
                return full;
        }

        int suffix = 2;
        while (true)
        {
            var full = $"{pool[^1]}-{tier}-{suffix}";
            if (!usedNames.Contains(full))
                return full;
            suffix++;
        }
    }

    /// <summary>Returns the default temperature for a null-temperature config based on tier cycling.</summary>
    public static float DefaultTemperatureForUnnamed(int unnamedCounter) =>
        DefaultTemperature(Tiers[unnamedCounter % Tiers.Length]);
}
