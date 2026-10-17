using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpMind.Model;

/// <summary>
/// (E2) A per-layer expert "heat" profile — how often each routed expert of each block is
/// selected for a given workload. It seeds the streaming MoE pin set so the hottest experts
/// across layer unloads start resident (pre-warmed) instead of being discovered token by
/// token. A profile is captured from a previous run's
/// <see cref="TransformerWeightsStreaming.GetExpertRouteCounts"/>, and can ship inside an
/// .SMM's metadata (key <c>smm.expert_heat</c>).
/// </summary>
public sealed class ExpertHeatProfile
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
    };

    /// <summary>Number of transformer blocks the profile covers.</summary>
    [JsonPropertyName("num_layers")]
    public int NumLayers { get; init; }

    /// <summary>Number of routed experts per block.</summary>
    [JsonPropertyName("num_experts")]
    public int NumExperts { get; init; }

    /// <summary>Row-major heat, <c>[layer * NumExperts + expert]</c>.</summary>
    [JsonPropertyName("heat")]
    public long[] Heat { get; init; } = [];

    /// <summary>True when the profile carries no usable heat.</summary>
    [JsonIgnore]
    public bool IsEmpty => NumLayers <= 0 || NumExperts <= 0 || Heat.Length == 0;

    /// <summary>Heat recorded for one expert of one layer (0 when out of range).</summary>
    public long HeatFor(int layer, int expert)
    {
        if (layer < 0 || layer >= NumLayers || expert < 0 || expert >= NumExperts) return 0;
        long index = (long)layer * NumExperts + expert;
        return index < Heat.Length ? Heat[index] : 0;
    }

    /// <summary>
    /// The <paramref name="count"/> hottest expert ids for a layer, hottest first with ties
    /// broken by id for determinism. Empty when the layer is out of range.
    /// </summary>
    public int[] TopExperts(int layer, int count)
    {
        if (layer < 0 || layer >= NumLayers || NumExperts <= 0 || count <= 0) return [];
        count = Math.Min(count, NumExperts);

        var order = new (int id, long count)[NumExperts];
        for (int e = 0; e < NumExperts; e++) order[e] = (e, HeatFor(layer, e));
        Array.Sort(order, (a, b) => b.count != a.count ? b.count.CompareTo(a.count) : a.id.CompareTo(b.id));

        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = order[i].id;
        return result;
    }

    /// <summary>
    /// Builds a profile from per-layer route counts (e.g.
    /// <see cref="TransformerWeightsStreaming.GetExpertRouteCounts"/>). Rows may vary in
    /// length; the widest row sets <see cref="NumExperts"/>.
    /// </summary>
    public static ExpertHeatProfile FromRouteCounts(IReadOnlyList<int[]>? routeCounts)
    {
        if (routeCounts is null || routeCounts.Count == 0) return new ExpertHeatProfile();

        int layers = routeCounts.Count;
        int experts = 0;
        for (int l = 0; l < layers; l++) experts = Math.Max(experts, routeCounts[l]?.Length ?? 0);

        var heat = new long[(long)layers * experts];
        for (int l = 0; l < layers; l++)
        {
            var row = routeCounts[l];
            if (row is null) continue;
            int n = Math.Min(row.Length, experts);
            for (int e = 0; e < n; e++) heat[(long)l * experts + e] = row[e];
        }
        return new ExpertHeatProfile { NumLayers = layers, NumExperts = experts, Heat = heat };
    }

    /// <summary>Creates a profile directly from a flat heat array.</summary>
    public static ExpertHeatProfile FromHeat(int numLayers, int numExperts, long[] heat)
    {
        ArgumentNullException.ThrowIfNull(heat);
        if (numLayers < 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numExperts < 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        return new ExpertHeatProfile { NumLayers = numLayers, NumExperts = numExperts, Heat = heat };
    }

    /// <summary>Serialises the profile to its compact JSON form.</summary>
    public string ToJson() => JsonSerializer.Serialize(this, JsonOpts);

    /// <summary>Parses a profile from <see cref="ToJson"/>, or <see langword="null"/> when invalid.</summary>
    public static ExpertHeatProfile? FromJson(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            return JsonSerializer.Deserialize<ExpertHeatProfile>(json, JsonOpts);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
