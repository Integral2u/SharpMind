namespace SharpMind.Core;

public static class MappingHash
{
    public static int Compute(Dictionary<string, string> mapping)
    {
        var h = new HashCode();
        foreach (var key in mapping.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            h.Add(key);
            h.Add(mapping[key]);
        }
        return h.ToHashCode();
    }
}
