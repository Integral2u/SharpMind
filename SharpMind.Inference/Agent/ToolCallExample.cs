using System.Text;
using System.Text.Json.Nodes;
using SharpMind.Inference.Chat;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Builds one tool-call example in the session's active format. Shared by the
/// agent prompt (which teaches the shape) and by the recovery feedback (which
/// shows a model the corrected call after a binding error).
/// </summary>
internal static class ToolCallExample
{
    /// <summary>Formats <paramref name="name"/> and <paramref name="args"/> exactly as the prompt teaches.</summary>
    public static string Build(ToolCallFormat format, string name, JsonObject args)
    {
        if (format == ToolCallFormat.QwenXml)
        {
            var sb = new StringBuilder();
            sb.AppendLine("<tool_call>");
            sb.AppendLine($"<function={name}>");
            foreach (var (key, value) in args)
            {
                sb.AppendLine($"<parameter={key}>");
                sb.AppendLine(ValueText(value));
                sb.AppendLine("</parameter>");
            }
            sb.Append("</function>\n</tool_call>");
            return sb.ToString();
        }

        string argKey = format == ToolCallFormat.Gemma ? "args" : "arguments";
        string json = new JsonObject { ["name"] = name, [argKey] = args.DeepClone() }.ToJsonString();
        return format switch
        {
            ToolCallFormat.Mistral => "[TOOL_CALLS]" + json,
            ToolCallFormat.SharpMind => "<tool_call>" + json + "</tool_call>",
            _ => json
        };
    }

    /// <summary>
    /// Arguments for a prompt example: every required argument (up to three)
    /// and at most one optional one, so the example stays short.
    /// </summary>
    public static JsonObject ExampleArguments(JsonObject tool)
    {
        var args = new JsonObject();
        if (tool["parameters"] is not JsonObject pars || pars["properties"] is not JsonObject props)
            return args;

        var required = RequiredNames(tool);
        var chosen = props.Where(p => required.Contains(p.Key)).Take(3)
            .Concat(props.Where(p => !required.Contains(p.Key)).Take(1))
            .ToList();
        foreach (var (name, schema) in chosen)
            args[name] = SampleValue(name, schema as JsonObject);
        return args;
    }

    /// <summary>
    /// Fills in sample values for required arguments the call omitted, keeping
    /// every supplied argument as the model wrote it. Used to show a corrected
    /// call after a missing-argument error.
    /// </summary>
    public static JsonObject FillRequired(JsonObject tool, JsonObject supplied)
    {
        var merged = supplied.DeepClone().AsObject();
        var props = tool["parameters"]?["properties"] as JsonObject;
        foreach (var required in RequiredNames(tool))
        {
            if (HasArgument(merged, required)) continue;
            merged[required] = SampleValue(required, props?[required] as JsonObject);
        }
        return merged;
    }

    public static List<string> RequiredNames(JsonObject tool)
    {
        var names = new List<string>();
        if (tool["parameters"] is JsonObject pars && pars["required"] is JsonArray req)
            foreach (var r in req)
                if (r?.GetValue<string>() is { Length: > 0 } n)
                    names.Add(n);
        return names;
    }

    private static bool HasArgument(JsonObject args, string name)
    {
        var normalized = ToolIdentifier.Normalize(name);
        foreach (var kv in args)
            if (ToolIdentifier.Normalize(kv.Key) == normalized)
                return true;
        return false;
    }

    public static JsonNode SampleValue(string name, JsonObject? schema)
    {
        if (schema?["enum"] is JsonArray en && en.Count > 0 && en[0] is not null)
            return en[0]!.DeepClone();

        string type = schema?["type"]?.GetValue<string>() ?? "string";
        switch (type)
        {
            case "integer":
                return JsonValue.Create(3)!;
            case "number":
                return JsonValue.Create(2.5)!;
            case "boolean":
                return JsonValue.Create(true)!;
            case "array":
                bool choices = name.Contains("option", StringComparison.OrdinalIgnoreCase)
                    || name.Contains("choice", StringComparison.OrdinalIgnoreCase);
                return choices ? new JsonArray("red", "green", "blue") : new JsonArray(SampleText(name));
            default:
                return JsonValue.Create(SampleText(name))!;
        }
    }

    private static string SampleText(string argName)
    {
        string n = argName.ToLowerInvariant();
        if (n.Contains("city") || n.Contains("location") || n.Contains("place")) return "Paris";
        if (n.Contains("path") || n.Contains("file")) return "notes.txt";
        if (n.Contains("dir") || n.Contains("folder")) return ".";
        if (n.Contains("url")) return "https://example.com";
        if (n.Contains("query") || n.Contains("pattern") || n.Contains("search")) return "meeting";
        if (n.Contains("content") || n.Contains("text") || n.Contains("message")) return "Hello";
        if (n.EndsWith("name", StringComparison.Ordinal)) return "Alex";
        return "example";
    }

    private static string ValueText(JsonNode? value) =>
        value is JsonValue jv && jv.TryGetValue<string>(out var s) ? s : value?.ToJsonString() ?? "";
}
