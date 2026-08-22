using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using SharpMind.Inference.Agent;

namespace SharpMind.Extensions.AI;

/// <summary>
/// Bridges MEAI <see cref="AIFunction"/> instances into SharpMind's
/// <see cref="IAgentBuilder"/> tool infrastructure.
/// </summary>
internal static class AiFunctionToolAdapter
{
    /// <summary>
    /// Registers every <see cref="AIFunction"/> found in
    /// <paramref name="tools"/> with <paramref name="builder"/> using the
    /// delegate-based <see cref="IAgentBuilder.WithTool"/> overload. Tool
    /// names and descriptions come from the <see cref="AITool"/> base class;
    /// the parameter schema is extracted via
    /// <see cref="AIFunction.AsDeclarationOnly"/>.
    /// </summary>
    public static void RegisterTools(IAgentBuilder builder, IList<AITool>? tools)
    {
        if (tools is null or { Count: 0 }) return;

        foreach (var tool in tools)
        {
            if (tool is AIFunction fn)
                RegisterOne(builder, fn);
        }
    }

    private static void RegisterOne(IAgentBuilder builder, AIFunction fn)
    {
        string name = fn.Name ?? fn.GetType().Name;
        string description = fn.Description ?? "";
        JsonObject schema = BuildParameterSchema(fn);

        builder.WithTool(name, description, schema, args => InvokeFunction(fn, args));
    }

    /// <summary>
    /// Builds a JSON Schema <c>parameters</c> object from the AIFunction.
    /// Uses <see cref="AIFunction.AsDeclarationOnly"/> to obtain the
    /// function's declared parameter schema. Falls back to an empty
    /// parameterless schema when unavailable.
    /// </summary>
    private static JsonObject BuildParameterSchema(AIFunction fn)
    {
        try
        {
            var declaration = fn.AsDeclarationOnly();
            if (declaration.JsonSchema.ValueKind == JsonValueKind.Object)
            {
                var raw = declaration.JsonSchema.GetRawText();
                var schemaNode = JsonNode.Parse(raw);
                if (schemaNode is JsonObject obj)
                    return obj;
            }
        }
        catch
        {
            // Fall through to default schema.
        }

        return new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject(),
            ["required"] = new JsonArray()
        };
    }

    /// <summary>
    /// Invokes a MEAI <see cref="AIFunction"/> with arguments parsed from a
    /// <see cref="JsonObject"/> and returns the result as a string.
    /// </summary>
    private static async Task<string> InvokeFunction(AIFunction fn, JsonObject args)
    {
        var functionArgs = new AIFunctionArguments();
        foreach (var (key, value) in args)
        {
            if (value is null) continue;
            functionArgs[key] = value.Deserialize<object?>();
        }

        var result = await fn.InvokeAsync(functionArgs);
        return result?.ToString() ?? string.Empty;
    }
}
