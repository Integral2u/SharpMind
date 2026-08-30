using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;

namespace SharpMind.Extensions.AI;

/// <summary>
/// Bridges MEAI <see cref="AIFunction"/> instances into SharpMind's
/// <see cref="IAgentBuilder"/> tool infrastructure.
/// <para>
/// MEAI functions are (a) registered into the native agent builder so they
/// appear in the tool loop's gate and prompt, and (b) kept here so the
/// <see cref="SharpMindChatClient"/>'s <see cref="IChatSession.ProcessToolRequest"/>
/// seam can dispatch them itself — the host owns the loop for MEAI tools while
/// native SharpMind tools defer to the session's own agent dispatch.
/// </para>
/// </summary>
internal sealed class AiFunctionToolAdapter
{
    private readonly Dictionary<string, AIFunction> _functions = new(StringComparer.Ordinal);

    /// <summary>
    /// Registers every <see cref="AIFunction"/> found in <paramref name="tools"/>
    /// with <paramref name="builder"/> using the delegate-based
    /// <see cref="IAgentBuilder.WithTool"/> overload, and remembers it so the
    /// seam can dispatch by name. Tool names and descriptions come from the
    /// <see cref="AITool"/> base class; the parameter schema is extracted via
    /// <see cref="AIFunction.AsDeclarationOnly"/>.
    /// </summary>
    public void RegisterTools(IAgentBuilder builder, IList<AITool>? tools)
    {
        if (tools is null or { Count: 0 }) return;

        foreach (var tool in tools)
        {
            if (tool is not AIFunction fn) continue;
            string name = fn.Name ?? fn.GetType().Name;
            if (!_functions.TryAdd(name, fn)) continue;

            JsonObject schema = BuildParameterSchema(fn);
            builder.WithTool(name, fn.Description ?? "", schema, args => InvokeFunction(fn, args));
        }
    }

    /// <summary>
    /// Handles a tool-call request through the <see cref="IChatSession.ProcessToolRequest"/>
    /// seam. When the tool name matches a registered MEAI function, invokes it
    /// and returns <see cref="ToolRequestResult.Handled(string?)"/>; otherwise
    /// returns <see cref="ToolRequestResult.Defer"/> so the session's native
    /// agent loop dispatches it.
    /// </summary>
    public async Task<ToolRequestResult> DispatchAsync(string toolName, JsonObject args, CancellationToken ct)
    {
        if (_functions.TryGetValue(toolName, out var fn))
            return ToolRequestResult.Handled(await InvokeFunction(fn, args));
        return ToolRequestResult.Defer();
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
