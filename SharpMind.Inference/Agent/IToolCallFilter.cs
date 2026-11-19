using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Optional gate in front of tool dispatch. A host that wants to inspect a
/// model's tool call before anything runs supplies one; the engine does not
/// depend on any implementation and works unchanged without it.
/// </summary>
public interface IToolCallFilter
{
    /// <summary>
    /// Returns null to allow the call, or a refusal message. The message is
    /// returned to the model as the tool result, so it can adapt instead of the
    /// turn failing.
    /// </summary>
    Task<string?> ReviewAsync(
        string tool,
        JsonObject arguments,
        CancellationToken cancellationToken = default);
}
