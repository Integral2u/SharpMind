namespace SharpMind.Inference.Agent;

public sealed record AgentConfig
{
    public string? Name { get; init; }
    public required string SystemPrompt { get; init; }

    /// <summary>
    /// One line that says what the agent is for, shown to the model when it chooses a
    /// delegate. Falls back to the start of <see cref="SystemPrompt"/> when unset.
    /// </summary>
    public string? Purpose { get; init; }
    public float? Temperature { get; init; }
    public int? Seed { get; init; }
}
