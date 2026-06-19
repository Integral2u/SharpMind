namespace SharpMind.Inference.Agent;

public sealed record AgentConfig
{
    public string? Name { get; init; }
    public required string SystemPrompt { get; init; }
    public float? Temperature { get; init; }
    public int? Seed { get; init; }
}
