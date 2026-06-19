namespace SharpMind.Inference.Agent;

internal sealed class Agent(string name, AgentConfig config) : IAgent
{
    public string Name { get; } = name;
    public AgentConfig Config { get; } = config;
}
