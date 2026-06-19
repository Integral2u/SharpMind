namespace SharpMind.Inference.Agent;

internal sealed class Agent : IAgent
{
    public string Name { get; }
    public AgentConfig Config { get; }

    public Agent(string name, AgentConfig config)
    {
        Name = name;
        Config = config;
    }
}
