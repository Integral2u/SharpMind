namespace SharpMind.Inference.Agent;

public interface IAgent
{
    string Name { get; }
    AgentConfig Config { get; }
}
