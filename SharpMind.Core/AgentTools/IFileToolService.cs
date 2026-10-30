namespace SharpMind.Core.AgentTools;

/// <summary>
/// Marks a tool class whose methods read or write the local file system. The agent
/// classifies such tools as File access for permission checks; this replaces matching
/// on type names.
/// </summary>
public interface IFileToolService;

/// <summary>
/// Marks a tool class whose methods make outbound network calls. The agent classifies
/// such tools as Network access for permission checks.
/// </summary>
public interface INetworkToolService;
