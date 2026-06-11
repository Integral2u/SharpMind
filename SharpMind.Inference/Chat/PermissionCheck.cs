using SharpMind.Inference.Agent;
using System.Text.Json.Nodes;
namespace SharpMind.Inference.Chat;
/// <summary>
/// Invoked by an interceptor when a tool attempts IO.
/// Returns true when the access is permitted, false to block it.
/// </summary>
public delegate Task<bool> IoPermissionCheck(string toolName, ToolCategory category, string resource, JsonObject arguments);
