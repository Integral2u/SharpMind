using SharpMind.Inference.Agent;
using System.Text.Json.Nodes;

namespace SharpMind.Inference.Chat;

/// <summary>
/// <see cref="HttpMessageHandler"/> decorator that gates every outbound HTTP request
/// through <see cref="IoPermissionCheck"/> while a tool call is in flight.
/// <para>
/// Pass the <see cref="HttpClient"/> built from this handler wherever HTTP calls
/// are needed. ChatSession enables and disables it around each
/// <see cref="IAgentBuilder.CallToolAsync"/> call.
/// </para>
/// </summary>
public sealed class InterceptingNetworkHandler : DelegatingHandler
{
    private IoPermissionCheck? _check;
    private string _currentTool = string.Empty;
    private JsonObject _currentArgs = [];

    public InterceptingNetworkHandler() : base(new HttpClientHandler()) { }

    public InterceptingNetworkHandler(HttpMessageHandler innerHandler) : base(innerHandler)
    {
    }

    // Internal activation (ChatSession only)

    internal void Activate(string toolName, JsonObject args, IoPermissionCheck check)
    {
        _currentTool = toolName;
        _currentArgs = args;
        _check = check;
    }

    internal void Deactivate()
    {
        _check = null;
        _currentTool = string.Empty;
        _currentArgs = [];
    }

    // HttpMessageHandler override

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken ct)
    {
        if (_check is not null)
        {
            var resource = request.RequestUri?.ToString() ?? "(unknown)";
            bool ok = await _check(_currentTool, ToolCategory.Network, resource, _currentArgs);
            if (!ok)
                throw new HttpRequestException(
                    $"Tool '{_currentTool}' was denied network access to '{resource}'.");
        }
        return await base.SendAsync(request, ct);
    }
}
