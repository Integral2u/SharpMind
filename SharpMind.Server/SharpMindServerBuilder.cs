using Microsoft.Extensions.DependencyInjection;

namespace SharpMind.Server;

/// <summary>
/// Options for configuring the SharpMind HTTP server. Set via the builder
/// API (highest priority), CLI args, or appsettings.json.
/// </summary>
public sealed class SharpMindServerOptions
{
    /// <summary>
    /// Directory to scan for model files (.gguf, .smm). When empty, defaults to
    /// <c>~/SharpMind/Models</c> (resolved from the user profile at runtime).
    /// </summary>
    public string ModelsDir { get; set; } = "";

    /// <summary>
    /// HTTP port the server listens on. Defaults to 11435.
    /// </summary>
    public int Port { get; set; } = 11435;

    /// <summary>
    /// Hostname or IP address to bind to. Defaults to "localhost".
    /// Set to "0.0.0.0" to listen on all interfaces.
    /// </summary>
    public string Host { get; set; } = "localhost";

    /// <summary>
    /// Resolve the effective models directory. Empty falls back to ~/SharpMind/Models.
    /// </summary>
    public string ResolvedModelsDir =>
        string.IsNullOrWhiteSpace(ModelsDir)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "SharpMind", "Models")
            : ModelsDir;

    /// <summary>
    /// When true, tool calls that perform file system IO are denied.
    /// </summary>
    public bool DisableFileIO { get; set; }

    /// <summary>
    /// When true, tool calls that perform network IO are denied.
    /// </summary>
    public bool DisableNetworkIO { get; set; }

    /// <summary>
    /// Optional cap on KV cache length. When null, auto-caps by available
    /// memory. Set via --max-cache-len CLI flag.
    /// </summary>
    public int? MaxCacheLen { get; set; }
}

/// <summary>
/// Extension methods for registering SharpMind server services.
/// </summary>
public static class SharpMindServerExtensions
{
    /// <summary>
    /// Register SharpMind server services (ModelManager, SessionFactory, protocol types).
    /// </summary>
    public static IServiceCollection AddSharpMindServer(
        this IServiceCollection services,
        Action<SharpMindServerOptions>? configure = null)
    {
        var options = new SharpMindServerOptions();
        configure?.Invoke(options);
        services.AddSingleton(options);
        services.AddSingleton<ModelManager>();
        services.AddSingleton<SessionFactory>();
        return services;
    }
}
