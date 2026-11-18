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

    /// <summary>
    /// How many chat completions may generate at the same time. Decoding is not
    /// free concurrency: each admitted request owns a KV cache and a native
    /// workspace, and the CPU kernels already use every core, so the default is
    /// one. Raise it only when the model is small enough that decode leaves
    /// cores idle.
    /// </summary>
    public int MaxConcurrentGenerations { get; set; } = 1;

    /// <summary>
    /// How many arrival requests may wait behind the concurrency cap. Beyond
    /// this the server answers 429 instead of parking the caller, so overload
    /// fails fast rather than becoming a timeout for everyone. Zero means
    /// "reject as soon as the cap is busy".
    /// </summary>
    public int MaxQueuedGenerations { get; set; } = 32;

    /// <summary>
    /// Budget in MB for the cross-request KV prefix cache: 0 disables it. A chat
    /// request restates its whole history, so this keeps a finished request's KV
    /// and hands it to the next request whose history extends it. Budgeted in
    /// bytes of KV rather than entries, because one long conversation can dwarf a
    /// hundred short ones.
    /// </summary>
    public int PrefixCacheMB { get; set; }

    /// <summary>
    /// How many <c>/v1/completions</c> sequences one model decodes together.
    /// All requests for a model share a single batcher, so this is the width of
    /// that batch, not a per-request setting. The per-model KV cache budget is
    /// divided between the rows. 1 still works but gains nothing.
    /// </summary>
    public int CompletionBatchSize { get; set; } = 8;

    /// <summary>
    /// Optional bearer/API key. When set, the destructive operational
    /// endpoints (<c>POST /v1/shutdown</c>, <c>DELETE /v1/models/{model}</c>)
    /// require it (<c>Authorization: Bearer &lt;key&gt;</c> or
    /// <c>X-Api-Key: &lt;key&gt;</c>). Leave empty to keep them open.
    /// Set via --api-key CLI flag.
    /// </summary>
    public string ApiKey { get; set; } = "";
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
        return services.AddSharpMindServer(options);
    }

    /// <summary>
    /// Register SharpMind server services with an existing options instance.
    /// </summary>
    public static IServiceCollection AddSharpMindServer(
        this IServiceCollection services,
        SharpMindServerOptions options)
    {
        services.AddSingleton(options);
        services.AddSingleton<ModelManager>();
        services.AddSingleton<ModelLeaseService>();
        services.AddSingleton<BatchService>();
        services.AddSingleton(_ => new PrefixCache(options.PrefixCacheMB * 1024L * 1024L));
        services.AddSingleton<SessionFactory>();
        services.AddSingleton(_ => new GenerationGate(
            options.MaxConcurrentGenerations, options.MaxQueuedGenerations));
        return services;
    }
}
