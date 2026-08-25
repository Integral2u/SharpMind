using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SharpMind.Inference.Chat;
using SharpMind.Server.Protocol;

namespace SharpMind.Server;

/// <summary>
/// OpenAI-compatible HTTP server. Call <see cref="StartAsync"/> to spin up the
/// ASP.NET host, or <see cref="BuildHost"/> to get the raw <see cref="IHost"/>
/// for advanced scenarios.
/// </summary>
public sealed class SharpMindService : IAsyncDisposable
{
    private IHost? _host;
    private bool _started;
    private string? _lastTick;

    public SharpMindService(SharpMindServerOptions options, TextWriter? output = null)
    {
        Options = options;
        Output = output ?? Console.Out;
        Progress = Output;
    }

    /// <summary>
    /// Server options. Can be modified before calling <see cref="StartAsync"/>
    /// or <see cref="StartBackgroundAsync"/>.
    /// </summary>
    public SharpMindServerOptions Options { get; }

    /// <summary>
    /// General output writer (startup messages, errors).
    /// </summary>
    public TextWriter Output { get; }

    /// <summary>
    /// Progress output writer. Defaults to <see cref="Output"/>. Set to a
    /// custom writer to control how loading progress is displayed
    /// (e.g. overwrite lines with \r).
    /// </summary>
    public TextWriter Progress { get; set; }

    /// <summary>
    /// Whether the server is currently running.
    /// </summary>
    public bool IsRunning => _started && _host != null;

    /// <summary>
    /// Build the ASP.NET host without starting it. Useful for integration testing
    /// or when the caller owns the host lifecycle.
    /// </summary>
    public IHost BuildHost()
    {
        var builder = WebApplication.CreateBuilder();

        builder.Services.AddSharpMindServer(opts =>
        {
            opts.ModelsDir = Options.ModelsDir;
            opts.Host = Options.Host;
            opts.Port = Options.Port;
            opts.DisableFileIO = Options.DisableFileIO;
            opts.DisableNetworkIO = Options.DisableNetworkIO;
        });

        var app = builder.Build();
        app.Urls.Add($"http://{Options.Host}:{Options.Port}");
        MapEndpoints(app);
        return app;
    }

    /// <summary>
    /// Load a single model by name. Useful before the host is built (e.g. to
    /// pre-warm a model at startup). Requires that <see cref="BuildHost"/>
    /// was called first.
    /// </summary>
    public async Task PreloadModelAsync(string modelId, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var app = (WebApplication)(_host ?? throw new InvalidOperationException("Host not built."));
        var modelManager = app.Services.GetRequiredService<ModelManager>();
        await modelManager.LoadAsync(modelId, progress ?? CreateProgress(), ct);
    }

    /// <summary>
    /// Start the server and block until shutdown. The caller is responsible for
    /// cancellation — e.g. wiring <see cref="Console.CancelKeyPress"/> to a
    /// <see cref="CancellationTokenSource"/> and passing the token here.
    /// Returns cleanly when <paramref name="ct"/> is cancelled.
    /// </summary>
    public async Task StartAsync(CancellationToken ct = default)
    {
        if (_started)
            throw new InvalidOperationException("Server is already running.");

        _host ??= BuildHost();

        Output.WriteLine($"[SharpMind] Models directory: {Options.ResolvedModelsDir}");
        Output.WriteLine($"[SharpMind] Listening on http://{Options.Host}:{Options.Port}");

        _started = true;

        try
        {
            await ((WebApplication)_host).RunAsync(ct);
        }
        catch (OperationCanceledException) { }
        finally
        {
            _started = false;
        }
    }

    /// <summary>
    /// Start the server in the background. The host begins listening
    /// immediately; preloading runs concurrently. Use <see cref="StopAsync"/>
    /// to shut it down later.
    /// </summary>
    public async Task StartBackgroundAsync(CancellationToken ct = default)
    {
        if (_started)
            throw new InvalidOperationException("Server is already running.");

        _host ??= BuildHost();

        Output.WriteLine($"[SharpMind] Models directory: {Options.ResolvedModelsDir}");
        Output.WriteLine($"[SharpMind] Listening on http://{Options.Host}:{Options.Port}");

        _started = true;
        _ = ((WebApplication)_host).StartAsync(ct);
    }

    /// <summary>
    /// Stop a running server.
    /// </summary>
    public async Task StopAsync(CancellationToken ct = default)
    {
        if (_host is null)
            return;

        await _host.StopAsync(ct);
        _host.Dispose();
        _host = null;
        _started = false;
    }

    public async ValueTask DisposeAsync()
    {
        if (_host is not null)
        {
            await _host.StopAsync();
            _host.Dispose();
            _host = null;
            _started = false;
        }
    }

    /// <summary>
    /// Weight-loading ticks are the only progress messages containing '%'.
    /// They are written in place (no newline) so hosts can render them as a
    /// single updating line; everything else is an ordinary status line.
    /// When stdout is piped, the leading '\r' ensures each tick flushes the
    /// previous one so the CLI can render in-place overwrites.
    /// </summary>
    private Progress<string> CreateProgress() =>
        new(msg =>
        {
            if (msg.Contains('%'))
            {
                if (msg == _lastTick)
                    return;
                _lastTick = msg;
                Output.Write($"\r[SharpMind] {msg}                          ");
                Output.Flush();
            }
            else
            {
                // If the previous message was a tick, advance past the tick
                // line so the status appears on its own line.
                if (_lastTick is not null)
                    Output.WriteLine();
                _lastTick = null;
                Output.WriteLine($"[SharpMind] {msg}");
            }
        });

    private void MapEndpoints(WebApplication app)
    {
        var modelManager = app.Services.GetRequiredService<ModelManager>();
        var sessionFactory = app.Services.GetRequiredService<SessionFactory>();

        app.MapGet("/v1/health", () => Results.Ok(new { status = "ok" }));

        // Operational extension (not part of the OpenAI spec). Schedules a
        // graceful stop after the response is written, so in-flight requests
        // can drain.
        app.MapPost("/v1/shutdown", () =>
        {
            Output.WriteLine("[SharpMind] Shutdown requested.");
            _ = Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(200);
                    await StopAsync();
                }
                catch { /* host may already be gone */ }
            });
            return Results.Ok(new { status = "stopping" });
        });

        app.MapGet("/v1/models", () =>
        {
            var models = modelManager.GetAvailableModels();
            var response = new ListModelsResponse
            {
                Data = [.. models.Select(m => OpenAiMapper.ToModelInfo(m.ModelId, m.CreatedUnix))]
            };
            return Results.Ok(response);
        });

        app.MapGet("/v1/models/loaded", () =>
        {
            var loaded = modelManager.GetLoadedModels();
            var response = new ListModelsResponse
            {
                Data = [.. loaded.Select(m =>
                {
                    var info = modelManager.GetModelInfo(m.ModelId);
                    return OpenAiMapper.ToModelInfo(m.ModelId, info?.CreatedUnix ?? 0);
                })]
            };
            return Results.Ok(response);
        });

        app.MapGet("/v1/models/{model}", (string model) =>
        {
            var info = modelManager.GetModelInfo(model);
            if (info is null)
                return Results.Json(new { error = new { message = $"Model '{model}' not found", type = "invalid_request_error" } }, statusCode: 404);
            return Results.Ok(OpenAiMapper.ToModelInfo(info.ModelId, info.CreatedUnix));
        });

        app.MapDelete("/v1/models/{model}", (string model) =>
        {
            var unloaded = modelManager.Unload(model);
            if (!unloaded)
                return Results.Json(new { error = new { message = $"Model '{model}' not found or not loaded", type = "invalid_request_error" } }, statusCode: 404);
            return Results.Ok(new DeleteModelResponse { Id = model, Deleted = true });
        });

        app.MapPost("/v1/models/{model}/load", async (string model, HttpContext httpContext) =>
        {
            var ct = httpContext.RequestAborted;
            var info = modelManager.GetModelInfo(model);
            if (info is null)
                return Results.Json(new { error = new { message = $"Model '{model}' not found", type = "invalid_request_error" } }, statusCode: 404);

            try
            {
                var loaded = await modelManager.LoadAsync(model, CreateProgress(), ct);
                if (loaded is null)
                    return Results.Json(new { error = new { message = $"Failed to load model '{model}'", type = "server_error" } }, statusCode: 500);
                return Results.Ok(new { status = "loaded", model });
            }
            catch (OperationCanceledException)
            {
                return Results.Json(new { error = new { message = "Model loading timed out", type = "timeout" } }, statusCode: 408);
            }
            catch (Exception ex)
            {
                return Results.Json(new { error = new { message = $"Failed to load model: {ex.Message}", type = "server_error" } }, statusCode: 500);
            }
        });

        app.MapPost("/v1/chat/completions", async (CreateChatCompletionRequest request, HttpContext httpContext) =>
        {
            var ct = httpContext.RequestAborted;

            if (string.IsNullOrWhiteSpace(request.Model))
                return Results.Json(new { error = new { message = "model is required", type = "invalid_request_error" } }, statusCode: 400);
            if (request.Messages.Count == 0)
                return Results.Json(new { error = new { message = "messages is required", type = "invalid_request_error" } }, statusCode: 400);

            var info = modelManager.GetModelInfo(request.Model);
            if (info is null)
                return Results.Json(new { error = new { message = $"Model '{request.Model}' not found", type = "invalid_request_error" } }, statusCode: 404);

            LoadedModel? loaded;
            try
            {
                loaded = await modelManager.LoadAsync(request.Model, CreateProgress(), ct);
                if (loaded is null)
                    return Results.Json(new { error = new { message = $"Failed to load model '{request.Model}'", type = "server_error" } }, statusCode: 500);
            }
            catch (OperationCanceledException)
            {
                return Results.Json(new { error = new { message = "Model loading timed out", type = "timeout" } }, statusCode: 408);
            }
            catch (Exception ex)
            {
                return Results.Json(new { error = new { message = $"Failed to load model: {ex.Message}", type = "server_error" } }, statusCode: 500);
            }

            IChatSession session;
            string lastUserMessage;
            try
            {
                (session, lastUserMessage) = sessionFactory.CreateSession(loaded, request);
            }
            catch (Exception ex)
            {
                modelManager.Release(request.Model);
                return Results.Json(new { error = new { message = $"Failed to create session: {ex.Message}", type = "server_error" } }, statusCode: 500);
            }

            var completionId = $"chatcmpl-{Guid.NewGuid():N}";
            var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            var isStreaming = request.Stream == true;

            try
            {
                if (isStreaming)
                {
                    await HandleStreamingResponse(httpContext.Response, session, lastUserMessage, loaded, completionId, created, request.StreamOptions, ct);
                    return Results.Empty;
                }
                else
                {
                    return await HandleNonStreamingResponse(session, lastUserMessage, loaded, completionId, created, ct);
                }
            }
            finally
            {
                modelManager.Release(request.Model);
            }
        });
    }

    private static async Task<IResult> HandleNonStreamingResponse(
        IChatSession session,
        string lastUserMessage,
        LoadedModel loaded,
        string completionId,
        long created,
        CancellationToken ct)
    {
        var sb = new StringBuilder();
        int tokenCount = 0;

        try
        {
            await foreach (var entry in session.GetResponseStreamAsync(lastUserMessage, null, ct))
            {
                if (entry.Token is { Length: > 0 } delta)
                {
                    sb.Append(delta);
                    Interlocked.Increment(ref tokenCount);
                }
            }
        }
        catch (OperationCanceledException)
        {
            return Results.Json(new { error = new { message = "Generation cancelled", type = "timeout" } }, statusCode: 408);
        }

        var usage = new CompletionUsage
        {
            PromptTokens = 0,
            CompletionTokens = tokenCount,
            TotalTokens = tokenCount
        };

        var response = OpenAiMapper.ToResponse(completionId, created, loaded.ModelId, sb.ToString(), usage);
        return Results.Ok(response);
    }

    private static async Task HandleStreamingResponse(
        HttpResponse httpResponse,
        IChatSession session,
        string lastUserMessage,
        LoadedModel loaded,
        string completionId,
        long created,
        StreamOptions? streamOptions,
        CancellationToken ct)
    {
        httpResponse.ContentType = "text/event-stream; charset=utf-8";
        httpResponse.Headers.CacheControl = "no-cache";
        httpResponse.Headers.Connection = "keep-alive";

        await using var writer = new StreamWriter(httpResponse.Body, Encoding.UTF8, leaveOpen: true);
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

        var roleChunk = OpenAiMapper.ToStreamRoleChunk(completionId, created, loaded.ModelId);
        await WriteSSE(writer, roleChunk, options);

        int tokenCount = 0;

        try
        {
            await foreach (var entry in session.GetResponseStreamAsync(lastUserMessage, null, ct))
            {
                // Skip prefill progress and other status-only entries — only
                // stream actual token content to the client.
                if (entry.Status != ChatStatus.Responding && entry.Status != ChatStatus.Thinking)
                    continue;

                if (entry.Token is { Length: > 0 } delta)
                {
                    Interlocked.Increment(ref tokenCount);
                    var chunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId, delta, null, null);
                    await WriteSSE(writer, chunk, options);
                }
            }
        }
        catch (OperationCanceledException) { }

        var finalChunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId, null, "stop", null);
        await WriteSSE(writer, finalChunk, options);

        if (streamOptions?.IncludeUsage == true)
        {
            var usageChunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId, null, null, new CompletionUsage
            {
                CompletionTokens = tokenCount,
                TotalTokens = tokenCount
            });
            await WriteSSE(writer, usageChunk, options);
        }

        await writer.WriteAsync("data: [DONE]\n\n");
        await writer.FlushAsync(ct);
    }

    private static async Task WriteSSE<T>(StreamWriter writer, T data, JsonSerializerOptions options)
    {
        var json = JsonSerializer.Serialize(data, options);
        await writer.WriteAsync($"data: {json}\n\n");
        await writer.FlushAsync();
    }
}
