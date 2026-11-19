using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Grammar;
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
        // Cache it. This used to build and return without assigning _host, so
        // PreloadModelAsync — which reads _host — threw "Host not built" even
        // when the caller had just called BuildHost(), and StartAsync then built
        // a second host through its own `_host ??=`.
        if (_host is not null) return _host;

        var builder = WebApplication.CreateBuilder();

        // Register this instance rather than copying fields into a new one: the
        // copy skipped MaxCacheLen, so --max-cache-len was accepted and ignored.
        builder.Services.AddSharpMindServer(Options);

        var app = builder.Build();
        app.Urls.Add($"http://{Options.Host}:{Options.Port}");
        MapEndpoints(app);
        _host = app;
        return app;
    }

    /// <summary>
    /// Load a single model by name at startup, e.g. to pre-warm it before the
    /// server begins listening. Calls <see cref="BuildHost"/> if it has not run
    /// yet. Returns false when <paramref name="modelId"/> is not in the models
    /// directory, so a caller can report an unknown id instead of assuming success.
    /// </summary>
    public async Task<bool> PreloadModelAsync(string modelId, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        var app = (WebApplication)(_host ??= BuildHost());
        var modelManager = app.Services.GetRequiredService<ModelManager>();
        return await modelManager.PreloadAsync(modelId, progress ?? CreateProgress(), ct);
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
        _ = ((WebApplication)_host).StartAsync(ct).ContinueWith(task =>
        {
            if (!task.IsFaulted) return;
            _started = false;
            Exception error = task.Exception?.GetBaseException() ?? new InvalidOperationException("Server failed to start.");
            Output.WriteLine($"[SharpMind] Failed to start server: {error.Message}");
        }, TaskScheduler.Default);
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
        var options = app.Services.GetRequiredService<SharpMindServerOptions>();
        var gate = app.Services.GetRequiredService<GenerationGate>();
        var prefixCache = app.Services.GetRequiredService<PrefixCache>();
        var modelLeases = app.Services.GetRequiredService<ModelLeaseService>();
        var batchService = app.Services.GetRequiredService<BatchService>();

        // Capacity is part of health: a server that is queueing is up, but a
        // caller deciding whether to send work wants to know.
        app.MapGet("/v1/health", () =>
        {
            var batch = batchService.Snapshot();
            return Results.Ok(new
            {
                status = "ok",
                generating = gate.Active,
                queued = gate.Queued,
                maxGenerating = gate.MaxConcurrent,
                maxQueued = gate.MaxQueued,
                prefixCache = new
                {
                    enabled = prefixCache.Enabled,
                    entries = prefixCache.Count,
                    bytes = prefixCache.Bytes,
                    hits = prefixCache.Hits,
                    misses = prefixCache.Misses,
                    reusedTokens = prefixCache.ReusedTokens,
                },
                batching = new
                {
                    models = batch.Models,
                    steps = batch.Steps,
                    tokens = batch.Tokens,
                    tokensPerSecond = batch.TokensPerSecond,
                    averageBatchSize = batch.AverageBatchSize,
                },
            });
        });

        // Operational extension (not part of the OpenAI spec). Schedules a
        // graceful stop after the response is written, so in-flight requests
        // can drain. Gated by API key when one is configured — an unauthenticated
        // shutdown endpoint on a shared host lets anyone take the server down.
        app.MapPost("/v1/shutdown", (HttpContext httpContext) =>
        {
            if (!IsAuthorized(options, httpContext))
                return Results.Json(new { error = new { message = "Unauthorized", type = "auth_error" } }, statusCode: 401);
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

        app.MapDelete("/v1/models/{model}", (string model, HttpContext httpContext) =>
        {
            if (!IsAuthorized(options, httpContext))
                return Results.Json(new { error = new { message = "Unauthorized", type = "auth_error" } }, statusCode: 401);
            var unloaded = modelManager.Unload(model);
            if (!unloaded)
            {
                // A loaded model that refused to unload is still in use by an
                // in-flight request, not missing — report that distinctly.
                var inUse = modelManager.GetLoaded(model) is not null;
                string message = inUse
                    ? $"Model '{model}' is in use; unload while no request holds it"
                    : $"Model '{model}' not found or not loaded";
                return Results.Json(new { error = new { message, type = inUse ? "resource_in_use" : "invalid_request_error" } }, statusCode: inUse ? 409 : 404);
            }
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
                // LoadAsync holds a reference. Models are KeepAlive, so releasing
                // leaves it resident-but-idle, which DELETE /v1/models/{model}
                // can still unload.
                modelManager.Release(model);
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

            // Validate response_format up front: a malformed or unsupported
            // format is a client error (400), and resolving it before the
            // (expensive) model load beats failing after.
            try
            {
                OpenAiMapper.ResolveGrammar(request);
            }
            catch (JsonSchemaException ex)
            {
                return Results.Json(new { error = new { message = ex.Message, type = "invalid_request_error" } }, statusCode: 400);
            }

            // Admission control before the model load: an overloaded server must
            // not start loads it cannot serve. A request past the cap waits in
            // line; past the queue it is refused now rather than timing out.
            // The lease is released when the HTTP response is disposed, which for
            // a streamed completion is when the last chunk has been written.
            GenerationGate.Lease lease;
            try
            {
                lease = await gate.EnterAsync(ct);
            }
            catch (GenerationQueueFullException ex)
            {
                httpContext.Response.Headers.RetryAfter = "1";
                return Results.Json(new { error = new { message = ex.Message, type = "rate_limit_error" } }, statusCode: 429);
            }
            httpContext.Response.RegisterForDispose(lease);

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

            // One transformer, many engines: chat generation must not overlap the
            // batched completion generator's forward passes (the transformer keeps
            // activations in instance fields). Held for the whole response — the
            // session prefills while creating, so it is taken before that.
            //
            // Batched chat is the exception: its generation runs on the batcher's
            // scheduler thread, which takes the same lease for each forward pass.
            // Holding it here as well would deadlock (the scheduler could never
            // run the step that finishes this request).
            bool batchedChat = options.BatchChatCompletions;
            ModelLeaseService.Lease? modelLease = null;
            if (!batchedChat)
            {
                try
                {
                    modelLease = await modelLeases.AcquireAsync(request.Model, ct);
                }
                catch (OperationCanceledException)
                {
                    modelManager.Release(request.Model);
                    return Results.Json(new { error = new { message = "Request cancelled while waiting for the model", type = "timeout" } }, statusCode: 408);
                }
            }

            IChatSession session;
            string lastUserMessage;
            bool continueFromHistory;
            try
            {
                (session, lastUserMessage, continueFromHistory) = await sessionFactory.CreateSessionAsync(loaded, request);
            }
            catch (Exception ex)
            {
                modelLease?.Dispose();
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
                    await HandleStreamingResponse(httpContext.Response, session, lastUserMessage, continueFromHistory, loaded, completionId, created, request.StreamOptions, ct);
                    return Results.Empty;
                }
                else
                {
                    return await HandleNonStreamingResponse(session, lastUserMessage, continueFromHistory, loaded, completionId, created, ct);
                }
            }
            finally
            {
                // Keep this turn's KV for the next turn of the same conversation.
                // Taken before Dispose: a disposed session cannot snapshot. A
                // batched session has no KV of its own, so there is nothing to keep.
                if (prefixCache.Enabled && !batchedChat)
                    prefixCache.Put(request.Model, session.GetSnapshot());

                // The session owns its generator, KV caches and a native-memory
                // workspace (≥100 MiB, no finalizer). Only the model reference was
                // released here, so every completion leaked its workspace for good.
                // Dispose the session before Release: Release may unload the model
                // at ref count zero, and the session must not outlive it.
                await session.DisposeAsync();
                modelLease?.Dispose();
                modelManager.Release(request.Model);
            }
        });

        // ── POST /v1/completions — batched, stateless text completions ───────
        //
        // Every request for a model feeds the same BatchedGenerator, so N callers
        // share one forward pass per token instead of N of them. No chat history,
        // tools or prefix cache: the prompt is tokenised, enqueued and streamed
        // back. Admission is the batcher's own queue (BatchQueueFullException ->
        // 429), not the chat GenerationGate, because the batcher bounds memory
        // by rows rather than by concurrent requests.
        app.MapPost("/v1/completions", async (CreateCompletionRequest request, HttpContext httpContext) =>
        {
            var ct = httpContext.RequestAborted;

            if (string.IsNullOrWhiteSpace(request.Model))
                return ErrorResult("model is required", 400);
            if (request.Prompts.Count == 0 || request.Prompts.Any(string.IsNullOrEmpty))
                return ErrorResult("prompt is required", 400);
            if (request.Suffix is not null)
                return ErrorResult("suffix is not supported", 400);
            if (request.Echo == true)
                return ErrorResult("echo is not supported", 400);

            int n = Math.Clamp(request.N ?? 1, 1, 8);
            int totalChoices = request.Prompts.Count * n;
            if (totalChoices > 32)
                return ErrorResult("at most 32 completions may be requested at once", 400);

            if (modelManager.GetModelInfo(request.Model) is null)
                return ErrorResult($"Model '{request.Model}' not found", 404);

            LoadedModel? loaded;
            try
            {
                loaded = await modelManager.LoadAsync(request.Model, CreateProgress(), ct);
                if (loaded is null)
                    return ErrorResult($"Failed to load model '{request.Model}'", 500, "server_error");
            }
            catch (OperationCanceledException)
            {
                return ErrorResult("Model loading timed out", 408, "timeout");
            }
            catch (Exception ex)
            {
                return ErrorResult($"Failed to load model: {ex.Message}", 500, "server_error");
            }

            var sequences = new List<(int Index, int PromptTokens, BatchedSequence Sequence)>(totalChoices);
            try
            {
                BatchedGenerator batcher;
                try
                {
                    batcher = batchService.Get(loaded);
                }
                catch (Exception ex)
                {
                    return ErrorResult($"Failed to start batching: {ex.Message}", 500, "server_error");
                }

                var (sampling, generation) = OpenAiMapper.ToCompletionOptions(request, loaded.Tokenizer);
                generation = generation with
                {
                    MtpDraft = request.Mtp ?? (request.MtpTokens is not null || Options.MtpDraft),
                    MtpDraftTokens = request.MtpTokens is { } mtpTokens
                        ? Math.Clamp(mtpTokens, 1, GenerationConfig.MaxMtpDraftTokens)
                        : Options.MtpDraftTokens,
                };
                bool addBos = loaded.Tokenizer.BosId >= 0;
                int index = 0;

                try
                {
                    foreach (string prompt in request.Prompts)
                    {
                        int[] promptIds = loaded.Tokenizer.Encode(prompt, addBos, addEos: false);
                        for (int rep = 0; rep < n; rep++)
                        {
                            // n > 1 means independent draws: a pinned seed on the
                            // request must not turn every choice into a copy.
                            var choiceSampling = rep == 0
                                ? sampling
                                : sampling with { Seed = sampling.Seed + rep };

                            var sequence = await batcher.EnqueueTokensAsync(promptIds, choiceSampling, generation, ct);
                            sequences.Add((index++, promptIds.Length, sequence));
                        }
                    }
                }
                catch (ArgumentException ex)
                {
                    await DisposeSequencesAsync(sequences);
                    return ErrorResult(ex.Message, 400);
                }
                catch (BatchQueueFullException ex)
                {
                    await DisposeSequencesAsync(sequences);
                    httpContext.Response.Headers.RetryAfter = "1";
                    return ErrorResult(ex.Message, 429, "rate_limit_error");
                }
                catch (OperationCanceledException)
                {
                    await DisposeSequencesAsync(sequences);
                    return Results.Empty;
                }

                var completionId = $"cmpl-{Guid.NewGuid():N}";
                long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                int promptTokens = sequences.Sum(s => s.PromptTokens);

                if (request.Stream == true)
                {
                    await StreamCompletionsAsync(httpContext, sequences, loaded, completionId, created,
                        promptTokens, request.StreamOptions, ct);
                    return Results.Empty;
                }

                try
                {
                    await Task.WhenAll(sequences.Select(s => s.Sequence.WhenCompleted)).WaitAsync(ct);
                }
                catch (OperationCanceledException)
                {
                    return Results.Empty;
                }
                catch (Exception ex)
                {
                    return ErrorResult($"Generation failed: {ex.Message}", 500, "server_error");
                }

                var choices = new List<CompletionChoice>(sequences.Count);
                int completionTokens = 0;
                foreach (var (choiceIndex, _, sequence) in sequences)
                {
                    completionTokens += sequence.GeneratedIds.Count;
                    choices.Add(new CompletionChoice
                    {
                        Index = choiceIndex,
                        Text = sequence.Text,
                        FinishReason = OpenAiMapper.ToCompletionFinishReason(sequence.FinishReason),
                    });
                }

                return Results.Ok(new CreateCompletionResponse
                {
                    Id = completionId,
                    Created = created,
                    Model = loaded.ModelId,
                    Choices = choices,
                    Usage = new CompletionUsage
                    {
                        PromptTokens = promptTokens,
                        CompletionTokens = completionTokens,
                        TotalTokens = promptTokens + completionTokens,
                    },
                });
            }
            finally
            {
                // Already-completed sequences are cheap no-ops; in-flight ones
                // (client gone, error) release their rows at the next step. Wait
                // for that before releasing the model, so an unload cannot
                // dispose the transformer mid-forward.
                var released = new List<Task>(sequences.Count);
                foreach (var (_, _, sequence) in sequences)
                {
                    await sequence.DisposeAsync();
                    released.Add(sequence.WhenReleased);
                }
                await Task.WhenAll(released);
                modelManager.Release(request.Model);
            }
        });
    }

    private static IResult ErrorResult(string message, int statusCode, string type = "invalid_request_error")
        => Results.Json(new { error = new { message, type } }, statusCode: statusCode);

    private static async Task DisposeSequencesAsync(IEnumerable<(int Index, int PromptTokens, BatchedSequence Sequence)> sequences)
    {
        var pending = new List<Task>();
        foreach (var (_, _, sequence) in sequences)
        {
            await sequence.DisposeAsync();
            // The row must be free before the caller releases the model: an
            // unload between release and the batcher's next step would dispose
            // the transformer under a forward pass.
            pending.Add(sequence.WhenReleased);
        }

        await Task.WhenAll(pending);
    }

    private static async Task StreamCompletionsAsync(
        HttpContext httpContext,
        List<(int Index, int PromptTokens, BatchedSequence Sequence)> sequences,
        LoadedModel loaded,
        string completionId,
        long created,
        int promptTokens,
        StreamOptions? streamOptions,
        CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream; charset=utf-8";
        httpContext.Response.Headers.CacheControl = "no-cache";
        httpContext.Response.Headers.Connection = "keep-alive";

        await using var writer = new StreamWriter(httpContext.Response.Body, Encoding.UTF8, leaveOpen: true);
        var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        using var writeLock = new SemaphoreSlim(1, 1);
        long completionTokens = 0;

        async Task WriteChunkAsync(CompletionChoice choice)
        {
            await writeLock.WaitAsync(ct);
            try
            {
                await WriteSSE(writer, new CreateCompletionStreamResponse
                {
                    Id = completionId,
                    Created = created,
                    Model = loaded.ModelId,
                    Choices = [choice],
                }, options);
            }
            finally
            {
                writeLock.Release();
            }
        }

        try
        {
            var writers = sequences.Select(async entry =>
            {
                try
                {
                    await foreach (string fragment in entry.Sequence.StreamAsync(ct))
                        await WriteChunkAsync(new CompletionChoice { Index = entry.Index, Text = fragment });

                    Interlocked.Add(ref completionTokens, entry.Sequence.GeneratedIds.Count);
                    await WriteChunkAsync(new CompletionChoice
                    {
                        Index = entry.Index,
                        Text = "",
                        FinishReason = OpenAiMapper.ToCompletionFinishReason(entry.Sequence.FinishReason),
                    });
                }
                catch (OperationCanceledException) { }
                catch (Exception)
                {
                    // Bytes may already be on the wire; the final [DONE] closes
                    // the stream and the per-choice finish chunk is skipped.
                }
            });

            await Task.WhenAll(writers);

            if (streamOptions?.IncludeUsage == true)
            {
                await writeLock.WaitAsync(ct);
                try
                {
                    await WriteSSE(writer, new CreateCompletionStreamResponse
                    {
                        Id = completionId,
                        Created = created,
                        Model = loaded.ModelId,
                        Usage = new CompletionUsage
                        {
                            PromptTokens = promptTokens,
                            CompletionTokens = (int)Interlocked.Read(ref completionTokens),
                            TotalTokens = promptTokens + (int)Interlocked.Read(ref completionTokens),
                        },
                    }, options);
                }
                finally
                {
                    writeLock.Release();
                }
            }
        }
        finally
        {
            // A client that walked away has nothing to receive; never let the
            // terminating frame turn into an unhandled response exception.
            try
            {
                await writer.WriteAsync("data: [DONE]\n\n");
                await writer.FlushAsync(CancellationToken.None);
            }
            catch (IOException) { }
            catch (ObjectDisposedException) { }
        }
    }

    private static bool IsAuthorized(SharpMindServerOptions options, HttpContext httpContext)
    {
        if (string.IsNullOrEmpty(options.ApiKey)) return true;

        string provided = httpContext.Request.Headers.Authorization.ToString();
        if (provided.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            return provided.Length > "Bearer ".Length
                   && ConstantTimeEquals(provided["Bearer ".Length..], options.ApiKey);
        return ConstantTimeEquals(httpContext.Request.Headers["X-Api-Key"].ToString(), options.ApiKey);
    }

    private static bool ConstantTimeEquals(string provided, string expected)
    {
        Span<byte> providedHash = stackalloc byte[32];
        Span<byte> expectedHash = stackalloc byte[32];
        SHA256.HashData(Encoding.UTF8.GetBytes(provided), providedHash);
        SHA256.HashData(Encoding.UTF8.GetBytes(expected), expectedHash);
        return CryptographicOperations.FixedTimeEquals(providedHash, expectedHash);
    }

    private static async Task<IResult> HandleNonStreamingResponse(
        IChatSession session,
        string lastUserMessage,
        bool continueFromHistory,
        LoadedModel loaded,
        string completionId,
        long created,
        CancellationToken ct)
    {
        var sb = new StringBuilder();
        var toolCalls = new List<ChatCompletionToolCall>();
        int tokenCount = 0;

        try
        {
            var entries = continueFromHistory
                ? session.ContinueResponseAsync(null, ct)
                : session.GetResponseStreamAsync(lastUserMessage, null, ct);
            await foreach (var entry in entries)
            {
                if (entry.Status == ChatStatus.ToolCall && entry.ToolCall is not null)
                {
                    toolCalls.Add(OpenAiMapper.ToToolCall(entry.ToolCall, $"call_{Guid.NewGuid():N}"));
                    continue;
                }

                // Status-only entries carry their status text in Token; appending
                // every non-empty Token put "Prefilling 50.25%..." in the completion.
                // The streaming path below filtered these out, this one did not.
                if (!OpenAiMapper.IsContent(entry)) continue;

                sb.Append(entry.Token);
                Interlocked.Increment(ref tokenCount);
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

        var response = OpenAiMapper.ToResponse(completionId, created, loaded.ModelId, sb.ToString(), usage, toolCalls);
        return Results.Ok(response);
    }

    private static async Task HandleStreamingResponse(
        HttpResponse httpResponse,
        IChatSession session,
        string lastUserMessage,
        bool continueFromHistory,
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
        string finishReason = "stop";

        try
        {
            var entries = continueFromHistory
                ? session.ContinueResponseAsync(null, ct)
                : session.GetResponseStreamAsync(lastUserMessage, null, ct);
            await foreach (var entry in entries)
            {
                if (entry.Status == ChatStatus.ToolCall && entry.ToolCall is not null)
                {
                    var call = OpenAiMapper.ToToolCall(entry.ToolCall, $"call_{Guid.NewGuid():N}");
                    var callChunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId,
                        null, null, null, [call]);
                    await WriteSSE(writer, callChunk, options);
                    finishReason = "tool_calls";
                    continue;
                }

                // Skip prefill progress and other status-only entries - only
                // stream actual token content to the client.
                if (!OpenAiMapper.IsContent(entry)) continue;

                Interlocked.Increment(ref tokenCount);
                var chunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId, entry.Token, null, null);
                await WriteSSE(writer, chunk, options);
            }
        }
        catch (OperationCanceledException) { }

        var finalChunk = OpenAiMapper.ToStreamChunk(completionId, created, loaded.ModelId, null, finishReason, null);
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
