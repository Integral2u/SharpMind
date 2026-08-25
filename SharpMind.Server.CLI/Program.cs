using System.Diagnostics;
using System.Text;
using System.Text.Json;
using SharpMind.Server;

// ── Parse args (shared by both modes) ─────────────────────────────────
var cliArgs = Environment.GetCommandLineArgs().Skip(1).ToArray();
var options = new SharpMindServerOptions();
var modelArgs = new List<string>();
bool serviceMode = false;
bool stopMode = false;
bool noCli = false;

for (int i = 0; i < cliArgs.Length; i++)
{
    switch (cliArgs[i])
    {
        case "--service":
            serviceMode = true;
            break;
        case "--stop":
            stopMode = true;
            break;
        case "--nocli":
            noCli = true;
            break;
        case "--no-files":
            options.DisableFileIO = true;
            break;
        case "--no-network":
            options.DisableNetworkIO = true;
            break;
        case "--models" when i + 1 < cliArgs.Length:
            options.ModelsDir = cliArgs[++i];
            break;
        case "--host" when i + 1 < cliArgs.Length:
            options.Host = cliArgs[++i];
            break;
        case "--port" when i + 1 < cliArgs.Length:
            if (int.TryParse(cliArgs[++i], out var port))
                options.Port = port;
            break;
        case "--model" when i + 1 < cliArgs.Length:
            var raw = cliArgs[++i];
            foreach (var m in raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                modelArgs.Add(m);
            break;
        case "--help" or "-h":
            if (!serviceMode) PrintUsage();
            return;
    }
}

if (serviceMode)
    await RunServiceAsync(options, modelArgs);
else if (stopMode)
    await RunStopAsync(options);
else
    await RunClientAsync(options, modelArgs, noCli);

// ══════════════════════════════════════════════════════════════════════
//  SERVICE MODE (--service): host the OpenAI-compatible API
// ══════════════════════════════════════════════════════════════════════

static async Task RunServiceAsync(SharpMindServerOptions options, List<string> modelArgs)
{
    using var cts = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        cts.Cancel();
    };

    await using var server = new SharpMindService(options);

    // Build the host first so ModelManager is available, then optionally
    // pre-load models before listening for requests.
    _ = server.BuildHost();

    foreach (var modelArg in modelArgs)
    {
        try
        {
            Console.WriteLine($"[SharpMind] Pre-loading model: {modelArg}");
            var progress = new Progress<string>(msg => Console.WriteLine($"[SharpMind] {msg}"));
            await server.PreloadModelAsync(modelArg, progress, cts.Token);
            Console.WriteLine($"[SharpMind] Model loaded: {modelArg}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[SharpMind] Failed to load '{modelArg}': {ex.Message}");
        }
    }

    await server.StartAsync(cts.Token);
}

// ══════════════════════════════════════════════════════════════════════
//  STOP MODE (--stop): shut down a running service and exit
// ══════════════════════════════════════════════════════════════════════

static async Task RunStopAsync(SharpMindServerOptions options)
{
    var baseUrl = $"http://{options.Host}:{options.Port}";
    using var http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromSeconds(5) };

    if (!await IsServerRunning(http))
    {
        Console.WriteLine("[SharpMind] No service running.");
        return;
    }

    Console.WriteLine("[SharpMind] Shutting down service...");
    await ShutdownServiceAsync(http);
    Console.WriteLine("[SharpMind] Service stopped.");
}

// ══════════════════════════════════════════════════════════════════════
//  CLIENT MODE (default): interactive REPL, spawns service if needed
// ══════════════════════════════════════════════════════════════════════

static async Task RunClientAsync(SharpMindServerOptions options, List<string> modelArgs, bool noCli)
{
    Console.WriteLine("SharpMind Open AI Protocol Server CLI");

    var baseUrl = $"http://{options.Host}:{options.Port}";
    using var http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };

    // ── Start or connect ────────────────────────────────────────────
    bool weStartedIt = false;

    if (await IsServerRunning(http))
    {
        Console.WriteLine($"[SharpMind] Connected to existing service at {baseUrl}");
    }
    else
    {
        Console.WriteLine("[SharpMind] Starting service...");
        await SpawnServiceProcessAsync(options, modelArgs);
        weStartedIt = true;
    }

    // ── Resolve model ───────────────────────────────────────────────
    // The server is the source of truth. Adopt whatever it has loaded,
    // or fall through to a one-model auto-select hint.
    string? selectedModel = modelArgs.Count > 0 ? modelArgs[0] : null;
    var availableModels = await FetchModels(http);
    var loadedModels = await FetchLoaded(http);
    bool modelLoaded = loadedModels.Count > 0;

    if (selectedModel is null && loadedModels.Count > 0)
    {
        selectedModel = loadedModels[0];
        Console.WriteLine($"[SharpMind] Using active model: {selectedModel}");
    }

    if (selectedModel is null && availableModels.Count == 1)
    {
        selectedModel = availableModels[0];
        Console.WriteLine($"[SharpMind] Available model: {selectedModel} (not loaded)");
    }
    else if (selectedModel is null && availableModels.Count > 1)
    {
        Console.WriteLine("[SharpMind] Multiple models available. Use /model <name> to select.");
    }
    else if (selectedModel is null && availableModels.Count == 0)
    {
        Console.WriteLine("[SharpMind] No models found. Use /models to see what's available.");
    }
    {
        Console.WriteLine("[SharpMind] No models found. Use /models to see what's available.");
    }

    Console.WriteLine("[SharpMind] Commands: /help, /model, /models, /loaded, /unload, /unloadall, /clear, /restart, /stop, /exit");
    if (options.DisableFileIO || options.DisableNetworkIO)
    {
        var gated = new List<string>();
        if (options.DisableFileIO) gated.Add("file IO");
        if (options.DisableNetworkIO) gated.Add("network IO");
        Console.WriteLine($"[SharpMind] Gated: {string.Join(", ", gated)}");
    }
    Console.WriteLine();

    // ── Message loop ────────────────────────────────────────────────
    var conversation = new List<Dictionary<string, object>>();

    try
    {
        while (true)
        {
            var prompt = selectedModel is not null && modelLoaded
                ? $"you ({selectedModel})> " : "you> ";
            Console.Write(prompt);

            var input = Console.ReadLine();
            if (input is null) break;

            var trimmed = input.Trim();
            if (trimmed.Length == 0) continue;

            if (trimmed.Equals("/exit", StringComparison.OrdinalIgnoreCase) ||
                trimmed.Equals("/quit", StringComparison.OrdinalIgnoreCase))
                break;

            if (trimmed.Equals("/stop", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("[SharpMind] Shutting down service...");
                await ShutdownServiceAsync(http);
                break;
            }

            if (trimmed.Equals("/restart", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("[SharpMind] Restarting service...");
                await ShutdownServiceAsync(http);
                await Task.Delay(500);
                await SpawnServiceProcessAsync(options, modelArgs);
                weStartedIt = true;
                Console.WriteLine("[SharpMind] Service restarted.");
                continue;
            }

            if (trimmed.Equals("/help", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("  /help              Show this help");
                Console.WriteLine("  /model             Show current model");
                Console.WriteLine("  /model <name>      Switch to a different model");
                Console.WriteLine("  /models            List available models on disk");
                Console.WriteLine("  /loaded            List models loaded in memory");
                Console.WriteLine("  /unload            Unload current model");
                Console.WriteLine("  /unload <name>     Unload a model by name");
                Console.WriteLine("  /unloadall         Unload all models");
                Console.WriteLine("  /clear             Clear conversation history");
                Console.WriteLine("  /restart           Restart the service");
                Console.WriteLine("  /stop              Shut down the service and exit");
                Console.WriteLine("  /exit              Quit (asks whether to stop the service)");
                continue;
            }

            if (trimmed.Equals("/model", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine(selectedModel is not null
                    ? $"[SharpMind] Current model: {selectedModel}"
                    : "[SharpMind] No model selected. Use /model <name>.");
                continue;
            }

            if (trimmed.StartsWith("/model ", StringComparison.OrdinalIgnoreCase))
            {
                var arg = trimmed[7..].Trim();
                availableModels = await FetchModels(http);
                var match = availableModels.FirstOrDefault(m =>
                    m.Equals(arg, StringComparison.OrdinalIgnoreCase));
                if (match is not null)
                {
                    selectedModel = match;
                    conversation.Clear();
                    Console.Write($"[SharpMind] Switched to model: {selectedModel} (conversation cleared)");
                    await LoadModelOnServer(http, selectedModel);
                    modelLoaded = true;
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine($"[SharpMind] Model '{arg}' not found. Available:");
                    foreach (var m in availableModels)
                        Console.WriteLine($"  - {m}");
                }
                continue;
            }

            if (trimmed.Equals("/models", StringComparison.OrdinalIgnoreCase))
            {
                availableModels = await FetchModels(http);
                if (availableModels.Count == 0)
                    Console.WriteLine("[SharpMind] No models available.");
                else
                {
                    Console.WriteLine("[SharpMind] Available models:");
                    foreach (var m in availableModels)
                    {
                        var marker = m == selectedModel ? " (active)" : "";
                        Console.WriteLine($"  - {m}{marker}");
                    }
                }
                continue;
            }

            if (trimmed.Equals("/loaded", StringComparison.OrdinalIgnoreCase))
            {
                await ListLoadedModels(http);
                continue;
            }

            if (trimmed.Equals("/unload", StringComparison.OrdinalIgnoreCase))
            {
                if (selectedModel is null)
                    Console.WriteLine("[SharpMind] No model selected. Use /unload <name>.");
                else
                {
                    await UnloadModel(http, selectedModel);
                    selectedModel = null;
                    modelLoaded = false;
                }
                continue;
            }

            if (trimmed.StartsWith("/unload ", StringComparison.OrdinalIgnoreCase))
            {
                var name = trimmed[8..].Trim();
                await UnloadModel(http, name);
                if (selectedModel is not null &&
                    selectedModel.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    selectedModel = null;
                    modelLoaded = false;
                }
                continue;
            }

            if (trimmed.Equals("/unloadall", StringComparison.OrdinalIgnoreCase))
            {
                foreach (var m in await FetchLoaded(http))
                    await UnloadModel(http, m);
                Console.WriteLine("[SharpMind] All models unloaded.");
                selectedModel = null;
                modelLoaded = false;
                continue;
            }

            if (trimmed.Equals("/clear", StringComparison.OrdinalIgnoreCase))
            {
                conversation.Clear();
                Console.WriteLine("[SharpMind] Conversation cleared.");
                continue;
            }

            if (selectedModel is null)
            {
                Console.WriteLine("[SharpMind] No model selected. Use /model <name> first.");
                continue;
            }

            if (!await IsServerRunning(http))
            {
                Console.WriteLine("[SharpMind] Service is not running. Use /restart to start it.");
                continue;
            }

            conversation.Add(new Dictionary<string, object> { ["role"] = "user", ["content"] = trimmed });

            var request = new Dictionary<string, object>
            {
                ["model"] = selectedModel,
                ["messages"] = conversation,
                ["stream"] = true
            };

            Console.Write("assistant> ");
            try
            {
                var sb = await StreamChatCompletion(http, request);
                modelLoaded = true;
                conversation.Add(new Dictionary<string, object> { ["role"] = "assistant", ["content"] = sb.ToString() });
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n[SharpMind] Error: {ex.Message}");
            }
        }
    }
    finally
    {
        if (await IsServerRunning(http))
        {
            bool shutDown;
            if (noCli)
            {
                shutDown = weStartedIt;
            }
            else if (!Console.IsInputRedirected)
            {
                var label = weStartedIt ? "service we started" : "service you connected to";
                Console.Write($"\n[SharpMind] {label} is running at {baseUrl}. Shut down? (Y/n): ");
                var answer = Console.ReadLine();
                shutDown = string.IsNullOrWhiteSpace(answer) ||
                           answer.TrimStart().StartsWith('y');
            }
            else
            {
                shutDown = weStartedIt;
            }

            if (shutDown)
            {
                Console.WriteLine("[SharpMind] Shutting down service...");
                await ShutdownServiceAsync(http);
            }
            else
            {
                Console.WriteLine($"[SharpMind] Service left running at {baseUrl}.");
            }
        }
    }
}

// ── Service management ────────────────────────────────────────────────

static async Task SpawnServiceProcessAsync(SharpMindServerOptions options, List<string> modelArgs)
{
    var exe = Environment.ProcessPath ?? Environment.GetCommandLineArgs()[0];

    var args = $"--service --models-dir \"{options.ModelsDir}\" --host {options.Host} --port {options.Port}";
    foreach (var m in modelArgs)
        args += $" --model \"{m}\"";
    if (options.DisableFileIO) args += " --no-files";
    if (options.DisableNetworkIO) args += " --no-network";

    var psi = new ProcessStartInfo
    {
        FileName = exe,
        Arguments = args,
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true,
        Environment = {
            ["ASPNETCORE_ENVIRONMENT"] = "Production",
            ["ASPNETCORE_HOSTINGSTARTUPASSEMBLIES"] = ""
        }
    };

    var proc = Process.Start(psi)!;

    // Pipe service stdout/stderr to CLI for the lifetime of the service
    // process. These tasks end naturally when the process exits (streams
    // close, ReadLineAsync returns null).
    _ = PipeStreamAsync(proc.StandardOutput, Console.Out);
    _ = PipeStreamAsync(proc.StandardError, Console.Error);

    var probe = new HttpClient
    {
        BaseAddress = new Uri($"http://{options.Host}:{options.Port}"),
        Timeout = TimeSpan.FromSeconds(2)
    };

    for (int retries = 0; retries < 120; retries++)
    {
        if (await IsServerRunning(probe))
            return;
        await Task.Delay(250);
    }

    Console.WriteLine("[SharpMind] WARNING: Service did not respond in time.");
}

static async Task PipeStreamAsync(StreamReader reader, TextWriter destination)
{
    try
    {
        char[] buf = new char[4096];
        var sb = new StringBuilder();
        while (true)
        {
            int n = await reader.ReadAsync(buf, 0, buf.Length);
            if (n == 0) break;

            for (int i = 0; i < n; i++)
            {
                char ch = buf[i];
                if (ch == '\n')
                {
                    // \n is always a line break — flush whatever we have
                    if (sb.Length > 0)
                    {
                        destination.Write(sb);
                        sb.Clear();
                    }
                    destination.Write('\n');
                    destination.Flush();
                }
                else if (ch == '\r')
                {
                    // \r may be standalone (in-place overwrite) or part of
                    // \r\n.  Flush the accumulated text first so it appears
                    // before the cursor resets.
                    if (sb.Length > 0)
                    {
                        destination.Write(sb);
                        sb.Clear();
                    }
                    // Write the \r through so the terminal overwrites the
                    // current line.  If a \n follows on the next read it
                    // will advance to the next line.
                    destination.Write('\r');
                    destination.Flush();
                }
                else
                {
                    sb.Append(ch);
                }
            }

            // Flush any partial line accumulated after the last \r so the
            // terminal always shows the latest progress tick.  The next \r
            // (or stream close) will overwrite / discard it.
            if (sb.Length > 0)
            {
                destination.Write(sb);
                destination.Flush();
                sb.Clear();
            }
        }
        if (sb.Length > 0)
        {
            destination.Write(sb);
            destination.Flush();
        }
    }
    catch { /* stream closed or process exited */ }
}

static async Task ShutdownServiceAsync(HttpClient http)
{
    try { await http.PostAsync("/v1/shutdown", content: null); } catch { }
    for (int i = 0; i < 40; i++)
    {
        await Task.Delay(250);
        if (!await IsServerRunning(http)) return;
    }
}

// ── HTTP helpers ──────────────────────────────────────────────────────

static async Task<bool> IsServerRunning(HttpClient http)
{
    try
    {
        var resp = await http.GetAsync("/v1/health");
        return resp.IsSuccessStatusCode;
    }
    catch { return false; }
}

static async Task<List<string>> FetchModels(HttpClient http)
{
    try
    {
        var json = await http.GetFromJsonAsync<JsonElement>("/v1/models");
        return [.. json.GetProperty("data").EnumerateArray()
            .Select(m => m.GetProperty("id").GetString()!)
            .OrderBy(m => m, StringComparer.OrdinalIgnoreCase)];
    }
    catch { return []; }
}

static async Task<List<string>> FetchLoaded(HttpClient http)
{
    try
    {
        var json = await http.GetFromJsonAsync<JsonElement>("/v1/models/loaded");
        return [.. json.GetProperty("data").EnumerateArray()
            .Select(m => m.GetProperty("id").GetString()!)
            .OrderBy(m => m, StringComparer.OrdinalIgnoreCase)];
    }
    catch { return []; }
}

static async Task ListLoadedModels(HttpClient http)
{
    var loaded = await FetchLoaded(http);
    if (loaded.Count == 0)
    {
        Console.WriteLine("[SharpMind] No models loaded in memory.");
        return;
    }
    Console.WriteLine("[SharpMind] Loaded models:");
    foreach (var id in loaded)
        Console.WriteLine($"  - {id}");
}

static async Task UnloadModel(HttpClient http, string name)
{
    try
    {
        var resp = await http.DeleteAsync($"/v1/models/{Uri.EscapeDataString(name)}");
        Console.WriteLine(resp.IsSuccessStatusCode
            ? $"[SharpMind] Unloaded: {name}"
            : $"[SharpMind] Failed to unload '{name}': {resp.StatusCode}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[SharpMind] Error unloading '{name}': {ex.Message}");
    }
}

static async Task LoadModelOnServer(HttpClient http, string model)
{
    // Check if already loaded
    var loaded = await FetchLoaded(http);
    if (loaded.Any(m => m.Equals(model, StringComparison.OrdinalIgnoreCase)))
        return;

    Console.Write($"[SharpMind] Loading {model}...");
    try
    {
        var resp = await http.PostAsync($"/v1/models/{Uri.EscapeDataString(model)}/load", content: null);
        if (resp.IsSuccessStatusCode)
            Console.WriteLine(" done.");
        else
            Console.WriteLine($" failed ({resp.StatusCode}).");
    }
    catch (Exception ex)
    {
        Console.WriteLine($" error: {ex.Message}");
    }
}

static async Task<StringBuilder> StreamChatCompletion(HttpClient http, Dictionary<string, object> request)
{
    var json = JsonSerializer.Serialize(request, new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    });

    using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
    var content = new StringContent(json, Encoding.UTF8, "application/json");
    var response = await http.PostAsync("/v1/chat/completions", content, cts.Token);
    response.EnsureSuccessStatusCode();

    var stream = await response.Content.ReadAsStreamAsync(cts.Token);
    var reader = new StreamReader(stream);
    var sb = new StringBuilder();

    using var lineCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

    while (true)
    {
        string? line;
        try
        {
            line = await reader.ReadLineAsync(lineCts.Token);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n[SharpMind] Service not responding (30 s timeout).");
            break;
        }

        if (line is null) break;

        if (!line.StartsWith("data: "))
            continue;

        var data = line[6..];
        if (data == "[DONE]")
            break;

        try
        {
            var chunk = JsonSerializer.Deserialize<JsonElement>(data);
            var choices = chunk.GetProperty("choices");
            if (choices.GetArrayLength() == 0) continue;

            var delta = choices[0].GetProperty("delta");
            if (delta.TryGetProperty("content", out var contentProp))
            {
                var text = contentProp.GetString() ?? "";
                sb.Append(text);
                Console.Write(text);
            }
        }
        catch { }
    }

    return sb;
}

static void PrintUsage()
{
    Console.WriteLine("""
    Usage: sharpmind-server [options]

    Options:
      --models-dir, --models <path>
                            Directory containing .gguf model files
                            Default: ~/SharpMind/Models
      --host <host>         Hostname or IP to bind to (default: localhost)
      --port <port>         HTTP port to listen on (default: 11435)
      --model <names>       Model(s) to load (comma-separated for multiple)
      --stop                Shut down a running service and exit
      --nocli               Process args and exit without interactive REPL
                            (default: shut down service we started on exit)
      --no-files            Disable file IO for tool calls (read/write only)
      --no-network          Disable network IO for tool calls
      -h, --help            Show this help message
    """);
}
