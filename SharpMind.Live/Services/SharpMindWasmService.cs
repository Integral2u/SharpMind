using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.JSInterop;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Live.Services;

/// <summary>
/// Blazor WASM interop entry point. All [JSInvokable] methods are static
/// (required by DotNet.invokeMethodAsync) and delegate to a singleton
/// <see cref="EngineState"/> that holds the loaded model and session.
/// Rooted from Program.cs so ILLink keeps it (only referenced by name from JS).
/// </summary>
public static class SharpMindEngine
{
    private static EngineState? _state;
    private static IJSRuntime? _js;

    /// <summary>Wired up from Program.cs after the host is built.</summary>
    public static void SetRuntime(IJSRuntime js) => _js = js;

    private static void Log(StringBuilder log, string line)
    {
        log.AppendLine(line);
        if (_js is IJSInProcessRuntime inProc)
        {
            try { inProc.InvokeVoid("SharpMindEngine.logLine", line); }
            catch { /* the page may already be tearing down */ }
        }
    }

    [JSInvokable]
    public static async Task<string> LoadModel(string modelUrl)
    {
        var log = new StringBuilder();

        try
        {
            var fileName = Path.GetFileName(new Uri(modelUrl).AbsolutePath);
            var modelPath = $"/models/{fileName}";

            var dir = Path.GetDirectoryName(modelPath)!;
            Directory.CreateDirectory(dir);

            Log(log, $"fetching {fileName}…");

            using var http = new HttpClient();
            var bytes = await http.GetByteArrayAsync(modelUrl);
            await File.WriteAllBytesAsync(modelPath, bytes);

            Log(log, "writing model bytes to Blazor virtual filesystem…");

            GgufLoader.Load(modelPath, null, out var meta, out var config, out var tokenizer);

            Log(log, "ModelFactory.CreateWeights(..., useSafeIo: true)…  This will take some time.");

            var sharpConfig = SharpMindConfig.ForModel(
                config.NumHeads, config.NumKvHeads, config.Architecture);

            var mapping = new MappingBuilder(HardwareTier.Scalar)
                .ApplyQuantPreset(sharpConfig)
                .Build();
            var qOps = QuantizationFactory.Create(mapping);

            var weights = ModelFactory.CreateWeights(
                config, sharpConfig, qOps, modelPath,
                loadMode: LoadMode.Full,
                quantizedResident: true,
                useSafeIo: true);

            weights.InitializeWeights();

            Log(log, "hardware tier: <span class=\"warn\">Scalar</span> (WebAssembly has no AVX2/FMA)");
            Log(log, "threads: <span class=\"warn\">1</span> (GitHub Pages can't set the headers WASM threading needs)");

            var transformer = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

            var session = ChatSessionFactory.CreateChatSession<
                StandardGeneratorBuilder<KVCacherBuilder>,
                KVCacherBuilder>(transformer, tokenizer!, meta);

            session.InitializeChat();

            if (_state is not null)
                await _state.DisposeAsync();
            _state = new EngineState(transformer, tokenizer!, session);

            Log(log, "model ready");
        }
        catch (Exception ex)
        {
            Log(log, $"<span class=\"err\">{ex.GetType().Name}: {ex.Message}</span>");
        }

        return log.ToString();
    }

    [JSInvokable]
    public static async Task<string> Generate(string prompt)
    {
        if (_state?.Session is null)
            return "Error: no model loaded. Call LoadModel first.";

        var sb = new StringBuilder();

        try
        {
            await foreach (var entry in _state.Session.GetResponseStreamAsync(prompt))
            {
                if (entry.Status == ChatStatus.Complete ||
                    entry.Status == ChatStatus.Interrupted)
                    break;

                if (entry.Token is not null)
                    sb.Append(entry.Token);
            }
        }
        catch (Exception ex)
        {
            return $"Error: {ex.GetType().Name}: {ex.Message}";
        }

        return sb.ToString();
    }

    private sealed class EngineState : IAsyncDisposable
    {
        public Transformer Transformer { get; }
        public Tokenizer Tokenizer { get; }
        public IChatSession Session { get; }

        public EngineState(Transformer transformer, Tokenizer tokenizer, IChatSession session)
        {
            Transformer = transformer;
            Tokenizer = tokenizer;
            Session = session;
        }

        public async ValueTask DisposeAsync()
        {
            await Session.DisposeAsync();
        }
    }
}
