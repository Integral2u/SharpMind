using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Server;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Server;

/// <summary>
/// /v1/completions is the server's batched path: every request for a model feeds
/// one shared generator. These drive the real host with a tiny exported model and
/// pin the protocol shape (text/choices/usage, SSE framing, validation) plus the
/// plumbing fact that requests share a batcher rather than starting one each.
/// </summary>
[Collection("Non-Parallel")]
public sealed class CompletionsEndpointTests : IAsyncLifetime
{
    private string _dir = "";
    private SharpMindService? _service;
    private WebApplication? _app;
    private HttpClient? _http;

    public async Task InitializeAsync()
    {
        _dir = Path.Combine(Path.GetTempPath(), "sharpmind_completions_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        ExportTinyModel(Path.Combine(_dir, "tiny.smm"));

        _service = new SharpMindService(new SharpMindServerOptions
        {
            ModelsDir = _dir,
            Host = "127.0.0.1",
            Port = 0,
            CompletionBatchSize = 4,
        });
        _app = (WebApplication)_service.BuildHost();
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public async Task DisposeAsync()
    {
        _http?.Dispose();
        if (_app is not null) await _app.StopAsync();
        if (_service is not null) await _service.DisposeAsync();
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    private static object Body(string prompt, int maxTokens = 3, bool stream = false) => new
    {
        model = "tiny.smm",
        prompt,
        max_tokens = maxTokens,
        temperature = 0,
        stream,
    };

    [Fact]
    public async Task Completion_ReturnsTextChoicesAndUsage()
    {
        using var resp = await _http!.PostAsJsonAsync("/v1/completions", Body("king"));
        Assert.True(resp.IsSuccessStatusCode, await resp.Content.ReadAsStringAsync());

        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        var root = doc.RootElement;
        Assert.Equal("text_completion", root.GetProperty("object").GetString());
        Assert.Equal("tiny.smm", root.GetProperty("model").GetString());

        var choice = root.GetProperty("choices")[0];
        Assert.Equal(0, choice.GetProperty("index").GetInt32());
        Assert.True(choice.TryGetProperty("text", out _));
        Assert.Equal("length", choice.GetProperty("finish_reason").GetString());

        var usage = root.GetProperty("usage");
        Assert.True(usage.GetProperty("prompt_tokens").GetInt32() > 0);
        Assert.Equal(3, usage.GetProperty("completion_tokens").GetInt32());
        Assert.Equal(
            usage.GetProperty("prompt_tokens").GetInt32() + usage.GetProperty("completion_tokens").GetInt32(),
            usage.GetProperty("total_tokens").GetInt32());
    }

    [Fact]
    public async Task Completion_StreamsSseChunksAndDone()
    {
        using var resp = await _http!.PostAsJsonAsync("/v1/completions", Body("king", maxTokens: 2, stream: true));
        Assert.True(resp.IsSuccessStatusCode, await resp.Content.ReadAsStringAsync());

        string body = await resp.Content.ReadAsStringAsync();
        var data = body.Split('\n')
            .Where(line => line.StartsWith("data: ", StringComparison.Ordinal))
            .Select(line => line["data: ".Length..])
            .ToList();

        Assert.Equal("[DONE]", data[^1]);
        Assert.Contains(data, d => d.Contains("\"finish_reason\":\"length\""));

        foreach (string chunk in data.Where(d => d != "[DONE]"))
        {
            using var doc = JsonDocument.Parse(chunk);
            Assert.Equal("text_completion", doc.RootElement.GetProperty("object").GetString());
            Assert.True(doc.RootElement.GetProperty("choices").GetArrayLength() == 1);
        }
    }

    [Fact]
    public async Task Completion_NProducesOneChoicePerRequestedCompletion()
    {
        using var resp = await _http!.PostAsJsonAsync("/v1/completions", new
        {
            model = "tiny.smm",
            prompt = "king",
            n = 3,
            max_tokens = 2,
            temperature = 0,
        });
        Assert.True(resp.IsSuccessStatusCode, await resp.Content.ReadAsStringAsync());

        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        var choices = doc.RootElement.GetProperty("choices");
        Assert.Equal(3, choices.GetArrayLength());
        Assert.Equal(new[] { 0, 1, 2 }, Enumerable.Range(0, choices.GetArrayLength())
            .Select(i => choices[i].GetProperty("index").GetInt32()));
    }

    [Fact]
    public async Task Completion_AcceptsPromptArray()
    {
        using var resp = await _http!.PostAsJsonAsync("/v1/completions", new
        {
            model = "tiny.smm",
            prompt = new[] { "king", "queen" },
            max_tokens = 2,
            temperature = 0,
        });
        Assert.True(resp.IsSuccessStatusCode, await resp.Content.ReadAsStringAsync());

        using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync());
        Assert.Equal(2, doc.RootElement.GetProperty("choices").GetArrayLength());
        Assert.True(doc.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task Completion_RejectsBadRequests()
    {
        using (var missing = await _http!.PostAsJsonAsync("/v1/completions", new { prompt = "x" }))
            Assert.Equal(400, (int)missing.StatusCode);

        using (var noPrompt = await _http!.PostAsJsonAsync("/v1/completions", new { model = "tiny.smm" }))
            Assert.Equal(400, (int)noPrompt.StatusCode);

        using (var unknown = await _http!.PostAsJsonAsync("/v1/completions", new { model = "nope.gguf", prompt = "x" }))
            Assert.Equal(404, (int)unknown.StatusCode);

        using (var tokens = await _http!.PostAsJsonAsync("/v1/completions", new { model = "tiny.smm", prompt = new[] { 1, 2, 3 } }))
            Assert.Equal(400, (int)tokens.StatusCode);

        using (var echo = await _http!.PostAsJsonAsync("/v1/completions", new { model = "tiny.smm", prompt = "x", echo = true }))
            Assert.Equal(400, (int)echo.StatusCode);
    }

    /// <summary>
    /// The point of the endpoint: requests share one batcher for the model. The
    /// generator instance must be stable across calls and must have actually run
    /// steps that produced tokens.
    /// </summary>
    [Fact]
    public async Task Completion_RequestsShareOneBatcher()
    {
        string firstBody, secondBody;
        using (var first = await _http!.PostAsJsonAsync("/v1/completions", Body("king", maxTokens: 2)))
        {
            firstBody = await first.Content.ReadAsStringAsync();
            Assert.True(first.IsSuccessStatusCode, firstBody);
        }
        using (var second = await _http!.PostAsJsonAsync("/v1/completions", Body("queen", maxTokens: 2)))
        {
            secondBody = await second.Content.ReadAsStringAsync();
            Assert.True(second.IsSuccessStatusCode, secondBody);
        }

        var models = _app!.Services.GetRequiredService<ModelManager>();
        var batches = _app.Services.GetRequiredService<BatchService>();
        var loaded = models.GetLoaded("tiny.smm");
        Assert.NotNull(loaded);

        var batcher = batches.Get(loaded!);
        Assert.Same(batcher, batches.Get(loaded!));
        var stats = batcher.Stats;
        Assert.True(stats.GeneratedTokens >= 2,
            $"expected the shared batcher to have produced tokens, saw {stats.GeneratedTokens} " +
            $"(steps {stats.Steps}, batch {stats.AverageBatchSize:0.00}); first={firstBody} second={secondBody}");
        Assert.Equal(1, batches.Snapshot().Models);
    }

    private static void ExportTinyModel(string smmPath)
    {
        var cfg = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        using var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);

        var tokens = new string[cfg.VocabSize];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = Vocabulary.ByteTokenString(i);
        var tokenizer = Tokenizer.FromGguf(tokens, merges: null, tokenTypes: null, bosId: -1, eosId: -1);

        SmmTrainingExporter.Export(weights, tokenizer, smmPath, new SmmWriteOptions { Source = "training" });
    }
}
