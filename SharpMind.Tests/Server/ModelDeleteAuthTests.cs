using System.Net.Http.Headers;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Builder;
using SharpMind.Core;
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
/// P1: POST /v1/shutdown and DELETE /v1/models/{model} were unauthenticated —
/// anyone who could reach the host could take the server down or evict a model.
/// A configured <see cref="SharpMindServerOptions.ApiKey"/> gates both; without
/// one they stay open. Drives the real Kestrel host end to end via the tiny-model
/// pattern from <see cref="CompletionSessionDisposalTests"/>.
/// </summary>
[Collection("Non-Parallel")]
public sealed class ModelDeleteAuthTests
{
    private sealed class Host : IAsyncDisposable
    {
        public required SharpMindService Service { get; init; }
        public required WebApplication App { get; init; }
        public required HttpClient Http { get; init; }

        public static async Task<Host> Start(SharpMindServerOptions options, string modelPath)
        {
            var service = new SharpMindService(options);
            var app = (WebApplication)service.BuildHost();
            await app.StartAsync();
            var http = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
            return new Host { Service = service, App = app, Http = http };
        }

        public async ValueTask DisposeAsync()
        {
            Http.Dispose();
            await App.StopAsync();
            await Service.DisposeAsync();
        }
    }

    private static async Task<(Host Host, string Dir)> StartHost(string? apiKey)
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var options = new SharpMindServerOptions
        {
            ModelsDir = dir,
            Host = "127.0.0.1",
            Port = 0,
            ApiKey = apiKey ?? "",
        };
        try
        {
            return (await Host.Start(options, dir), dir);
        }
        catch
        {
            Directory.Delete(dir, true);
            throw;
        }
    }

    private static async Task<(Host Host, string Dir)> StartHostTinyModel(string? apiKey)
    {
        var (host, dir) = await StartHost(apiKey);
        try
        {
            ExportTinyModel(Path.Combine(dir, "tiny.smm"));
            return (host, dir);
        }
        catch
        {
            Directory.Delete(dir, true);
            throw;
        }
    }

    [Fact]
    public async Task DeleteModel_WithoutApiKey_UnloadsLoadedModel()
    {
        var (host, dir) = await StartHostTinyModel(null);
        await using var _ = host;
        try
        {
            using var load = await host.Http.PostAsync("/v1/models/tiny.smm/load", content: null);
            Assert.True(load.IsSuccessStatusCode, $"load: {(int)load.StatusCode}: {await load.Content.ReadAsStringAsync()}");
            Assert.True(await IsLoaded(host.Http, "tiny.smm"));

            using var del = await host.Http.DeleteAsync("/v1/models/tiny.smm");
            Assert.True(del.IsSuccessStatusCode, $"delete: {(int)del.StatusCode}: {await del.Content.ReadAsStringAsync()}");
            Assert.False(await IsLoaded(host.Http, "tiny.smm"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task DeleteModel_Unauthenticated_RejectedWhenKeyConfigured()
    {
        var (host, dir) = await StartHostTinyModel("sesame");
        await using var _ = host;
        try
        {
            using var resp = await host.Http.DeleteAsync("/v1/models/tiny.smm");
            Assert.Equal(System.Net.HttpStatusCode.Unauthorized, resp.StatusCode);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task DeleteModel_WithBearerKey_Succeeds()
    {
        var (host, dir) = await StartHostTinyModel("sesame");
        await using var _ = host;
        try
        {
            using var load = await host.Http.PostAsync("/v1/models/tiny.smm/load", content: null);
            Assert.True(load.IsSuccessStatusCode, $"load: {(int)load.StatusCode}: {await load.Content.ReadAsStringAsync()}");

            using var req = new HttpRequestMessage(HttpMethod.Delete, "/v1/models/tiny.smm");
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", "sesame");
            using var resp = await host.Http.SendAsync(req);
            Assert.True(resp.IsSuccessStatusCode, $"delete with bearer: {(int)resp.StatusCode}: {await resp.Content.ReadAsStringAsync()}");
            Assert.False(await IsLoaded(host.Http, "tiny.smm"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task DeleteModel_WithXApiKeyHeader_Succeeds()
    {
        var (host, dir) = await StartHostTinyModel("sesame");
        await using var _ = host;
        try
        {
            using var load = await host.Http.PostAsync("/v1/models/tiny.smm/load", content: null);
            Assert.True(load.IsSuccessStatusCode, $"load: {(int)load.StatusCode}: {await load.Content.ReadAsStringAsync()}");

            using var req = new HttpRequestMessage(HttpMethod.Delete, "/v1/models/tiny.smm");
            req.Headers.Add("X-Api-Key", "sesame");
            using var resp = await host.Http.SendAsync(req);
            Assert.True(resp.IsSuccessStatusCode, $"delete with x-api-key: {(int)resp.StatusCode}: {await resp.Content.ReadAsStringAsync()}");
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task DeleteModel_WrongKey_Rejected()
    {
        var (host, dir) = await StartHostTinyModel("sesame");
        await using var _ = host;
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Delete, "/v1/models/tiny.smm");
            req.Headers.Add("X-Api-Key", "wrong");
            using var resp = await host.Http.SendAsync(req);
            Assert.Equal(System.Net.HttpStatusCode.Unauthorized, resp.StatusCode);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task DeleteModel_MissingModel_Returns404()
    {
        var (host, dir) = await StartHost(null);
        await using var _ = host;
        try
        {
            using var resp = await host.Http.DeleteAsync("/v1/models/ghost.smm");
            Assert.Equal(System.Net.HttpStatusCode.NotFound, resp.StatusCode);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Shutdown_RequiresKey_WhenConfigured()
    {
        var (host, dir) = await StartHost("sesame");
        await using var _ = host;
        try
        {
            using var unauth = await host.Http.PostAsync("/v1/shutdown", content: null);
            Assert.Equal(System.Net.HttpStatusCode.Unauthorized, unauth.StatusCode);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Shutdown_Unauthenticated_AllowedWhenNoKeyConfigured()
    {
        var (host, dir) = await StartHost(null);
        await using var _ = host;
        try
        {
            using var resp = await host.Http.PostAsync("/v1/shutdown", content: null);
            Assert.True(resp.IsSuccessStatusCode, $"shutdown: {(int)resp.StatusCode}: {await resp.Content.ReadAsStringAsync()}");
        }
        finally { Directory.Delete(dir, true); }
    }

    private static async Task<bool> IsLoaded(HttpClient http, string modelId)
    {
        using var resp = await http.GetAsync("/v1/models/loaded");
        Assert.True(resp.IsSuccessStatusCode, $"loaded list: {(int)resp.StatusCode}: {await resp.Content.ReadAsStringAsync()}");
        using var json = await resp.Content.ReadAsStreamAsync();
        using var doc = await System.Text.Json.JsonDocument.ParseAsync(json);
        foreach (var item in doc.RootElement.GetProperty("data").EnumerateArray())
            if (item.GetProperty("id").GetString() == modelId)
                return true;
        return false;
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