using System.Diagnostics;
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
/// Every /v1/chat/completions request creates its own <see cref="SharpMind.Inference.Chat.IChatSession"/>,
/// and the session owns a native-memory <see cref="SharpMind.Core.Memory.Workspace"/> (at least
/// 100 MiB, no finalizer) plus the KV caches. The endpoint released the shared model in its
/// <c>finally</c> but never disposed the session, so a server serving a 0.5B model grew from
/// 5 GB to 18 GB over twenty one-token completions. Managed GC cannot reclaim any of it.
///
/// This drives the real host end to end with a tiny exported model and asserts on the
/// process's private bytes: the leak is ≥100 MiB per request, so a handful of requests either
/// costs well over the threshold or almost nothing.
/// </summary>
[Collection("Non-Parallel")]
public sealed class CompletionSessionDisposalTests
{
    private const int Requests = 6;
    private const long MaxGrowthBytes = 300L * 1024 * 1024;   // leak would be ≥ 600 MiB

    [Fact]
    public async Task ChatCompletions_DisposesSessionPerRequest()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            ExportTinyModel(Path.Combine(dir, "tiny.smm"));

            await using var service = new SharpMindService(new SharpMindServerOptions
            {
                ModelsDir = dir, Host = "127.0.0.1", Port = 0,
            });
            var app = (WebApplication)service.BuildHost();
            await app.StartAsync();
            try
            {
                using var http = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
                var body = new
                {
                    model = "tiny.smm",
                    messages = new[] { new { role = "user", content = "king" } },
                    max_tokens = 1,
                    stream = false,
                };

                // First request pays for JIT, kernel warm-up and the HTTP pipeline; measure after it.
                await PostOk(http, body);
                long before = PrivateBytes();

                for (int i = 0; i < Requests; i++)
                    await PostOk(http, body);

                long growth = PrivateBytes() - before;
                Assert.True(growth < MaxGrowthBytes,
                    $"{Requests} completions grew private memory by {growth / (1024 * 1024)} MiB — the per-request session is not being disposed.");
            }
            finally { await app.StopAsync(); }
        }
        finally { Directory.Delete(dir, true); }
    }

    private static async Task PostOk(HttpClient http, object body)
    {
        using var resp = await http.PostAsJsonAsync("/v1/chat/completions", body);
        Assert.True(resp.IsSuccessStatusCode, $"{(int)resp.StatusCode}: {await resp.Content.ReadAsStringAsync()}");
    }

    private static long PrivateBytes()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var p = Process.GetCurrentProcess();
        p.Refresh();
        return p.PrivateMemorySize64;
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
