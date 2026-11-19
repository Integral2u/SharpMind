using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
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
/// Batched chat: /v1/chat/completions goes through the model's shared batcher by
/// default, so concurrent chats share forward passes. The session still owns the
/// prompt, tools and streaming; only generation moves into the batch. These pin
/// the OpenAI response shape in both modes and that the batcher is really used.
/// </summary>
[Collection("Non-Parallel")]
public sealed class ChatBatchingTests
{
    private static async Task WithServerAsync(bool batchChat, Func<HttpClient, WebApplication, Task> test)
    {
        string dir = Path.Combine(Path.GetTempPath(), "sharpmind_chatbatch_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            ExportTinyModel(Path.Combine(dir, "tiny.smm"));

            await using var service = new SharpMindService(new SharpMindServerOptions
            {
                ModelsDir = dir,
                Host = "127.0.0.1",
                Port = 0,
                CompletionBatchSize = 4,
                BatchChatCompletions = batchChat,
            });
            var app = (WebApplication)service.BuildHost();
            await app.StartAsync();
            try
            {
                using var http = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
                await test(http, app);
            }
            finally { await app.StopAsync(); }
        }
        finally
        {
            try { Directory.Delete(dir, true); } catch (IOException) { }
        }
    }

    private static object ChatBody(string text, int maxTokens = 2) => new
    {
        model = "tiny.smm",
        messages = new[] { new { role = "user", content = text } },
        max_tokens = maxTokens,
        temperature = 0,
        stream = false,
    };

    [Fact]
    public async Task ChatCompletion_RunsThroughTheSharedBatcher()
    {
        await WithServerAsync(batchChat: true, async (http, app) =>
        {
            using var resp = await http.PostAsJsonAsync("/v1/chat/completions", ChatBody("king"));
            string body = await resp.Content.ReadAsStringAsync();
            Assert.True(resp.IsSuccessStatusCode, body);

            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            Assert.Equal("chat.completion", root.GetProperty("object").GetString());
            var choice = root.GetProperty("choices")[0];
            Assert.Equal("assistant", choice.GetProperty("message").GetProperty("role").GetString());
            // The chat path reports stop/tool_calls only (it does not distinguish
            // a token limit), which is pre-existing behaviour.
            Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
            Assert.Equal(2, root.GetProperty("usage").GetProperty("completion_tokens").GetInt32());

            var models = app.Services.GetRequiredService<ModelManager>();
            var batches = app.Services.GetRequiredService<BatchService>();
            var loaded = models.GetLoaded("tiny.smm");
            Assert.NotNull(loaded);
            Assert.True(batches.Get(loaded!).Stats.GeneratedTokens >= 1,
                "the chat turn did not run through the batcher");
        });
    }

    /// <summary>
    /// The pre-batching path must keep working: no batcher is created and the
    /// session owns its generator/caches as before.
    /// </summary>
    [Fact]
    public async Task ChatCompletion_WorksWithBatchingDisabled()
    {
        await WithServerAsync(batchChat: false, async (http, app) =>
        {
            using var resp = await http.PostAsJsonAsync("/v1/chat/completions", ChatBody("queen"));
            string body = await resp.Content.ReadAsStringAsync();
            Assert.True(resp.IsSuccessStatusCode, body);

            using var doc = JsonDocument.Parse(body);
            Assert.Equal(2, doc.RootElement.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
            Assert.Equal(0, app.Services.GetRequiredService<BatchService>().Snapshot().Models);
        });
    }

    [Fact]
    public async Task ChatCompletion_StreamsWithBatching()
    {
        await WithServerAsync(batchChat: true, async (http, app) =>
        {
            using var resp = await http.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "tiny.smm",
                messages = new[] { new { role = "user", content = "king" } },
                max_tokens = 2,
                temperature = 0,
                stream = true,
            });
            string body = await resp.Content.ReadAsStringAsync();
            Assert.True(resp.IsSuccessStatusCode, body);
            Assert.Contains("data: [DONE]", body);
            Assert.Contains("\"chat.completion.chunk\"", body);
        });
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
