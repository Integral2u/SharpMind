global using MeChatMessage = Microsoft.Extensions.AI.ChatMessage;
global using MeChatRole = Microsoft.Extensions.AI.ChatRole;

using Microsoft.Extensions.AI;
using SharpMind.Core.Quantization;
using SharpMind.Extensions.AI;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Samples.ExtensionsAI;

/// <summary>
/// Demonstrates <see cref="SharpMindChatClient"/> — the
/// <see cref="IChatClient"/> adapter that bridges SharpMind into the
/// Microsoft.Extensions.AI ecosystem.
///
/// Shows three usage patterns:
/// <list type="number">
///   <item>Single-shot response via <c>GetResponseAsync</c>.</item>
///   <item>Streaming response via <c>GetStreamingResponseAsync</c>.</item>
///   <item>MEAI-defined tools routed through SharpMind's agent loop via
///         <see cref="AiFunctionToolAdapter"/>.</item>
/// </list>
///
/// Requires a GGUF model file on disk — set <see cref="ModelPath"/> to a
/// valid path before running.
/// </summary>
public static class SmmMeaiChatExample
{
    public const string Name = "meai-chat";

    /// <summary>Path to a GGUF model file. Update this to a model you have on disk.</summary>
    private const string ModelPath = @"c:\temp\models\SmolLM2-135M-Instruct-Q4_K_M.gguf";

    public static async Task RunAsync()
    {
        if (!File.Exists(ModelPath))
        {
            await Console.Out.WriteLineAsync($"MEAI chat example — model not found: {ModelPath}");
            await Console.Out.WriteLineAsync("Update ModelPath to point at any GGUF instruct model.");
            return;
        }

        await Console.Out.WriteLineAsync("== SharpMind + Microsoft.Extensions.AI chat example ==");
        await Console.Out.WriteLineAsync();

        // ------------------------------------------------------------------
        // 1. Load the model — same as the basic Quick Start.
        // ------------------------------------------------------------------
        await Console.Out.WriteLineAsync($"Loading model: {Path.GetFileName(ModelPath)}");

        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(ModelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);

        if (tokenizer is null)
        {
            await Console.Out.WriteLineAsync("No tokenizer data in this GGUF file.");
            return;
        }

        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);

        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, ModelPath, LoadMode.Full);
        weights.InitializeWeights();

        using var model = ModelFactory.CreateTransformer(weights, sharpConfig);
        await Console.Out.WriteLineAsync("Model loaded.");
        await Console.Out.WriteLineAsync();

        // ------------------------------------------------------------------
        // 2. Create a SharpMind chat session + agent builder.
        // ------------------------------------------------------------------
        var agentBuilder = new AgentBuilder("MeaiExampleAgent");

        await using IChatSession session = ChatSessionFactory.CreateChatSession(
            typeof(StandardGeneratorBuilder<KVCacherBuilder>),
            typeof(KVCacherBuilder),
            model,
            tokenizer,
            meta,
            agentBuilder: agentBuilder);

        // ------------------------------------------------------------------
        // 3. Wrap it in the IChatClient adapter.
        // ------------------------------------------------------------------
        await using var client = new SharpMindChatClient(session, agentBuilder);

        // ------------------------------------------------------------------
        // 4. Single-shot response.
        // ------------------------------------------------------------------
        await Console.Out.WriteLineAsync("--- Single-shot response ---");

        var messages = new List<MeChatMessage>
        {
            new(MeChatRole.System, "You are a helpful assistant. Be brief."),
            new(MeChatRole.User, "What is the capital of France?"),
        };

        ChatResponse response = await client.GetResponseAsync(messages);
        await Console.Out.WriteLineAsync($"Assistant: {response.Text}");
        await Console.Out.WriteLineAsync();

        // ------------------------------------------------------------------
        // 5. Streaming response.
        // ------------------------------------------------------------------
        await Console.Out.WriteLineAsync("--- Streaming response ---");

        var streamMessages = new List<MeChatMessage>
        {
            new(MeChatRole.User, "Name three programming languages used for LLM inference."),
        };

        await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync(streamMessages))
        {
            if (update.Text is { Length: > 0 } tok)
                Console.Write(tok);
        }
        Console.WriteLine();
        await Console.Out.WriteLineAsync();

        // ------------------------------------------------------------------
        // 6. Tool calling — define a tool via MEAI and have SharpMind use it.
        // ------------------------------------------------------------------
        await Console.Out.WriteLineAsync("--- Tool calling via MEAI ---");

        var getWeather = AIFunctionFactory.Create(
            (string city) => $"The weather in {city} is 72°F and sunny.",
            name: "GetWeather",
            description: "Gets the current weather for a city.");

        var toolMessages = new List<MeChatMessage>
        {
            new(MeChatRole.User, "What is the weather in Tokyo? Use the GetWeather tool."),
        };

        var toolOptions = new ChatOptions { Tools = [getWeather] };

        ChatResponse toolResponse = await client.GetResponseAsync(toolMessages, toolOptions);
        await Console.Out.WriteLineAsync($"Assistant: {toolResponse.Text}");
        await Console.Out.WriteLineAsync();

        await Console.Out.WriteLineAsync("Done.");
    }
}
