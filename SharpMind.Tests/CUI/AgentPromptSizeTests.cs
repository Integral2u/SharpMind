using System.Text.Json;
using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.CUI.App;
using SharpMind.Inference.Agent;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Measures the agent system prompt size for the CUI's real tool set
/// (CuiTools + WeatherTool + FileSystemTool) and asserts the compact tool
/// listing keeps it small. Every turn in the CUI re-prefills the full
/// conversation (the Auto formatter always does a full rebuild), and at
/// ~200ms/prefill-token the size of the agent prompt is seconds of wait — this
/// test exists to stop a future change from silently tripling it.
/// </summary>
public sealed class AgentPromptSizeTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";

    private readonly ITestOutputHelper _output;

    public AgentPromptSizeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CompactToolListing_KeepsAgentPromptSmall()
    {
        if (!File.Exists(ModelPath))
            return; // dev-machine diagnostic; no GGUF shipped in-repo

        var load = await SessionLauncher.LoadModelAsync(new SessionOptions
        {
            ModelPath = ModelPath,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
        });
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var builder = new AgentBuilder("Delta")
            {
                DisabledTools = []
            };
            builder.WithTools(new CuiTools(new CuiToolContext()));
            builder.WithTools(new WeatherTool());
            builder.WithTools(new FileSystemTool(Path.GetTempPath()));

            string prompt = builder.BuildAgentPrompt();
            string oldToolsJson = builder.ToolDefinitions.ToJsonString(new JsonSerializerOptions { WriteIndented = true });

            var tokenizer = load.Loaded!.Tokenizer;
            int promptTokens = tokenizer.Encode(prompt, addBos: false, addEos: false).Length;
            int oldToolsTokens = tokenizer.Encode(oldToolsJson, addBos: false, addEos: false).Length;

            _output.WriteLine($"agent prompt tokens (compact): {promptTokens}");
            _output.WriteLine($"tools JSON tokens (old indented): {oldToolsTokens}");
            _output.WriteLine($"tool count: {builder.ToolDefinitions.Count}");

            Assert.True(promptTokens <= 1800,
                $"Agent prompt must stay small enough to prefill quickly; got {promptTokens} tokens.");
            Assert.True(promptTokens < oldToolsTokens,
                $"Compact tool listing should be smaller than the old indented dump; got {promptTokens} vs {oldToolsTokens}.");
        }
        finally
        {
            load.Loaded!.Model.Dispose();
        }
    }
}