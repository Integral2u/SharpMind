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
///
/// Size is measured in characters rather than BPE tokens: a real checkpoint is
/// deliberately not loaded (see <see cref="TinyReferenceModel"/>), and the
/// reference tokenizer would answer in a token unit that has no correspondence
/// to a real vocabulary. Characters are tokenizer-independent and stable, and
/// the property under test — the compact listing must shrink the verbose dump
/// and stay small in absolute terms — is faithfully expressed by them.
/// </summary>
public sealed class AgentPromptSizeTests
{
    private readonly ITestOutputHelper _output;

    public AgentPromptSizeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CompactToolListing_KeepsAgentPromptSmall()
    {
        using var temp = new TempDirectory();
        var options = new SessionOptions
        {
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
        };
        options.ModelPath = TinyReferenceModel.Create(temp).SmmPath;

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");
        using var model = load.Loaded!.Model;

        var builder = new AgentBuilder("Delta")
        {
            DisabledTools = []
        };
        builder.WithTools(new CuiTools(new CuiToolContext()));
        builder.WithTools(new WeatherTool());
        builder.WithTools(new FileSystemTool(Path.GetTempPath()));

        string prompt = builder.BuildAgentPrompt();
        string oldToolsJson = builder.ToolDefinitions.ToJsonString(new JsonSerializerOptions { WriteIndented = true });

        int promptChars = prompt.Length;
        int oldToolsChars = oldToolsJson.Length;

        _output.WriteLine($"agent prompt chars (compact): {promptChars}");
        _output.WriteLine($"tools JSON chars (old indented): {oldToolsChars}");
        _output.WriteLine($"tool count: {builder.ToolDefinitions.Count}");

        // The compact one-line listing drops the whitespace/newline overhead of
        // the indented dump; it must always be smaller than that dump.
        Assert.True(promptChars < oldToolsChars,
            $"Compact tool listing should be smaller than the old indented dump; got {promptChars} vs {oldToolsChars} chars.");

        // Absolute growth guard: with the current CUI tool set this sits well
        // under 20k characters (~5 k real-vocab tokens at a 4 chars/token ratio).
        // The bound exists to stop a future tool description or prompt section
        // from silently tripling prefill time — if the tool set grows
        // deliberately, the bound should be re-calibrated with the new count.
        Assert.True(promptChars <= 30_000,
            $"Agent prompt must stay small enough to prefill quickly; got {promptChars} chars.");
    }
}