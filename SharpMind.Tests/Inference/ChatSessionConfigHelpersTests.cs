using SharpMind.Inference;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Tests for <see cref="IChatSession.SetSamplingConfig"/> and
/// <see cref="IChatSession.SetGenerationConfig"/> — the bulk-apply helpers that
/// let a host map a <see cref="SamplingConfig"/>/<see cref="GenerationConfig"/>
/// onto a live session instead of setting each property by hand. These mirror
/// what the CUI's SessionLauncher does at launch.
/// </summary>
public sealed class ChatSessionConfigHelpersTests
{
    [Fact]
    public void SetSamplingConfig_AppliesTemperatureTopKTopP()
    {
        var session = ScriptedSession.Create("ok");

        session.SetSamplingConfig(new SamplingConfig { Temperature = 0.7f, TopK = 9, TopP = 0.3f });

        Assert.Equal(0.7f, session.Temperature);
        Assert.Equal(9, session.TopK);
        Assert.Equal(0.3f, session.TopP);
    }

    [Fact]
    public void SetSamplingConfig_PropagatesToAgentBuilderSamplingConfig()
    {
        var agent = new AgentBuilder("Bob");
        var session = ScriptedSession.Create(agent, "ok");
        var config = new SamplingConfig { Temperature = 0.8f, TopK = 40, TopP = 0.95f };

        session.SetSamplingConfig(config);

        Assert.Same(config, agent.SamplingConfig);
    }

    [Fact]
    public void SetGenerationConfig_AppliesMaxNewTokensAndRepetition()
    {
        var session = ScriptedSession.Create("ok");

        session.SetGenerationConfig(new GenerationConfig
        {
            MaxNewTokens = 77,
            RepetitionPenalty = 1.25f,
            RepetitionWindow = 128,
        });

        Assert.Equal(77, session.MaxNewTokens);
        Assert.Equal(1.25f, session.RepetitionPenalty);
        Assert.Equal(128, session.RepetitionWindow);
    }

    [Fact]
    public void SetGenerationConfig_AppliesStopsWhenSupplied()
    {
        var session = ScriptedSession.Create("ok");

        session.SetGenerationConfig(new GenerationConfig
        {
            StopTokenIds = [1, 2],
            StopStrings = ["</s>", "END"],
        });

        Assert.Equal([1, 2], session.StopTokenIds);
        Assert.Equal(["</s>", "END"], session.StopStrings);
    }

    [Fact]
    public void SetGenerationConfig_LeavesExistingStopsWhenConfigSuppliesNone()
    {
        var session = ScriptedSession.Create("ok");
        session.StopTokenIds = [42];
        session.StopStrings = ["STOP"];

        session.SetGenerationConfig(new GenerationConfig { MaxNewTokens = 5 });

        Assert.Equal([42], session.StopTokenIds);
        Assert.Equal(["STOP"], session.StopStrings);
    }

    [Fact]
    public void SetSamplingConfig_NullThrows()
    {
        var session = ScriptedSession.Create("ok");

        Assert.Throws<ArgumentNullException>(() => session.SetSamplingConfig(null!));
    }

    [Fact]
    public void SetGenerationConfig_NullThrows()
    {
        var session = ScriptedSession.Create("ok");

        Assert.Throws<ArgumentNullException>(() => session.SetGenerationConfig(null!));
    }
}
