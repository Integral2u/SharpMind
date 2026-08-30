using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Server;
using SharpMind.Server.Protocol;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Server;

/// <summary>
/// Guards the request -> session mapping. The server had no coverage here at all,
/// which is how it shipped answering prompts the caller never sent.
/// </summary>
public class SessionFactoryTests
{
    private const int ModelMaxSeqLen = 1024;

    private static LoadedModel MakeLoadedModel()
    {
        var cfg = new ModelConfig
        {
            VocabSize = 256, HiddenDim = 8, NumLayers = 1, NumHeads = 2,
            NumKvHeads = 2, FfnDim = 16, MaxSeqLen = ModelMaxSeqLen,
        };

        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        var tokenizer = Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);

        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        return new LoadedModel
        {
            ModelId = "test-model",
            FilePath = "test-model.gguf",
            Model = model,
            Tokenizer = tokenizer,
            Meta = new ModelMetaData(),
        };
    }

    private static CreateChatCompletionRequest Request(int? maxTokens, string userText) => new()
    {
        Model = "test-model",
        MaxTokens = maxTokens,
        Messages = [new UserMessage { Content = userText }],
    };

    /// <summary>
    /// `max_tokens` is a GENERATION limit, not a context window. Feeding it into
    /// <see cref="IChatSession.MaxTokens"/> made <c>TrimToFitContext</c> compute
    /// <c>contextBudget = MaxTokens - MaxNewTokens = 0</c>, fall back to
    /// <c>MaxTokens / 2</c>, and cut the prompt to a handful of tokens — so every
    /// completion answered a prompt the caller never sent.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(256)]
    public void CreateSession_KeepsFullContextWindow_RegardlessOfMaxTokens(int maxTokens)
    {
        using var loaded = MakeLoadedModel();
        var factory = new SessionFactory(new SharpMindServerOptions());

        var (session, _) = factory.CreateSession(loaded, Request(maxTokens, "hello"));

        int expected = ModelConfig.ComputeMaxCacheLength(loaded.Model.Config, null);

        Assert.Equal(maxTokens, session.MaxNewTokens);
        Assert.Equal(expected, session.MaxTokens);
    }

    /// <summary>
    /// The trim budget is <c>MaxTokens - MaxNewTokens</c>, so a prompt this long
    /// must still fit. Before the fix the budget was 8 tokens and the message was
    /// discarded entirely.
    /// </summary>
    [Fact]
    public void CreateSession_LeavesRoomForTheActualPrompt()
    {
        using var loaded = MakeLoadedModel();
        var factory = new SessionFactory(new SharpMindServerOptions());

        // Byte tokenizer: one token per character.
        string userText = new('x', 400);
        var (session, lastUserMessage) = factory.CreateSession(loaded, Request(16, userText));

        Assert.Equal(userText, lastUserMessage);
        Assert.True(
            session.MaxTokens - session.MaxNewTokens >= userText.Length,
            $"context budget {session.MaxTokens - session.MaxNewTokens} cannot hold a {userText.Length}-token prompt");
    }

    /// <summary>Omitting max_tokens must not change the context window either.</summary>
    [Fact]
    public void CreateSession_WithoutMaxTokens_UsesCacheLength()
    {
        using var loaded = MakeLoadedModel();
        var factory = new SessionFactory(new SharpMindServerOptions());

        var (session, _) = factory.CreateSession(loaded, Request(null, "hello"));

        Assert.Equal(ModelConfig.ComputeMaxCacheLength(loaded.Model.Config, null), session.MaxTokens);
    }
}
