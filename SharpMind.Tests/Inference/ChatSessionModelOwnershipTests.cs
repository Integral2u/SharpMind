using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// A ChatSession must not dispose a Transformer it was handed. The default used
/// to be disposeModel: true, and ChatSessionFactory hardcoded true, so ending one
/// chat disposed the caller's shared model. Every later session then threw
/// ObjectDisposedException inside generation, which StartChatAsync's bare catch
/// swallowed — the user saw an empty reply and no error.
/// </summary>
public sealed class ChatSessionModelOwnershipTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 300,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 32,
    };

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static void AssertModelUsable(Transformer model)
    {
        using var input = Tensor<int>.From([1, 2, 3], 1, 3);
        using var logits = model.Forward(input, null, 0, null);
        Assert.Equal(Cfg.VocabSize, logits.Shape[^1]);
    }

    [Fact]
    public async Task DisposingSession_LeavesCallerOwnedModelUsable()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        await using (var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            model, tokenizer))
        {
            session.InitializeChat();
        }

        AssertModelUsable(model);
    }

    [Fact]
    public async Task SecondSession_CanBeBuiltOnTheSameModel()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        for (int i = 0; i < 3; i++)
        {
            await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
                model, tokenizer);
            session.InitializeChat();
            AssertModelUsable(session.Model);
        }
    }

    [Fact]
    public async Task Factory_DoesNotTakeOwnershipByDefault()
    {
        using var model = BuildModel();
        var tokenizer = BuildTokenizer();

        await using (var session = ChatSessionFactory.CreateChatSession(
            typeof(StandardGeneratorBuilder<>), typeof(KVCacherBuilder), model, tokenizer))
        {
            session.InitializeChat();
        }

        AssertModelUsable(model);
    }

    [Fact]
    public async Task DisposeModelTrue_StillTransfersOwnership()
    {
        var model = BuildModel();
        var tokenizer = BuildTokenizer();

        await using (var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
            model, tokenizer, disposeModel: true))
        {
            session.InitializeChat();
        }

        using var input = Tensor<int>.From([1, 2, 3], 1, 3);
        Assert.Throws<ObjectDisposedException>(() => model.Forward(input, null, 0, null));
    }
}
