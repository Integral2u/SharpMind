using System.Text;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Grammar;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Speculative and Medusa rejection rolls recurrent short-conv state back by
/// restoring a snapshot and replaying accepted tokens. The replay must rebuild
/// exactly the state a plain greedy forward pass would have produced.
/// </summary>
public sealed class ShortConvRollbackTests
{
    private const int MaxNewTokens = 24;

    private static ModelConfig Config => new()
    {
        Architecture = "lfm2",
        VocabSize = 128,
        HiddenDim = 32,
        NumLayers = 4,
        NumHeads = 4,
        NumKvHeads = 2,
        LayerKvHeads = [2, 0, 2, 0],
        FfnDim = 64,
        ShortConvCacheLength = 3,
        MaxSeqLen = 64,
    };

    private static readonly Tokenizer Tokenizer = BuildTokenizer();
    private static readonly TokenByteTable ByteTable = TokenByteTable.Get(Tokenizer);

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++)
            tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.ForModel(
            Config.NumHeads, Config.NumKvHeads, Config.Architecture, HardwareTier.Scalar);
        var weights = ModelFactory.CreateForTraining(Config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 4321);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static IGrammarConstraint LettersConstraint()
    {
        string grammar = "root ::= " + string.Join(' ', Enumerable.Repeat("[a-z]", MaxNewTokens));
        return GbnfGrammar.Parse(grammar).CreateConstraint(ByteTable, [Tokenizer.EosId]);
    }

    private static async Task<string> Generate(
        IGenerator<KVCacherBuilder> generator,
        int[] prompt,
        SamplingConfig sampling)
    {
        generator.ResetCache();
        var output = new StringBuilder();
        var generation = new GenerationConfig
        {
            MaxNewTokens = MaxNewTokens,
            RepetitionPenalty = 1f,
            StopTokenIds = [Tokenizer.EosId],
            Stream = false,
        };

        await foreach (var fragment in generator.GenerateFromTokensAsync(prompt, sampling, generation))
            output.Append(fragment);

        return output.ToString();
    }

    private static int[] Prompt() => [100, 101, 102]; // "abc"

    [Fact]
    public async Task SpeculativeRejection_ReplaysShortConvStateLikeGreedy()
    {
        using var model = BuildModel();
        using var standard = new StandardGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);
        using var speculative = new SpeculativeGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);

        string expected = await Generate(standard, Prompt(),
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint() });

        // TopK=2 makes the draft differ from greedy often enough to reject, while
        // accepted drafts are still exactly the greedy token.
        string actual = await Generate(speculative, Prompt(),
            new SamplingConfig { Temperature = 1f, TopK = 2, Constraint = LettersConstraint() });

        Assert.Equal(MaxNewTokens, expected.Length);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public async Task MedusaRejection_ReplaysShortConvStateLikeGreedy()
    {
        using var model = BuildModel();
        using var standard = new StandardGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);
        using var medusa = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, Tokenizer, addBos: false, addEos: false, caches: null, seed: 1);

        string expected = await Generate(standard, Prompt(),
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint() });

        string actual = await Generate(medusa, Prompt(),
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint() });

        Assert.Equal(MaxNewTokens, expected.Length);
        Assert.Equal(expected, actual);
    }
}
