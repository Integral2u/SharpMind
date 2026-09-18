using System.Text;
using System.Text.RegularExpressions;
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
/// End-to-end check that a grammar constraint actually bounds what a real
/// generator emits. A tiny 1-layer model supplies arbitrary logits; the grammar
/// is what makes the output deterministic, so these assertions do not depend on
/// the random weights.
/// </summary>
public sealed class GrammarGenerationTests
{
    private const int VocabSize = 3 + 256; // specials + one token per byte

    private static readonly Tokenizer Tokenizer = Tokenizer.FromGguf(
        BuildTokens(), merges: null, tokenTypes: null, bosId: 1, eosId: 2);

    private static readonly TokenByteTable ByteTable = TokenByteTable.Get(Tokenizer);

    private static string[] BuildTokens()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++)
            tokens.Add(Vocabulary.ByteTokenString(b));
        return [.. tokens];
    }

    private static Transformer BuildModel()
    {
        var config = new ModelConfig
        {
            VocabSize = VocabSize,
            HiddenDim = 32,
            NumLayers = 1,
            NumHeads = 4,
            NumKvHeads = 4,
            FfnDim = 64,
            MaxSeqLen = 64,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 4321);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static GenerationConfig GenCfg => new()
    {
        MaxNewTokens = 8,
        StopTokenIds = [Tokenizer.EosId],
        RepetitionPenalty = 1.0f,
        Stream = false,
    };

    private static IGrammarConstraint Constraint(string gbnf)
        => GbnfGrammar.Parse(gbnf).CreateConstraint(ByteTable, [Tokenizer.EosId]);

    private static async Task<string> Run(StandardGenerator<KVCacherBuilder> generator, IGrammarConstraint constraint)
    {
        generator.ResetCache();
        var sampling = new SamplingConfig { Temperature = 0f, Constraint = constraint };
        var sb = new StringBuilder();
        await foreach (var fragment in generator.GenerateFromTokensAsync([Tokenizer.BosId], sampling, GenCfg))
            sb.Append(fragment);
        return sb.ToString();
    }

    [Fact]
    public async Task LiteralGrammar_ForcesExactOutput()
    {
        using var model = BuildModel();
        using var generator = new StandardGenerator<KVCacherBuilder>(model, Tokenizer, addBos: false, addEos: false, seed: 1);

        string output = await Run(generator, Constraint("root ::= \"ab\""));

        Assert.Equal("ab", output);
    }

    [Fact]
    public async Task ClassGrammar_ProducesOnlyMatchingCharacters()
    {
        using var model = BuildModel();
        using var generator = new StandardGenerator<KVCacherBuilder>(model, Tokenizer, addBos: false, addEos: false, seed: 1);

        string output = await Run(generator, Constraint("root ::= [a-c] [0-9]"));

        Assert.Matches(new Regex("^[a-c][0-9]$"), output);
    }

    [Fact]
    public async Task Constraint_IsResetBetweenGenerations()
    {
        using var model = BuildModel();
        using var generator = new StandardGenerator<KVCacherBuilder>(model, Tokenizer, addBos: false, addEos: false, seed: 1);
        var constraint = Constraint("root ::= \"ab\"");

        Assert.Equal("ab", await Run(generator, constraint));
        Assert.Equal("ab", await Run(generator, constraint));
    }

    [Fact]
    public async Task CompletedGrammar_StopsImmediatelyOnEos()
    {
        using var model = BuildModel();
        using var generator = new StandardGenerator<KVCacherBuilder>(model, Tokenizer, addBos: false, addEos: false, seed: 1);

        string output = await Run(generator, Constraint("root ::= \"a\""));

        Assert.Equal("a", output);
        Assert.Equal(new[] { 3 + 'a', Tokenizer.EosId }, generator.CurrentGeneratedIds);
    }

    private static async Task<string> RunGenerator(IGenerator<KVCacherBuilder> generator, IGrammarConstraint constraint)
    {
        generator.ResetCache();
        var sampling = new SamplingConfig { Temperature = 0f, Constraint = constraint };
        var sb = new StringBuilder();
        await foreach (var fragment in generator.GenerateFromTokensAsync([Tokenizer.BosId], sampling, GenCfg))
            sb.Append(fragment);
        return sb.ToString();
    }

    [Fact]
    public async Task SpeculativeGenerator_RespectsGrammar()
    {
        using var model = BuildModel();
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(model, Tokenizer, addBos: false, addEos: false, seed: 1);

        string output = await RunGenerator(generator, Constraint("root ::= \"abc\""));

        Assert.Equal("abc", output);
    }

    [Fact]
    public async Task MedusaGenerator_RespectsGrammar()
    {
        using var model = BuildModel();
        using var generator = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, Tokenizer, addBos: false, addEos: false, caches: null, seed: 1);

        string output = await RunGenerator(generator, Constraint("root ::= \"abc\""));

        Assert.Equal("abc", output);
    }
}