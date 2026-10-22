using SharpMind.Core;
using SharpMind.Core.Memory;
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
/// Absolute positions stop at the model's declared context. Previously a sliding
/// window let positions run past MaxSeqLen and the next RoPE lookup threw an
/// opaque ArgumentOutOfRangeException.
/// </summary>
public sealed class ContextLimitTests
{
    private static ModelConfig Config => new()
    {
        VocabSize = 128,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 8,
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
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static IKVCache[] BuildCaches() =>
        [.. Enumerable.Range(0, Config.NumLayers)
            .Select(_ => (IKVCache)new KVCache(1, Config.NumKvHeads, Config.MaxPosition, Config.HeadDim))];

    private static int[] Prompt(int length = 3) => [.. Enumerable.Range(0, length).Select(i => 100 + i)];

    private static IGrammarConstraint LettersConstraint(int count)
    {
        string grammar = "root ::= " + string.Join(' ', Enumerable.Repeat("[a-z]", count));
        return GbnfGrammar.Parse(grammar).CreateConstraint(ByteTable, [Tokenizer.EosId]);
    }

    private static async Task<string> Generate(
        IGenerator<KVCacherBuilder> generator,
        SamplingConfig sampling,
        int maxNewTokens,
        int[]? prompt = null)
    {
        generator.ResetCache();
        var output = new System.Text.StringBuilder();
        var generation = new GenerationConfig
        {
            MaxNewTokens = maxNewTokens,
            RepetitionPenalty = 1f,
            StopTokenIds = [Tokenizer.EosId],
            Stream = false,
        };

        await foreach (var fragment in generator.GenerateFromTokensAsync(prompt ?? Prompt(), sampling, generation))
            output.Append(fragment);

        return output.ToString();
    }

    [Fact]
    public void Prefill_PromptBeyondContext_ThrowsClearError()
    {
        var caches = BuildCaches();
        using var model = BuildModel();
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(
                Config.HiddenDim, Config.FfnDim, Config.VocabSize, Config.NumLayers, Config.MaxSeqLen));

        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                Prefill.ForwardLastLogitsChunked(
                    model, caches, Prompt(), workspace,
                    startPosition: Config.MaxPosition - 1));

            Assert.Contains("context window", ex.Message);
        }
        finally
        {
            foreach (var cache in caches) cache.Dispose();
        }
    }

    [Fact]
    public async Task StandardGenerator_StopsAtContextLimit()
    {
        using var model = BuildModel();
        using var generator = new StandardGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);

        int generated = 0;
        generator.OnTokenGenerated += _ => generated++;

        await Generate(generator, SamplingConfig.Greedy, maxNewTokens: 20);

        Assert.Equal(Config.MaxPosition - Prompt().Length, generated);
    }

    [Fact]
    public async Task SpeculativeGenerator_StopsAtContextLimit()
    {
        using var model = BuildModel();
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);

        string output = await Generate(generator,
            new SamplingConfig { Temperature = 1f, TopK = 2, Constraint = LettersConstraint(20) },
            maxNewTokens: 20);

        Assert.InRange(output.Length, 1, Config.MaxPosition - Prompt().Length);
    }

    [Fact]
    public async Task SpeculativeGenerator_UsesRemainingContextTail()
    {
        using var model = BuildModel();
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(model, Tokenizer,
            addBos: false, addEos: false, seed: 1);
        int[] prompt = Prompt(Config.MaxPosition - 2);

        await Generate(generator,
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint(20) },
            maxNewTokens: 20, prompt);

        Assert.Equal(Config.MaxPosition - prompt.Length, generator.CurrentGeneratedIds!.Count);
    }

    [Fact]
    public async Task MedusaGenerator_UsesRemainingContextTail()
    {
        using var model = BuildModel();
        using var generator = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, Tokenizer, addBos: false, addEos: false, caches: null, seed: 1);
        int[] prompt = Prompt(Config.MaxPosition - 2);

        await Generate(generator,
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint(20) },
            maxNewTokens: 20, prompt);

        Assert.Equal(Config.MaxPosition - prompt.Length, generator.CurrentGeneratedIds!.Count);
    }

    [Fact]
    public async Task MedusaGenerator_StopsAtContextLimit()
    {
        using var model = BuildModel();
        using var generator = new MedusaGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, Tokenizer, addBos: false, addEos: false, caches: null, seed: 1);

        string output = await Generate(generator,
            new SamplingConfig { Temperature = 0f, Constraint = LettersConstraint(20) },
            maxNewTokens: 20);

        Assert.InRange(output.Length, 1, Config.MaxPosition - Prompt().Length);
    }
}
