using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Regression: the generators used to prefill the entire prompt in one
/// <see cref="Transformer.ForwardLastLogits"/> call, but each generator's
/// workspace is sized for at most 128 prefill tokens (the
/// <c>min(MaxSeqLen, 128)</c> cap in Workspace.CalculateRequiredSize). A longer
/// prompt overflowed the workspace's bump allocator with an
/// <c>OutOfMemoryException</c>, which ChatSession's catch-all then surfaced as an
/// empty reply with no stream. Chunked prefill must let long prompts stream.
///
/// This model keeps the workspace at its 100 MB floor (~89 MB of 128-token
/// budget), while a single-shot prefill of the 500-token prompt needs ~214 MB
/// (the guaranteed per-token floor is ~107K floats for 13 layers). So the test
/// is red without the chunking and green with it; each 64-token chunk needs
/// only ~45 MB.
/// </summary>
public sealed class GeneratorPrefillTests : IClassFixture<GeneratorPrefillTests.ModelFixture>
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 300,
        HiddenDim = 512,
        NumLayers = 13,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 1536,
        MaxSeqLen = 2048,
    };

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;

    public GeneratorPrefillTests(ModelFixture fixture)
    {
        _model = fixture.Model;
        _tokenizer = fixture.Tokenizer;
    }

    public sealed class ModelFixture : IDisposable
    {
        public Transformer Model { get; }
        public Tokenizer Tokenizer { get; }

        public ModelFixture()
        {
            var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
            var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
            WeightInitializer.InitializeRandomly(weights, 1234);
            Model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
            Tokenizer = BuildTokenizer();
        }

        public void Dispose() => Model.Dispose();
    }

    private static Tokenizer BuildTokenizer()
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    }

    /// <summary>
    /// A prompt well past the workspace's single-shot prefill budget
    /// (~234 tokens at the 100 MB floor for this model's guaranteed usage).
    /// </summary>
    private static int[] BuildLongPrompt()
    {
        var prompt = new int[500];
        for (int i = 0; i < prompt.Length; i++)
            prompt[i] = 65 + (i % 60); // printable ASCII byte tokens, never specials
        return prompt;
    }

    private static async Task<int> CountStreamedFragments(IGenerator<KVCacherBuilder> generator)
    {
        int fragments = 0;
        var generation = new GenerationConfig { MaxNewTokens = 16 };
        await foreach (var fragment in generator.GenerateFromTokensAsync(BuildLongPrompt(), generation: generation))
        {
            if (fragment.Length > 0) fragments++;
        }
        return fragments;
    }

    [Fact]
    public async Task StandardGenerator_LongPrompt_StreamsWithoutOverflowingWorkspace()
    {
        using var generator = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        int fragments = await CountStreamedFragments(generator);

        Assert.True(fragments > 0, "Long-prompt generation should stream tokens; prefill must not overflow the workspace.");
    }

    [Fact]
    public async Task SpeculativeGenerator_LongPrompt_StreamsWithoutOverflowingWorkspace()
    {
        using var generator = new SpeculativeGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        int fragments = await CountStreamedFragments(generator);

        Assert.True(fragments > 0, "Long-prompt generation should stream tokens; prefill must not overflow the workspace.");
    }

    [Fact]
    public async Task MedusaGenerator_LongPrompt_StreamsWithoutOverflowingWorkspace()
    {
        using var generator = new MedusaGeneratorBuilder<KVCacherBuilder>().CreateGenerator(
            _model, _tokenizer, addBos: false, addEos: false, caches: null, seed: 1);

        int fragments = await CountStreamedFragments(generator);

        Assert.True(fragments > 0, "Long-prompt generation should stream tokens; prefill must not overflow the workspace.");
    }
}
