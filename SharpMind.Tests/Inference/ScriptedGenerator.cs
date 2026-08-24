using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;

namespace SharpMind.Tests.Inference;

/// <summary>
/// A generator that replays canned replies instead of running a model, so
/// ChatSession's turn logic (history, tool-call handling, streaming) is tested
/// against exact model output. Scripts are keyed by the session seed:
/// ChatSession constructs its generator itself, so the seed is the only value a
/// test can thread through to it. A reply is streamed in the fragments its
/// '|' separators delimit, so a test controls where tags and JSON split.
/// </summary>
internal sealed class ScriptedGenerator : IGenerator<KVCacherBuilder>
{
    private static readonly ConcurrentDictionary<int, Queue<string>> Scripts = new();
    private static int _nextSeed;

    public static int Register(params string[] replies)
    {
        int seed = Interlocked.Increment(ref _nextSeed);
        Scripts[seed] = new Queue<string>(replies);
        return seed;
    }

    private readonly Queue<string> _replies;
    public ScriptedGenerator(int seed) => _replies = Scripts[seed];

    public string Name => "scripted";
    public Action<double>? PrefillProgress { get; set; }
    public float CacheFillRatio => 0;
    public float? TokensPerSecond => null;
    public float? CumulativeTokensPerSecond => null;
    public float? TimeToFirstToken => null;
    public IReadOnlyList<IKVCache> Caches => [];
    public void ResetCache() { }
    public void Dispose() { }

    public IAsyncEnumerable<string> GenerateAsync(string prompt, SamplingConfig? sampling = null, GenerationConfig? generation = null, CancellationToken cancellationToken = default)
        => GenerateFromTokensAsync([], sampling, generation, cancellationToken);

    public async IAsyncEnumerable<string> GenerateFromTokensAsync(int[] promptIds, SamplingConfig? sampling = null, GenerationConfig? generation = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_replies.Count == 0)
            throw new InvalidOperationException("Script exhausted: the session asked for more turns than were registered.");
        foreach (var fragment in _replies.Dequeue().Split('|'))
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return fragment;
            await Task.Yield();
        }
    }
}

internal sealed class ScriptedGeneratorBuilder : IGeneratorBuilder<KVCacherBuilder>
{
    public IGenerator<KVCacherBuilder> CreateGenerator(Transformer model, Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null)
        => new ScriptedGenerator(seed ?? throw new ArgumentNullException(nameof(seed), "Pass the seed returned by ScriptedGenerator.Register."));
}

/// <summary>A real ChatSession over a scripted generator: tiny random model, byte tokenizer, SimpleFormatter.</summary>
internal static class ScriptedSession
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 256, HiddenDim = 8, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 16, MaxSeqLen = 1024,
    };

    public static ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder> Create(params string[] replies)
    {
        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        var tokenizer = Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);

        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        return new ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>(model, tokenizer, seed: ScriptedGenerator.Register(replies), disposeModel: true)
        {
            MaxTokens = 8192,   // byte tokenizer: one token per character; keep the tool prompt clear of trimming
        };
    }
}
