using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;

namespace SharpMind.Benchmarks.Support;

/// <summary>
/// Replays canned replies in place of a model, so ChatSession's host-side work
/// (prompt build, tokenisation, streaming, markup filtering, tool parse and
/// dispatch, regeneration) is measured without model math. A reply is streamed
/// in the fragments its '|' separators delimit.
/// </summary>
public sealed class ScriptedGenerator : IGenerator<KVCacherBuilder>
{
    private static readonly ConcurrentDictionary<int, Queue<string>> Scripts = new();
    private static int _nextSeed;

    public static int Register(params string[] replies)
    {
        int seed = Interlocked.Increment(ref _nextSeed);
        Scripts[seed] = new Queue<string>(replies);
        return seed;
    }

    private readonly int _seed;
    private readonly Queue<string> _replies;

    public ScriptedGenerator(int seed)
    {
        _seed = seed;
        _replies = Scripts[seed];
    }

    public string Name => "scripted";
    public Action<double>? PrefillProgress { get; set; }
    public float CacheFillRatio => 0;
    public float? TokensPerSecond => null;
    public float? CumulativeTokensPerSecond => null;
    public float? TimeToFirstToken => null;
    public IReadOnlyList<IKVCache> Caches => [];
    public void ResetCache() { }

    public void Dispose() => Scripts.TryRemove(_seed, out _);

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
            // No Task.Yield here: it would add scheduler cost that is not part of
            // the host loop being measured.
        }
    }
}

public sealed class ScriptedGeneratorBuilder : IGeneratorBuilder<KVCacherBuilder>
{
    public IGenerator<KVCacherBuilder> CreateGenerator(Transformer model, Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null, int? maxCacheLen = null)
        => new ScriptedGenerator(seed ?? throw new ArgumentNullException(nameof(seed), "Pass the seed returned by ScriptedGenerator.Register."));
}

/// <summary>
/// A real ChatSession over a scripted generator: tiny random model, byte tokenizer.
/// The model is built once and shared, so per-session cost is the session itself.
/// </summary>
public static class ScriptedSession
{
    private static readonly Lazy<(Tokenizer Tokenizer, Transformer Model)> Shared = new(Build);

    private static (Tokenizer, Transformer) Build()
    {
        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        var tokenizer = Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);

        var cfg = new ModelConfig
        {
            VocabSize = 256, HiddenDim = 8, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 16, MaxSeqLen = 1024,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return (tokenizer, ModelFactory.CreateTrainingTransformer(weights, sharpConfig));
    }

    /// <summary>Tokenizer shared by every scripted session.</summary>
    public static Tokenizer SharedTokenizer => Shared.Value.Tokenizer;

    public static ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder> Create(
        IAgentBuilder? agentBuilder, ToolCallFormat toolCallFormat, params string[] replies)
    {
        var (tokenizer, model) = Shared.Value;
        return new ChatSession<ScriptedGeneratorBuilder, KVCacherBuilder>(
            model, tokenizer, agentBuilder: agentBuilder, seed: ScriptedGenerator.Register(replies),
            disposeModel: false, meta: null, toolCallFormat: toolCallFormat)
        {
            MaxTokens = 8192, // byte tokenizer: one token per character
        };
    }
}
