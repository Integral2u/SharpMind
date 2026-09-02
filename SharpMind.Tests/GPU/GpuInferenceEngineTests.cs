using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.GPU;

/// <summary>
/// The GPU inference engine: the whole forward — first prefill, continued prefill, and
/// KV-cache-efficient single-token decode — runs on the device against a persistent device
/// K/V cache. The load-bearing assertion is that an <see cref="EngineGenerator{T}"/> backed by
/// the engine generates the SAME greedy token sequence as a pure-CPU <see cref="StandardGenerator{T}"/>
/// on the same model — the GPU path (including decode, which previous versions proxied on the
/// CPU) must never change what the model says. Runs on <see cref="GpuTestDevice.Device"/>,
/// which is ILGPU's CPU accelerator when no GPU is present, so no real hardware is needed.
/// </summary>
[Collection("GPU")]
public sealed class GpuInferenceEngineTests : IClassFixture<GpuInferenceEngineTests.ModelFixture>
{
    // qwen2 makes AttentionLayer.UsesNeoxRope true (RoPE required by ValidateSupported's
    // "RoPE or NoPE" rule) and the Llama preset gives SiLU/SwiGLU gated FFN + RMSNorm — the
    // exact shape GpuInferenceEngine.ValidateSupported accepts.
    private static ModelConfig Cfg => new()
    {
        VocabSize = 256, HiddenDim = 64, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, FfnDim = 128, MaxSeqLen = 64,
        Architecture = "qwen2",
    };
    private const int MaxCacheLength = 64;

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly SharpMindConfig _config;

    public GpuInferenceEngineTests(ModelFixture fixture)
    {
        _model = fixture.Model;
        _tokenizer = fixture.Tokenizer;
        _config = fixture.Config;
    }

    public sealed class ModelFixture : IDisposable
    {
        public Transformer Model { get; }
        public Tokenizer Tokenizer { get; }
        public SharpMindConfig Config { get; }

        public ModelFixture()
        {
            Config = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
            var weights = ModelFactory.CreateForTraining(Cfg, Config);
            WeightInitializer.InitializeRandomly(weights, 9001);
            Model = ModelFactory.CreateTrainingTransformer(weights, Config);
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

    private GpuInferenceEngine BuildEngine() =>
        new(GpuTestDevice.Device, _model, _config, MaxCacheLength, maxPromptTokens: 32);

    [Fact]
    public void Description_SurfacesTheBackingDevice()
    {
        // The engine-usage display contract: the engine names the accelerator it actually runs on,
        // so the CUI can show OpenCL / ILGPU-CUDA / cuBLAS / CPU to the user.
        using var engine = BuildEngine();
        var dev = GpuTestDevice.Device;
        Assert.Equal(dev.IsCpuFallback ? "CPU" : dev.Description, engine.Description);
        Assert.False(string.IsNullOrWhiteSpace(engine.Description));
    }

    /// <summary>A small, printable-ASCII prompt that decodes cleanly and stays under maxPromptTokens.</summary>
    private static int[] Prompt() => [65, 66, 67, 32, 68, 69]; // "ABC DE"

    private static async Task<List<int>> GreedyIds(IGenerator<KVCacherBuilder> gen, int[] prompt, int maxNew)
    {
        // Greedy + no repetition penalty. The generated token ids are read from the interface's
        // CurrentGeneratedIds after the (non-streamed) turn completes — identical on both paths.
        await foreach (var _ in gen.GenerateFromTokensAsync(prompt, sampling: SamplingConfig.Greedy,
            generation: new GenerationConfig { MaxNewTokens = maxNew, Stream = false }))
        {
        }
        return gen.CurrentGeneratedIds?.ToList() ?? new List<int>();
    }

    /// <summary>
    /// An engine-backed generator must reproduce a pure-CPU generator's greedy decode one-for-one.
    /// This exercises GPU first-prefill (empty cache) → CPU decode (DecodeStep against the
    /// materialised host cache) for a whole turn.
    /// </summary>
    [Fact]
    public async Task EngineGenerator_MatchesCpuGenerator_OnFirstTurn()
    {
        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        var engineIds = await GreedyIds(engineGen, Prompt(), maxNew: 12);
        var cpuIds = await GreedyIds(cpuGen, Prompt(), maxNew: 12);

        Assert.NotEmpty(engineIds);
        Assert.Equal(cpuIds, engineIds);

        // The engine's cache must have advanced: prompt + generated tokens resident (contiguous).
        Assert.Equal(Prompt().Length + engineIds.Count, engine.CachedLength);
    }

    /// <summary>
    /// A second turn with a warm cache must continue from the first turn's decoder state, exactly
    /// like the CPU generator does. The engine's continued prefill runs on the CPU (PrefillCpu)
    /// against the materialised host cache — never recomputing the first-turn prefix.
    /// </summary>
    [Fact]
    public async Task EngineGenerator_MatchesCpuGenerator_OnSecondTurn()
    {
        int[] turn1 = Prompt();
        int[] turn2 = [70, 71, 32, 72]; // "FG H"

        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        // First turn both ways.
        await GreedyIds(engineGen, turn1, maxNew: 6);
        await GreedyIds(cpuGen, turn1, maxNew: 6);

        var engineTurn2 = await GreedyIds(engineGen, turn2, maxNew: 6);
        var cpuTurn2 = await GreedyIds(cpuGen, turn2, maxNew: 6);

        Assert.NotEmpty(engineTurn2);
        Assert.Equal(cpuTurn2, engineTurn2);
    }

    /// <summary>
    /// A first prompt larger than maxPromptTokens must route through the CPU path rather than
    /// throwing — the engine degrades gracefully instead of failing an over-long first turn.
    /// </summary>
    [Fact]
    public async Task OverLongFirstPrompt_RoutesCpu_AndStillMatches()
    {
        var longPrompt = new int[40]; // > maxPromptTokens (32), < MaxCacheLength (64)
        for (int i = 0; i < longPrompt.Length; i++) longPrompt[i] = 65 + (i % 60);

        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        var engineIds = await GreedyIds(engineGen, longPrompt, maxNew: 8);
        var cpuIds = await GreedyIds(cpuGen, longPrompt, maxNew: 8);

        Assert.NotEmpty(engineIds);
        Assert.Equal(cpuIds, engineIds);
    }

    /// <summary>A longer generation forces many single-token GPU decode steps against a device cache
    /// that grows each step — exercising the positioned KV-cache-efficient attention on every token
    /// beyond the first. The generated sequence must still match the pure-CPU generator one-for-one.</summary>
    [Fact]
    public async Task EngineGenerator_LongDecode_MatchesCpuGenerator()
    {
        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        var engineIds = await GreedyIds(engineGen, Prompt(), maxNew: 20);
        var cpuIds = await GreedyIds(cpuGen, Prompt(), maxNew: 20);

        Assert.NotEmpty(engineIds);
        Assert.Equal(cpuIds, engineIds);

        // Every decoded token advanced the engine's cache (prompt + generated) on device.
        Assert.Equal(Prompt().Length + engineIds.Count, engine.CachedLength);
    }

    /// <summary>A third turn runs GPU continued prefill on top of an already GPU-decoded cache and
    /// must still track the CPU generator, which continues from its own matching state.</summary>
    [Fact]
    public async Task ThirdTurn_GpuContinuedPrefill_StillMatchesCpuGenerator()
    {
        int[] turn1 = Prompt();
        int[] turn3 = [73, 74, 75]; // "IJK"

        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        await GreedyIds(engineGen, turn1, maxNew: 5);
        await GreedyIds(cpuGen, turn1, maxNew: 5);

        var engineTurn3 = await GreedyIds(engineGen, turn3, maxNew: 6);
        var cpuTurn3 = await GreedyIds(cpuGen, turn3, maxNew: 6);

        Assert.NotEmpty(engineTurn3);
        Assert.Equal(cpuTurn3, engineTurn3);
    }

    /// <summary>Import restores a saved cache into a fresh engine; the next turn (GPU continued
    /// prefill + decode against the re-synced device cache) must produce the same greedy output a
    /// CPU generator produces from the same prompt — proving the device cache is rewritten from the
    /// imported host cache before the GPU attention reads it.</summary>
    [Fact]
    public async Task ImportedCache_GpuContinuedPrefill_MatchesCpuGenerator()
    {
        var prompt = Prompt();

        using var source = BuildEngine();
        source.Prefill(prompt);
        var snapshot = source.ExportCache([.. prompt]);

        using var engine = BuildEngine();
        engine.ImportCache(snapshot);
        Assert.Equal(source.CachedLength, engine.CachedLength);

        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);
        using var cpuGen = new StandardGenerator<KVCacherBuilder>(_model, _tokenizer, addBos: false, addEos: false, seed: 1);

        var engineIds = await GreedyIds(engineGen, prompt, maxNew: 5);
        var cpuIds = await GreedyIds(cpuGen, prompt, maxNew: 5);

        Assert.NotEmpty(engineIds);
        Assert.Equal(cpuIds, engineIds);
    }

    /// <summary>Reset empties the cache so a fresh conversation starts its own GPU prefill cleanly.</summary>
    [Fact]
    public async Task Reset_ClearsCache_AndEnablesFreshGpuPrefill()
    {
        using var engine = BuildEngine();
        using var engineGen = new EngineGenerator<KVCacherBuilder>(engine, _tokenizer, addBos: false, addEos: false, numLayers: Cfg.NumLayers, seed: 1);

        await GreedyIds(engineGen, Prompt(), maxNew: 4);
        Assert.True(engine.CachedLength > 0);

        // Cache-control surface the host also calls, on a non-empty cache.
        engine.TrimToLast(1);
        Assert.Equal(1, engine.CachedLength);
        Assert.False(engine.IsCacheFull);

        engineGen.ResetCache();
        Assert.Equal(0, engine.CachedLength);
        Assert.False(engine.IsCacheFull);
    }

    /// <summary>Export/Import round-trips the whole cache through the host blob format a session save uses.</summary>
    [Fact]
    public void ExportImport_RoundTrips_MatchingLayers()
    {
        using var engine = BuildEngine();
        var prompt = Prompt();

        engine.Prefill(prompt);
        var snapshot = engine.ExportCache([.. prompt]);

        Assert.Equal(prompt.Length, snapshot.PromptTokenCount);
        Assert.Equal(Cfg.NumLayers, snapshot.Layers.Count);

        using var engine2 = BuildEngine();
        engine2.ImportCache(snapshot);
        Assert.Equal(engine.CachedLength, engine2.CachedLength);
    }

    /// <summary>Import of a snapshot with the wrong layer count is rejected, not silently truncated.</summary>
    [Fact]
    public void ImportCache_RejectsForeignLayerCount()
    {
        using var engine = BuildEngine();
        var snapshot = engine.ExportCache([.. Prompt()]);
        snapshot.Layers.RemoveAt(snapshot.Layers.Count - 1);

        Assert.Throws<ArgumentException>(() => engine.ImportCache(snapshot));
    }

    [Fact]
    public void ValidateSupported_RejectsWhatItDoesNotImplement()
    {
        void Reject(SharpMindConfig sc, string needle)
        {
            sc = sc with { Hardware = HardwareTier.Scalar };
            var weights = ModelFactory.CreateForTraining(Cfg, sc);
            using var model = ModelFactory.CreateTrainingTransformer(weights, sc);
            var ex = Assert.Throws<NotSupportedException>(() => GpuInferenceEngine.ValidateSupported(model, sc));
            Assert.Contains(needle, ex.Message, StringComparison.OrdinalIgnoreCase);
        }

        Reject(SharpMindConfig.Llama with { Gate = GateKind.None, Ffn = FfnKind.Dense }, "dense");
        Reject(SharpMindConfig.ForModel(4, 4, "mixtral") with { Ffn = FfnKind.MoE }, "MoE");
        Reject(SharpMindConfig.Llama with { Norm = NormKind.LayerNorm }, "RMSNorm");
    }

    /// <summary>A model built for the standard chat loader (CreateTransformer) wires
    /// InferenceLinearLayer whose "weight" only backs RawQuantizedData — running the GPU F32 GEMMs on
    /// it reads far past the tensor. The engine must refuse it up front (so the picker can offer CPU)
    /// rather than crash on the first forward, as it used to ("B holds N floats, GEMM needs M").</summary>
    [Fact]
    public void ValidateSupported_RejectsQuantizedResidentModel()
    {
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sc);
        using var inference = ModelFactory.CreateTransformer(weights, sc, optimizeMemory: false);

        var ex = Assert.Throws<NotSupportedException>(() => GpuInferenceEngine.ValidateSupported(inference, sc));
        Assert.Contains("quantized", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("F32", ex.Message);
    }

    private static ModelMetaData Meta(params (string Name, QuantDType Dtype)[] tensors) => new()
    {
        Tensors = [.. tensors.Select(t => new TensorInfo { Name = t.Name, Dtype = t.Dtype, Shape = [8, 8], Offset = 0 })],
    };

    /// <summary>The metadata-only, pre-weight-load gate accepts the same dtypes the engine runs
    /// (F32 + the on-device block quants Q8_0/Q4_0/Q4_1/Q5_0/Q5_1) on the same architecture the
    /// built Transformer accepts.</summary>
    [Fact]
    public void CheckSupported_AcceptsF32AndQ8_0FromMetadata()
    {
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
        var meta = Meta(("token_embd.weight", QuantDType.Q8_0), ("blk.0.ffn_down.weight", QuantDType.Q8_0), ("output_norm.weight", QuantDType.F32));

        Assert.True(GpuInferenceEngine.CheckSupported(meta, Cfg, sc, out var reason), reason);
    }

    /// <summary>A quant the engine has no kernel for (Q6_K here) must be refused from mere metadata —
    /// this is the pre-load gate that keeps the host from loading a whole file it will reject at creation.</summary>
    [Fact]
    public void CheckSupported_RejectsMetadataWithUnsupportedQuant()
    {
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };
        var meta = Meta(("blk.0.attn_wq.weight", QuantDType.Q6_K));

        Assert.False(GpuInferenceEngine.CheckSupported(meta, Cfg, sc, out var reason));
        Assert.Contains("Q6_K", reason, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The plugin factory forwards the metadata gate so the host can refuse before loading weights.</summary>
    [Fact]
    public void Factory_CheckSupported_RefusesUnsupportedQuantMeta()
    {
        var factory = new GpuInferenceEngineFactory();
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Scalar };

        Assert.Null(factory.CheckSupported(Meta(("a", QuantDType.Q8_0)), Cfg, sc));

        var reason = factory.CheckSupported(Meta(("a", QuantDType.Q6_K)), Cfg, sc);
        Assert.NotNull(reason);
        Assert.Contains("Q6_K", reason, StringComparison.OrdinalIgnoreCase);
    }
}
