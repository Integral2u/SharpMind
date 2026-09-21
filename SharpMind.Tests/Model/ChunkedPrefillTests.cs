using System.Reflection;
using SharpMind.Core;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The workspace is a bump allocator sized for at most
/// Workspace.MaxPrefillTokens tokens per forward pass, but every generator
/// used to prefill the whole prompt in one pass. Any prompt longer than that
/// — i.e. every real chat prompt, since the agent system prompt alone runs
/// well over a thousand tokens — died with "Workspace capacity exceeded"
/// partway through the first turn, which the CUI then swallowed into a silent
/// "Thinking..." forever. <see cref="Prefill.ForwardLastLogitsChunked"/> feeds
/// the prompt through in chunks of at most <see cref="Prefill.MaxChunkLength"/>.
/// </summary>
public sealed class ChunkedPrefillTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 128,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 1024,
    };

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static IKVCache[] BuildCaches() =>
        [.. Enumerable.Range(0, Cfg.NumLayers)
            .Select(_ => (IKVCache)new KVCache(1, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDim))];

    private static int[] BuildPrompt(int length)
    {
        var rng = new Random(7);
        return [.. Enumerable.Range(0, length).Select(_ => rng.Next(Cfg.VocabSize))];
    }

    [Fact]
    public void ForwardLastLogitsChunked_MatchesOneShotForward_WhenPromptExceedsChunkSize()
    {
        int promptLen = Prefill.MaxChunkLength * 2 + 37; // spans several chunks, ragged last one
        int[] promptIds = BuildPrompt(promptLen);

        using var model = BuildModel();

        // Reference: one pass over the whole prompt with no workspace at all,
        // so nothing is chunked and nothing can run out of capacity.
        var refCaches = BuildCaches();
        using var wholeInput = Tensor<int>.From(promptIds, 1, promptLen);
        using var expected = model.ForwardLastLogits(wholeInput, refCaches, 0, null);

        // Actual: chunked, against a workspace sized the way the generators size it.
        var caches = BuildCaches();
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, Cfg.MaxSeqLen));
        using var actual = Prefill.ForwardLastLogitsChunked(model, caches, promptIds, workspace);

        Assert.Equal(expected.Shape[^1], actual.Shape[^1]);
        for (int i = 0; i < Cfg.VocabSize; i++)
            Assert.True(MathF.Abs(expected.Data[i] - actual.Data[i]) < 1e-3f,
                $"logit[{i}] chunked={actual.Data[i]} one-shot={expected.Data[i]}");

        // The KV cache must end up holding the whole prompt, not just the last chunk.
        Assert.Equal(promptLen, caches[0].Length);

        foreach (var c in refCaches) c.Dispose();
        foreach (var c in caches) c.Dispose();
    }

    [Fact]
    public void ForwardLastLogitsChunked_HonoursCurrentCacheLength_ForASecondTurn()
    {
        int[] first = BuildPrompt(Prefill.MaxChunkLength * 2 + 10);
        int[] second = BuildPrompt(Prefill.MaxChunkLength * 2 + 5);

        using var model = BuildModel();
        var caches = BuildCaches();
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, Cfg.MaxSeqLen));

        using (var _ = Prefill.ForwardLastLogitsChunked(model, caches, first, workspace)) { }
        Assert.Equal(first.Length, caches[0].Length);

        // Each chunk starts at the current cache length, so a second turn
        // continues where the first left off instead of restarting at 0.
        using (var _ = Prefill.ForwardLastLogitsChunked(model, caches, second, workspace)) { }
        Assert.Equal(first.Length + second.Length, caches[0].Length);

        foreach (var c in caches) c.Dispose();
    }

    [Fact]
    public void ForwardFill_LeavesTheSameCacheAsForwardLastLogits()
    {
        int[] first = BuildPrompt(Prefill.MaxChunkLength);
        int[] second = BuildPrompt(Prefill.MaxChunkLength + 9);

        using var model = BuildModel();
        var refCaches = BuildCaches();
        var caches = BuildCaches();
        using var firstInput = Tensor<int>.From(first, 1, first.Length);
        using var secondInput = Tensor<int>.From(second, 1, second.Length);

        using (var _ = model.ForwardLastLogits(firstInput, refCaches, 0, null)) { }
        using var expected = model.ForwardLastLogits(secondInput, refCaches, first.Length, null);

        model.ForwardFill(firstInput, caches, 0, null);
        Assert.Equal(first.Length, caches[0].Length);
        using var actual = model.ForwardLastLogits(secondInput, caches, first.Length, null);

        // Bit for bit: the fill runs the same blocks, it only skips the head.
        Assert.Equal(expected.Data.ToArray(), actual.Data.ToArray());

        foreach (var c in refCaches) c.Dispose();
        foreach (var c in caches) c.Dispose();
    }

    [Fact]
    public void ForwardLastLogitsChunked_ProjectsTheHeadForTheLastChunkOnly()
    {
        int[] promptIds = BuildPrompt(Prefill.MaxChunkLength * 2 + 37); // three chunks

        using var model = BuildModel();
        var head = CountingLogitOps.Install(model);
        var caches = BuildCaches();
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, Cfg.MaxSeqLen));

        using (var _ = Prefill.ForwardLastLogitsChunked(model, caches, promptIds, workspace)) { }

        Assert.Equal(1, head.Calls);
        Assert.Equal(promptIds.Length, caches[0].Length);

        foreach (var c in caches) c.Dispose();
    }

    /// <summary>Counts vocabulary projections by standing in for the model's head and
    /// delegating every call to the original.</summary>
    [Fact]
    public void ForwardLastLogitsChunked_SlidingWindow_ProcessesEntirePrompt()
    {
        // Capacity 100 and a 200-token prompt (the issue's repro): after any trim
        // the room left is 50, smaller than MaxChunkLength (64), so the loop's
        // advance must use the tokens actually processed. Using the full chunk
        // length once dropped tokens 114-127 and 178-191 and the prefill returned
        // logits for token 95 alongside a 0.75 progress high-water mark.
        int cacheCapacity = 100;
        int promptLen = 200;
        int[] promptIds = BuildPrompt(promptLen);

        using var model = BuildModel();

        var caches = new IKVCache[Cfg.NumLayers];
        for (int i = 0; i < Cfg.NumLayers; i++)
            caches[i] = new KVCache(1, Cfg.NumKvHeads, cacheCapacity, Cfg.HeadDim);

        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, cacheCapacity));

        var progressReports = new List<double>();
        using var logits = Prefill.ForwardLastLogitsChunked(
            model, caches, promptIds, workspace, p => progressReports.Add(p));

        // Every token must have been processed — the last progress report is 1.0.
        Assert.Equal(1.0, progressReports[^1]);

        // The window holds only its capacity-worth of tokens.
        Assert.True(caches[0].Length <= cacheCapacity);

        // Logits are for the last prompt token.
        Assert.Equal(Cfg.VocabSize, logits.Shape[^1]);

        foreach (var c in caches) c.Dispose();
    }

    [Theory]
    [InlineData(70)] // longer than MaxChunkLength: the chunk loop, which trims
    [InlineData(20)] // a single chunk: used to skip the capacity check and throw "KVCache overflow"
    public void ForwardLastLogitsChunked_ContinuedTurnPastCapacity_TrimsWhateverItsLength(int secondTurn)
    {
        int cacheCapacity = 100;
        using var model = BuildModel();

        var caches = new IKVCache[Cfg.NumLayers];
        for (int i = 0; i < Cfg.NumLayers; i++)
            caches[i] = new KVCache(1, Cfg.NumKvHeads, cacheCapacity, Cfg.HeadDim);

        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, cacheCapacity));

        using (var _ = Prefill.ForwardLastLogitsChunked(model, caches, BuildPrompt(90), workspace)) { }
        Assert.Equal(90, caches[0].Length);

        var progressReports = new List<double>();
        using var logits = Prefill.ForwardLastLogitsChunked(
            model, caches, BuildPrompt(secondTurn), workspace, p => progressReports.Add(p));

        Assert.Equal(1.0, progressReports[^1]);
        Assert.True(caches[0].Length <= cacheCapacity);
        Assert.Equal(Cfg.VocabSize, logits.Shape[^1]);

        foreach (var c in caches) c.Dispose();
    }

    private sealed unsafe class CountingLogitOps(LogitOps inner, Tensor<float> weight, byte[]? raw)
        : LogitOps(weight, raw)
    {
        public int Calls;

        public static CountingLogitOps Install(Transformer model)
        {
            const BindingFlags Instance = BindingFlags.NonPublic | BindingFlags.Instance;
            var field = typeof(Transformer).GetField("_logitOps", Instance)!;
            var original = (LogitOps)field.GetValue(model)!;
            var spy = new CountingLogitOps(original,
                (Tensor<float>)typeof(LogitOps).GetField("ProjectionWeight", Instance)!.GetValue(original)!,
                (byte[]?)typeof(LogitOps).GetField("RawWeight", Instance)!.GetValue(original));
            field.SetValue(model, spy);
            return spy;
        }

        public override void ProjectFn(float* input, byte* rawWeights, float* output, int M, int K, int N)
        {
            Calls++;
            inner.ProjectFn(input, rawWeights, output, M, K, N);
        }
    }
}
