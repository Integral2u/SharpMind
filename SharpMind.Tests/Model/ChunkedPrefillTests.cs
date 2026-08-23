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
}
