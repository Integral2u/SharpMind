using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// #58 race: EnsureLayerLoadedSync decided whether to wait for an in-flight async
/// preload by checking whether the block was completely empty
/// (Wq == null &amp;&amp; RawWq == null). LoadLayerWeights fills the block one tensor at a
/// time, so once the attention bytes landed the check was false, the wait was
/// skipped, and a half-loaded block was pushed into the TransformerBlock and
/// marked as done — later forwards reused it without ever seeing the FFN. The
/// method must wait for the layer's preload task before treating it as loaded.
/// </summary>
public sealed class StreamingLayerLoadRaceTests
{
    [Fact]
    public void EnsureLayerLoadedSync_WaitsForInFlightPreloadBeforePushingLayer()
    {
        var config = ModelConfig.Learnable;
        var embedding = new Tensor<float>(config.VocabSize, config.HiddenDim);
        var finalNorm = new Tensor<float>(config.HiddenDim);
        var blocks = new[] { new TransformerWeights.BlockWeights { LayerIndex = 0 } };

        var loader = new GatedLayerLoader();
        using var weights = new TransformerWeightsStreaming(
            config, embedding, null, finalNorm, null, blocks, loader);

        weights.PreloadLayerAsync(0);
        Assert.True(loader.Started.Wait(TimeSpan.FromSeconds(10)), "the preload never started");

        var forward = new Thread(() =>
        {
            weights.EnsureLayerLoadedSync(0);
            loader.ForwardReleased.Set();
        })
        { IsBackground = true };
        forward.Start();

        Assert.True(loader.Finished.Wait(TimeSpan.FromSeconds(10)), "the preload never finished");
        Assert.True(forward.Join(TimeSpan.FromSeconds(10)), "EnsureLayerLoadedSync did not return");
        Assert.False(loader.ReturnedWhileBlocked,
            "EnsureLayerLoadedSync returned while the preload was still filling the layer");
        Assert.NotNull(blocks[0].RawWf2);
    }

    /// <summary>
    /// Writes the attention bytes, parks, and records whether the forward thread
    /// returned from EnsureLayerLoadedSync before the rest of the block was
    /// written — exactly the buggy early push the fix must prevent. A correct
    /// caller blocks on the preload task here, so the wait times out.
    /// </summary>
    private sealed class GatedLayerLoader : IModelLoader
    {
        public readonly ManualResetEventSlim Started = new(false);
        public readonly ManualResetEventSlim Finished = new(false);
        public readonly ManualResetEventSlim ForwardReleased = new(false);
        public volatile bool ReturnedWhileBlocked;

        public void LoadGlobalTensors(TransformerWeights weights) { }

        public void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null) { }

        public void LoadLayerWeights(int layerIndex, TransformerWeights weights)
        {
            var block = weights.Blocks[layerIndex];
            block.RawWq = [0, 0, 0, 0];
            Started.Set();

            ReturnedWhileBlocked = ForwardReleased.Wait(TimeSpan.FromSeconds(2));

            block.RawWf2 = [0, 0, 0, 0];
            Finished.Set();
        }
    }
}
