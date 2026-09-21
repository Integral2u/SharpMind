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
/// An <see cref="Int8KVCache"/> plugged into a model runs the int8 attention kernel through
/// AttentionLayer's quantized-cache dispatch. Over a chunked prefill and a few decode steps its
/// logits stay close to the float cache's and pick the same token.
/// </summary>
public sealed class Int8KVCacheAttentionTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 128,
        HiddenDim = 160, // headDim 40: two whole key blocks and a short one
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 128,
        MaxSeqLen = 256,
    };

    [Theory]
    [InlineData(HardwareTier.Scalar)]
    [InlineData(HardwareTier.Auto)]
    public void PrefillAndDecode_WithTheInt8Cache_StayCloseToTheFloatCache(HardwareTier hardware)
    {
        var sharp = SharpMindConfig.Gpt with { Hardware = hardware };
        var weights = ModelFactory.CreateForTraining(Cfg, sharp);
        WeightInitializer.InitializeRandomly(weights, 4242);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharp);
        using var workspace = MemoryHelpers.CreateWorkspace(
            Workspace.CalculateRequiredSize(Cfg.HiddenDim, Cfg.FfnDim, Cfg.VocabSize, Cfg.NumLayers, Cfg.MaxSeqLen));
        IKVCache[] floatCaches = [.. Enumerable.Range(0, Cfg.NumLayers).Select(_ => (IKVCache)new KVCache(1, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDim))];
        IKVCache[] int8Caches = [.. Enumerable.Range(0, Cfg.NumLayers).Select(_ => (IKVCache)new Int8KVCache(1, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDim))];
        var rng = new Random(11);
        int[] prompt = [.. Enumerable.Range(0, 100).Select(_ => rng.Next(Cfg.VocabSize))];

        try
        {
            int token = 0;
            for (int step = 0; step < 4; step++)
            {
                int[] input = step == 0 ? prompt : [token];
                float[] expected;
                using (var logits = Prefill.ForwardLastLogitsChunked(model, floatCaches, input, workspace)) expected = logits.Data.ToArray();
                float[] actual;
                using (var logits = Prefill.ForwardLastLogitsChunked(model, int8Caches, input, workspace)) actual = logits.Data.ToArray();

                double diff = 0, norm = 0;
                for (int i = 0; i < expected.Length; i++) { diff += (expected[i] - actual[i]) * (expected[i] - actual[i]); norm += expected[i] * expected[i]; }
                double relError = Math.Sqrt(diff / norm);
                token = ArgMax(expected);
                Assert.True(ArgMax(actual) == token && relError <= 0.05,
                    $"{hardware} step {step}: argmax float={token} int8={ArgMax(actual)}, relative logit error {relError:F4}");
                Assert.Equal(floatCaches[0].Length, int8Caches[0].Length);
            }
        }
        finally
        {
            foreach (var c in floatCaches) c.Dispose();
            foreach (var c in int8Caches) c.Dispose();
        }
    }

    private static int ArgMax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }
}
