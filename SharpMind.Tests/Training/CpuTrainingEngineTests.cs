using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Training;

/// <summary>
/// <see cref="CpuTrainingEngine"/> is the reference implementation of
/// <see cref="ITrainingEngine"/>: it must produce exactly the loss and gradients
/// the raw BackpropEngine + loss path produces, because it *is* that path — so
/// the bit-exact assertion below is deliberate, verifying a verbatim code
/// extraction, not a tolerance-worthy numeric result. It is serialized (this
/// collection never runs in parallel with any other) because the underlying
/// <c>BackpropEngine</c> exhibits ULP-scale non-determinism when many test
/// classes share the process — a separate, pre-existing issue unrelated to
/// this branch and not what this test measures.
/// </summary>
[Collection("Non-Parallel")]
public sealed class CpuTrainingEngineTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static ModelConfig Cfg => new()
    {
        VocabSize = 16, HiddenDim = 8, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 16, MaxSeqLen = 16,
    };

    private async Task<TrainingBatch> FirstBatchAsync()
    {
        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 8).Select(i => $"the quick fox jumps over the lazy dog number {i}")));
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(pipeline, s => TestTokens.Encode(s, 16), new PackingBatcher(batchSize: 2, maxSeqLen: 16));
        // maxBatches: 1 lets the loader complete normally instead of the caller
        // abandoning the enumeration early — keeps this parity test independent of
        // DataLoader's abandon-early disposal behaviour (see eca6a13).
        await foreach (var batch in loader.LoadAsync(maxBatches: 1))
            return batch;
        throw new InvalidOperationException("loader produced no batch");
    }

    private static (Transformer Model, List<Parameter> Parameters) Model(SharpMindConfig sharpConfig)
    {
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 4321);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        return (model, model.Parameters().ToList());
    }

    [Fact]
    public async Task ForwardBackward_MatchesRawBackpropEngineAndLoss_BitExact()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var mapping = GradientMappingFactory.Create(sharpConfig);
        using var batch = await FirstBatchAsync();

        // Reference: the code TrainLoop.ForwardBackward ran before the seam existed.
        var (refModel, refParams) = Model(sharpConfig);
        using (refModel)
        {
            var loss = new CrossEntropyLoss(labelSmoothing: 0.1f);
            using var engine = new BackpropEngine(refModel, mapping, refParams, sharpConfig);
            int b = batch.TokenIds.Shape.Rows, s = batch.TokenIds.Shape.Cols, v = refModel.Config.VocabSize;
            using var ctx = new ForwardContext();
            using var flatLabels = batch.Labels.Reshape(b * s);
            var logits = engine.ForwardAndRecord(ctx, batch.TokenIds);
            using var logitsFlat = logits.Reshape(b * s, v);
            float refLoss = loss.Compute(logitsFlat, flatLabels);
            using var dLogits = loss.Backward(logitsFlat, flatLabels);
            using var flatIds = batch.TokenIds.Reshape(b * s);
            engine.Backward(ctx, dLogits, flatIds);

            // Under test: same weights (same seed), same batch, through the seam.
            var (cpuModel, cpuParams) = Model(sharpConfig);
            using (cpuModel)
            using (var cpu = new CpuTrainingEngine(cpuModel, mapping, cpuParams, sharpConfig, new CrossEntropyLoss(labelSmoothing: 0.1f)))
            {
                float cpuLoss = cpu.ForwardBackward(batch);

                Assert.Equal(refLoss, cpuLoss);
                Assert.Equal(refParams.Count, cpuParams.Count);
                for (int i = 0; i < refParams.Count; i++)
                {
                    Assert.Equal(refParams[i].Name, cpuParams[i].Name);
                    Assert.True(refParams[i].Grad.Data.SequenceEqual(cpuParams[i].Grad.Data),
                        $"gradient of '{refParams[i].Name}' differs");
                }
            }
        }
    }
}
