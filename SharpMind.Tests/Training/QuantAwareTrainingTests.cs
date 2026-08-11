using System.IO;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace SharpMind.Tests.Training;

/// <summary>
/// Verifies quantization-aware training:
/// <list type="bullet">
///   <item>QAT forwards on <see cref="TrainingLinearLayer"/> use quantized weights
///   (parity with an explicit quantize→dequantize→float-matmul reference);</item>
///   <item>gradients flow straight through to the F32 master weight
///   (<see cref="TrainingLinearLayer.Backward"/> never reads quantized weights);</item>
///   <item>block-format targets validate multiples of 32 while F16/F32/null stay
///   safe on any shape;</item>
///   <item>a full <see cref="TrainLoop"/> run with Q8_0/Q4_0/F16/K-quant QAT on
///   aligned dims descends loss.</item>
/// </list>
/// </summary>
public sealed class QuantAwareTrainingTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    // In is a multiple of both 32 (block formats) and 128 (K-quant column
    // alignment), so every target in the theory below is exercisable on this shape.
    private const int In = 128;
    private const int Out = 32;
    private const int Batch = 3;

    private static SharpMindConfig Scalar() => SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

    private static TrainingLinearLayer MakeLayer(string name, int inFeatures, int outFeatures)
        => new TrainingLinearLayer(name, inFeatures, outFeatures, bias: true, weight: null, biasTensor: null);

    private static void FillRandom(TrainingLinearLayer layer, Random rng)
    {
        var w = layer.Weight.Data;
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        var b = layer.Bias!.Data;
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 0.5 - 0.25);
    }

    /// <summary>Dequantizes raw bytes back to floats in flattened row-major order.</summary>
    private static float[] Dequantize(byte[] raw, QuantDType dtype, int count)
    {
        var ops = QuantizationFactory.Create(HardwareTier.Scalar);
        var result = new float[count];
        using var ms = new MemoryStream(raw, writable: false);
        using var reader = new BinaryReader(ms);
        ops.ReadFor(dtype, reader, result, count);
        return result;
    }

    /// <summary>
    /// Reference: quantize the transposed weight (layout [Out, In]) and dequantize
    /// back to floats, then run an explicit float matmul with the same input.
    /// QAT forwards should match this per-column-major layout.
    /// </summary>
    private static float[] ReferenceForward(TrainingLinearLayer layer, float[] input, QuantDType target)
    {
        var weightBT = layer.Weight.Transpose();
        try
        {
            var raw = TensorQuantizer.Quantize(weightBT.Data, [weightBT.Shape.Rows, weightBT.Shape.Cols], target);
            var deq = Dequantize(raw, target, weightBT.ElementCount);
            var bias = layer.Bias!.Data;
            var output = new float[Batch * Out];
            for (int m = 0; m < Batch; m++)
            {
                for (int o = 0; o < Out; o++)
                {
                    float s = 0;
                    for (int i = 0; i < In; i++)
                        s += input[m * In + i] * deq[o * In + i];
                    output[m * Out + o] = s + bias[o];
                }
            }
            return output;
        }
        finally
        {
            weightBT.Dispose();
        }
    }

    private static float[] RunForward(TrainingLinearLayer layer, float[] input)
    {
        using var t = Tensor<float>.From(input, Batch, In);
        using var output = layer.Forward(t);
        return [.. output.Data];
    }

    [Theory]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.Q4_0)]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.Q8_K)]
    [InlineData(QuantDType.Q6_K)]
    [InlineData(QuantDType.Q4_K)]
    public void Forward_MatchesQuantizedDequantizedReference(QuantDType target)
    {
        using var layer = MakeLayer("forward", In, Out);
        FillRandom(layer, new Random(11));
        layer.EnableQuantAwareTraining(target);

        var input = new float[Batch * In];
        var rng = new Random(77);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = ReferenceForward(layer, input, target);
        var actual = RunForward(layer, input);

        double maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual[i] - expected[i]);
            double scale = Math.Max(1e-3, Math.Abs(expected[i]));
            maxRel = Math.Max(maxRel, err / scale);
        }
        Assert.True(maxRel < 1e-3,
            $"QAT [{target}] forward deviates from dequantized reference: MaxRel={maxRel:E3}");
    }

    /// <summary>Straight-through: backward never reads quantized weights, so with
    /// identical input and grad-output, weight gradients are identical whether QAT
    /// is enabled or not.</summary>
    [Theory]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.Q4_K)]
    public void Backward_GradientIgnoresQuantization(QuantDType target)
    {
        var rng = new Random(91);
        using var qat = MakeLayer("gt", In, Out);
        using var flat = MakeLayer("bt", In, Out);
        FillRandom(qat, rng);
        FillRandom(flat, rng);
        qat.EnableQuantAwareTraining(target);
        flat.EnableQuantAwareTraining(null);

        var input = new float[Batch * In];
        var gradOut = new float[Batch * Out];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] weightGradQat = WeightGrad(qat, input, gradOut);
        float[] weightGradFlat = WeightGrad(flat, input, gradOut);

        for (int i = 0; i < weightGradQat.Length; i++)
            Assert.Equal(weightGradFlat[i], weightGradQat[i], precision: 3);
    }

    private static float[] WeightGrad(TrainingLinearLayer layer, float[] input, float[] gradOut)
    {
        using var t = Tensor<float>.From(input, Batch, In);
        var (output, state) = layer.ForwardWithState(t);
        using var g = Tensor<float>.From(gradOut, Batch, Out);
        using var _ = layer.Backward(g, state);
        output.Dispose();
        return [.. state.WeightGrad.Data];
    }

    [Theory]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.Q4_0)]
    public void EnableQuantAwareTraining_BlockTargetOnNon32Dims_Throws(QuantDType target)
    {
        using var layer = MakeLayer("odd", 7, Out);
        Assert.Throws<InvalidOperationException>(() => layer.EnableQuantAwareTraining(target));
    }

    [Theory]
    [InlineData(QuantDType.Q2_K)]
    [InlineData(QuantDType.Q3_K)]
    [InlineData(QuantDType.Q4_K)]
    [InlineData(QuantDType.Q5_K)]
    [InlineData(QuantDType.Q6_K)]
    [InlineData(QuantDType.Q8_K)]
    public void EnableQuantAwareTraining_KQuantOnNon256FlattenedLength_Throws(QuantDType target)
    {
        // 13 x 7 = 91 elements — not a multiple of 256.
        using var layer = MakeLayer("odd", 13, 7);
        Assert.Throws<InvalidOperationException>(() => layer.EnableQuantAwareTraining(target));
    }

    [Theory]
    [InlineData(QuantDType.Q2_K)]
    [InlineData(QuantDType.Q8_K)]
    public void EnableQuantAwareTraining_KQuantOnAlignedDims_Ok(QuantDType target)
    {
        // In x Out = 128 x 32 = 4096 elements (multiple of 256) and In is a
        // multiple of 128, so the K-quant column alignment holds.
        using var layer = MakeLayer("aligned", In, Out);
        layer.EnableQuantAwareTraining(target);
        Assert.Equal(target, layer.QuantAwareTarget);
    }

    [Theory]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.F32)]
    [InlineData(null)]
    public void EnableQuantAwareTraining_PassthroughTargetsWorkOnAnyDims(QuantDType? target = null)
    {
        using var layer = MakeLayer("any", 36, 12);
        layer.EnableQuantAwareTraining(target);
        using var _ = layer.Forward(Tensor<float>.From(new float[2 * 36], 2, 36));
    }

    private static ModelConfig QatModelConfig => new()
    {
        VocabSize = 128,  // % 32 == 0; tied head [128, 128] = 16384 % 256 == 0
        HiddenDim = 128,  // % 32 == 0 and % 128 == 0 (K-quant column alignment)
        NumLayers = 1,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 128,     // % 32 == 0 and % 128 == 0
        MaxSeqLen = 16,
    };

    [Theory]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.Q4_0)]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.Q4_K)]
    [InlineData(QuantDType.Q6_K)]
    public void TrainLoop_WithQat_LossDescends(QuantDType target)
    {
        var sharpConfig = Scalar();
        var weights = ModelFactory.CreateForTraining(QatModelConfig, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 2024);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        // Corpus sized so the loader's finite source yields exactly the batches
        // `TotalSteps` consumes: the background producer finishes by the time the
        // loop breaks, releasing the file handle before TempDirectory.Dispose runs.
        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 24)
                .Select(i => $"the quick brown fox jumps over {i}")));

        var pipeline = PipelineNode.From(new TextFileSource(path))
            .Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(
            pipeline,
            s => s.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                  .Select(w => Math.Abs(w.GetHashCode()) % QatModelConfig.VocabSize).ToArray(),
            new PackingBatcher(batchSize: 2, maxSeqLen: 8),
            prefetchBuffer: 1);

        var parameters = model.Parameters().ToList();
        var ops = TrainingOpsFactory.Create(sharpConfig);
        using var optimizer = new AdamW(parameters, ops, lr: 0.02f, weightDecay: 0f);
        var losses = new List<float>();
        var loop = new TrainLoop(
            model,
            parameters,
            loader,
            optimizer,
            new ConstantScheduler(0.02f),
            ops,
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = 12,
                LogInterval = 3,
                GradClipNorm = 0f,
                QuantAwareTraining = target,
            });

        loop.RunAsync(onStep: r => losses.Add(r.Loss)).GetAwaiter().GetResult();

        Assert.NotEmpty(losses);
        Assert.True(float.IsFinite(losses[^1]), "QAT produced non-finite loss.");
        Assert.True(losses[^1] < losses[0],
            $"QAT [{target}] loss did not descend: {losses[0]:F4} → {losses[^1]:F4}");
    }
}