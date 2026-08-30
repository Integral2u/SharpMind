using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.GPU;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Training;
using SharpMind.Training.Autograd;
using SharpMind.Training.LoRA;
using SharpMind.Training.Loss;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class GpuBackpropEngineTests
{
    /// <summary>
    /// Architecture "qwen2" is what makes <c>AttentionLayer.UsesNeoxRope</c> true, so the fixture
    /// exercises the NeoX (d, d + ropeDim/2) pairing rather than the adjacent one.
    /// <c>ModelFactory.CreateForTraining</c> always allocates Q/K/V/O and FFN biases, so the
    /// bias path is exercised too — the fixture fills them, because
    /// <c>WeightInitializer.InitializeRandomly</c> leaves biases (and norm weights) untouched
    /// and a zero bias added to a zero-mean activation proves nothing.
    /// </summary>
    internal static readonly ModelConfig Cfg = new()
    {
        VocabSize = 256, HiddenDim = 64, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, FfnDim = 128, MaxSeqLen = 64,
        Architecture = "qwen2",
    };
    internal const int B = 2, S = 8, Rank = 4;

    /// <summary>
    /// Llama (SiLU + SwiGLU) or the Gemma flavour (GELU + GeGLU). The second is the one that
    /// drives <c>_gelu</c> and <c>_gemmaScale</c>; both are false under Llama, so without it the
    /// engine's gate-kernel choice and its √H embedding scale are never executed.
    /// </summary>
    internal static SharpMindConfig Preset(bool geglu) => SharpMindConfig.Llama with
    {
        Hardware = HardwareTier.Scalar,
        Activation = geglu ? ActivationKind.GELU : ActivationKind.SiLU,
        Gate = geglu ? GateKind.GeGLU : GateKind.SwiGLU,
    };

    internal static (Transformer Model, LoRAModel Lora, List<Parameter> Params, SharpMindConfig Config) Fixture(bool geglu = false, int seed = 9001, int rank = Rank)
    {
        var sc = Preset(geglu);
        var model = Build(Cfg, sc, seed);
        var lora = new LoRAModel(model, new LoRAConfig { Rank = rank, TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"] }, seed: 1);
        var ps = lora.LoRAParameters().ToList();
        // B is zero after EnableLoRA — perturb so the adapter path carries signal.
        var rng = new Random(11);
        foreach (var p in ps.Where(p => p.Name.EndsWith("lora_B"))) for (int i = 0; i < p.Data.ElementCount; i++) p.Data.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.05f;
        return (model, lora, ps, sc);
    }

    private static Transformer Build(ModelConfig mc, SharpMindConfig sc, int seed = 9001)
    {
        var weights = ModelFactory.CreateForTraining(mc, sc);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sc);
        Perturb(model, mc, new Random(7));
        return model;
    }

    /// <summary>Gives the biases and the RMSNorm weights values, so neither is a no-op multiply by 1 / add of 0.</summary>
    private static void Perturb(Transformer model, ModelConfig mc, Random rng)
    {
        void Fill(Tensor<float>? t, float centre) { if (t is null) return; for (int i = 0; i < t.ElementCount; i++) t.Data[i] = centre + (float)(rng.NextDouble() - 0.5) * 0.1f; }
        for (int l = 0; l < mc.NumLayers; l++)
        {
            var b = model.GetBlock(l)!;
            foreach (var lin in new LinearLayer[] { b.Attention.Wq, b.Attention.Wk, b.Attention.Wv, b.Attention.Wo, b.Ffn.WGated!, b.Ffn.WDown! })
                Fill(lin.Bias, 0f);
            Fill(b.Norm1.NormWeight, 1f);
            Fill(b.Norm2.NormWeight, 1f);
        }
        Fill(model.FinalNorm.NormWeight, 1f);
    }

    internal static (Tensor<int> ids, Tensor<int> labels) Batch(int seed = 3)
    {
        var r = new Random(seed); var ids = new int[B * S]; var labels = new int[B * S];
        for (int i = 0; i < ids.Length; i++) { ids[i] = r.Next(Cfg.VocabSize); labels[i] = r.Next(Cfg.VocabSize); }
        labels[5] = -100;
        return (Tensor<int>.From(ids, B, S), Tensor<int>.From(labels, B, S));
    }

    /// <summary>
    /// One forward's rents at this shape: the entry x, then per block
    /// 6H + 2 rInv + 2·qDim + 2·kvDim + 3F + one hs per adapted linear, plus probs;
    /// then the head's normed, rInv and logits.
    /// </summary>
    private static long ForwardFloats(int rank)
    {
        long m = (long)B * S, H = Cfg.HiddenDim, q = (long)Cfg.NumHeads * Cfg.HeadDim, kv = (long)Cfg.NumKvHeads * Cfg.HeadDim;
        long probs = (long)B * Cfg.NumHeads * S * S;
        return m * H
             + Cfg.NumLayers * (m * (6 * H + 2 + 2 * q + 2 * kv + 3 * Cfg.FfnDim + 6 * rank) + probs)
             + m * (H + 1 + Cfg.VocabSize);
    }

    /// <summary>
    /// A whole ForwardBackward's rents: the forward above (the arena is not reset between the
    /// two halves), plus per block dN2, dN2in, dN1, dN1in, dAct + dFused, dAttnOut + dQ, dK + dV,
    /// one dH per adapted linear and the dProbs scratch; plus the head's rowLoss, dN and dX.
    /// </summary>
    private static long StepFloats(int rank)
    {
        long m = (long)B * S, H = Cfg.HiddenDim, q = (long)Cfg.NumHeads * Cfg.HeadDim, kv = (long)Cfg.NumKvHeads * Cfg.HeadDim;
        long probs = (long)B * Cfg.NumHeads * S * S;
        return ForwardFloats(rank)
             + Cfg.NumLayers * (m * (4 * H + 3 * Cfg.FfnDim + 2 * q + 2 * kv + 6 * rank) + probs)
             + m * (2 * H + 1);
    }

    /// <summary>
    /// Slack over the exact rent total for <see cref="DeviceArena"/>'s 32-float alignment of
    /// every rent. At this shape only the 16-float rents are not already a multiple of 32 —
    /// five rInv in a forward (80 floats of padding), plus rowLoss in a full step (96) — so 256
    /// never trips, and stays well under the 1024 floats of a stray m·H rent, which must trip.
    /// </summary>
    private const int AlignSlack = 256;

    /// <summary>
    /// The oracle is the real CPU engine. Tolerance 1e-4: the CPU attention softmax uses the
    /// degree-6 <c>ActivationKernels.FastExp</c> when AVX2 is available while the GPU kernel
    /// uses exact <c>Math.Exp</c>, so the two disagree by ~1e-6 per probability and that error
    /// is carried through two blocks and the LM head.
    /// </summary>
    [Theory]
    [InlineData(false)]   // Llama: SiLU + SwiGLU, no embedding scale
    [InlineData(true)]    // Gemma flavour: GELU + GeGLU — the only case that runs _gelu and _gemmaScale
    public void Forward_LogitsMatchCpuEngine(bool geglu)
    {
        var (model, lora, ps, sc) = Fixture(geglu);
        using var _ = model; using var __ = lora;
        var (ids, labels) = Batch();
        using var ___ = ids; using var ____ = labels;

        using var cpu = new BackpropEngine(model, GradientMappingFactory.Create(sc), ps, sc);
        using var ctx = new ForwardContext();
        var want = cpu.ForwardAndRecord(ctx, ids).Data.ToArray();

        using var gpu = new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, B, S);
        var got = gpu.ForwardLogitsForTest(ids);
        GpuTestDevice.AssertClose(want, got, 1e-4, "logits");
        Assert.InRange(gpu.ArenaUsed, ForwardFloats(Rank), ForwardFloats(Rank) + AlignSlack);
    }

    /// <summary>
    /// The whole step against the real CPU engine: the same scalar loss and, for every LoRA
    /// A and B, the same gradient. Tolerance 1e-4, the same as the forward's — the CPU
    /// attention softmax uses the degree-6 <c>ActivationKernels.FastExp</c> under AVX2 while
    /// the GPU uses exact <c>Math.Exp</c>, and that ~1e-6 per-probability disagreement is what
    /// the backward amplifies. Label smoothing is on so the smoothed branch of both the loss
    /// and its gradient is exercised, and <c>Batch</c> puts one ignored label in the batch.
    /// Both gate flavours, because <c>GateBwd(…, gelu: true)</c> is reachable through the engine
    /// only under GeGLU — the kernel's own tests cover both values, but not the composition.
    /// </summary>
    [Theory]
    [InlineData(false)]   // Llama: SiLU + SwiGLU
    [InlineData(true)]    // Gemma flavour: GELU + GeGLU — the only case that runs GateBwd's gelu branch
    public void ForwardBackward_LossAndLoRAGradsMatchCpuEngine(bool geglu)
    {
        var (model, lora, ps, sc) = Fixture(geglu);
        using var _ = model; using var __ = lora;
        var (ids, labels) = Batch();
        using var batch = new TrainingBatch(ids, labels, Tensor<float>.Zeros(B, S), B * S);

        using var cpu = new CpuTrainingEngine(model, GradientMappingFactory.Create(sc), ps, sc, new CrossEntropyLoss(labelSmoothing: 0.1f));
        float wantLoss = cpu.ForwardBackward(batch);
        var wantGrads = ps.Select(p => p.Grad.Data.ToArray()).ToList();
        foreach (var p in ps) p.ZeroGrad();

        using var gpu = new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, B, S, labelSmoothing: 0.1f);
        float gotLoss = gpu.ForwardBackward(batch);

        Assert.Equal(wantLoss, gotLoss, 4);
        for (int i = 0; i < ps.Count; i++) GpuTestDevice.AssertClose(wantGrads[i], ps[i].Grad.Data, 1e-4, ps[i].Name);
        Assert.InRange(gpu.ArenaUsed, StepFloats(Rank), StepFloats(Rank) + AlignSlack);
    }

    /// <summary>
    /// Same seed, same batches, AdamW on host in both runs: the curves must agree at every step.
    /// </summary>
    [Fact]
    public void TenSteps_LossCurvesCoincide()
    {
        float[] Run(bool gpu)
        {
            var (model, lora, ps, sc) = Fixture();
            using var _ = model; using var __ = lora;
            using ITrainingEngine engine = gpu
                ? new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, B, S)
                : new CpuTrainingEngine(model, GradientMappingFactory.Create(sc), ps, sc, new CrossEntropyLoss());
            using var opt = new SharpMind.Training.Optimizers.AdamW(ps, lr: 1e-2f, weightDecay: 0f);
            var losses = new float[10];
            for (int step = 0; step < 10; step++)
            {
                var (ids, labels) = Batch(seed: 100 + step);
                using var batch = new TrainingBatch(ids, labels, Tensor<float>.Zeros(B, S), B * S);
                opt.ZeroGrad();
                losses[step] = engine.ForwardBackward(batch);
                opt.Update();
            }
            return losses;
        }
        var cpu = Run(false); var gpu = Run(true);
        for (int i = 0; i < 10; i++) Assert.True(Math.Abs(cpu[i] - gpu[i]) < 1e-3 * Math.Max(1f, Math.Abs(cpu[i])), $"step {i}: cpu {cpu[i]} gpu {gpu[i]}");
        // No "loss decreased" assertion: Batch() draws fresh random labels every step, so the curve
        // carries no learning signal. The per-step CPU/GPU comparison above is the load-bearing one.
    }

    /// <summary>A batch shape other than the one the arena was sized for is rejected, not silently truncated.</summary>
    [Fact]
    public void ForwardLogits_RejectsAForeignBatchShape()
    {
        var (model, lora, ps, sc) = Fixture();
        using var _ = model; using var __ = lora;
        using var gpu = new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, B, S);
        using var shorter = Tensor<int>.From(new int[B * (S - 1)], B, S - 1);
        Assert.Throws<ArgumentException>(() => { gpu.ForwardLogitsForTest(shorter); });
        // Same token count, different split: RoPE would silently use the wrong positions.
        using var reshaped = Tensor<int>.From(new int[B * S], B * 2, S / 2);
        Assert.Throws<ArgumentException>(() => { gpu.ForwardLogitsForTest(reshaped); });
    }

    /// <summary>Everything M1 does not implement must name itself and point at the CPU engine.</summary>
    [Fact]
    public void ValidateSupported_RejectsWhatM1DoesNotImplement()
    {
        var sc = Preset(false);
        var (model, lora, ps, _) = Fixture();
        using var _m = model; using var _l = lora;
        GpuBackpropEngine.ValidateSupported(model, ps, sc);   // the fixture itself is supported

        var all = model.Parameters().ToList();
        Reject(() => GpuBackpropEngine.ValidateSupported(model, all, sc), "only LoRA");
        foreach (var p in all) p.Dispose();
        // One layer's A/B passed, the other five layers' adapters left without Parameters.
        Reject(() => GpuBackpropEngine.ValidateSupported(model, ps.Take(2).ToList(), sc), "not passed in");

        // Rank 1: GpuLinear rejects it, so validation must catch it before any device allocation.
        var (m1, lora1, ps1, _) = Fixture(rank: 1);
        using var _m1 = m1; using var _l1 = lora1;
        Reject(() => GpuBackpropEngine.ValidateSupported(m1, ps1, sc), "rank 1");

        RejectModel(SharpMindConfig.Llama with { Norm = NormKind.LayerNorm }, Cfg, "LayerNorm");
        RejectModel(SharpMindConfig.Llama with { Ffn = FfnKind.Dense, Gate = GateKind.None }, Cfg, "dense");
        RejectModel(SharpMindConfig.ForModel(Cfg.NumHeads, Cfg.NumKvHeads, "mixtral"), Cfg with { NumExperts = 4, TopKExperts = 2 }, "MoE");
        RejectModel(SharpMindConfig.Llama, Cfg with { PositionalEncoding = PositionalEncoding.ALiBi }, "ALiBi");
        RejectModel(SharpMindConfig.Llama, Cfg with { PositionalEncoding = PositionalEncoding.Learned }, "Learned");

        // Quantization-aware training, switched on after the model is built.
        using var qat = Build(Cfg, sc);
        qat.EnableQuantAwareTraining(QuantDType.Q8_0);
        Reject(() => GpuBackpropEngine.ValidateSupported(qat, [], sc), "quantization-aware");

        // Gemma-3 post norms: a block weight set that carries PostNorm1W builds a PostAttnNorm.
        var pw = ModelFactory.CreateForTraining(Cfg, sc);
        pw.Blocks[0].PostNorm1W = Tensor<float>.Ones(Cfg.HiddenDim);
        using var post = ModelFactory.CreateTrainingTransformer(pw, sc);
        Reject(() => GpuBackpropEngine.ValidateSupported(post, [], sc), "post-attention");

        // A model built for inference wires InferenceLinearLayers, which can carry a null Weight.
        var iw = ModelFactory.CreateForTraining(Cfg, sc);
        using var inference = ModelFactory.CreateTransformer(iw, sc, optimizeMemory: false);
        Reject(() => GpuBackpropEngine.ValidateSupported(inference, [], sc), "CreateTrainingTransformer");
    }

    private static void Reject(Action act, string needle)
    {
        var ex = Assert.Throws<NotSupportedException>(act);
        Assert.Contains(needle, ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("CPU engine", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static void RejectModel(SharpMindConfig sharp, ModelConfig mc, string needle)
    {
        var sc = sharp with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(mc, sc);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sc);
        Reject(() => GpuBackpropEngine.ValidateSupported(model, [], sc), needle);
    }
}
