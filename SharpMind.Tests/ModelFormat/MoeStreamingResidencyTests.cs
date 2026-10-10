using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
using SharpMind.Tokenization;
using SharpMind.Training;
using System.Text.Json.Nodes;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Covers streaming MoE expert residency (<c>DesiredResidentExperts</c>): the planner that
/// sizes the per-layer resident set from memory, the wiring that skips routed experts on a
/// residency layer load, the on-demand expert loads the router triggers, and the pin store
/// that carries pinned planes across unloads. The core assertion is that a residency-driven
/// streamed forward is bit-identical to the plain streamed forward (both run the standard
/// kernel over the same bytes — residency only changes which buffers are cached, not the math).
/// </summary>
public class MoeStreamingResidencyTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    // ── Planner ─────────────────────────────────────────────────────────────

    [Fact]
    public void Planner_RespectsBudgetAndClamps()
    {
        // No budget, or degenerate inputs, can't hold even one expert per layer.
        Assert.Equal(0, MoEResidencyPlanner.SuggestResidentExperts(0, 100, 3, 8));
        Assert.Equal(0, MoEResidencyPlanner.SuggestResidentExperts(1000, 100, 3, 0));
        Assert.Equal(0, MoEResidencyPlanner.SuggestResidentExperts(-10, 100, 3, 8));

        // 800 usable (20% headroom on 1000), one expert costs 100 x 3 layers = 300 → 2 fit.
        Assert.Equal(2, MoEResidencyPlanner.SuggestResidentExperts(1000, 100, 3, 8));

        // An attached budget caps at the model's expert count, never above.
        Assert.Equal(8, MoEResidencyPlanner.SuggestResidentExperts(1L << 40, 100, 3, 8));

        // Desired counts are clamped down, never up.
        Assert.Equal(0, MoEResidencyPlanner.ClampToBudget(5, 0));
        Assert.Equal(3, MoEResidencyPlanner.ClampToBudget(5, 3));
        Assert.Equal(5, MoEResidencyPlanner.ClampToBudget(5, 9));
    }

    // ── Wiring / engine behaviour ───────────────────────────────────────────

    [Fact]
    public void Residency_ZeroUntouched_DoesNotWireEngine()
    {
        using var fixture = NewMoEModel();
        string path = ExportGguf(fixture);
        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);

        using var weights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, fixture.SharpConfig);

        var sw = Assert.IsType<TransformerWeightsStreaming>(weights);
        Assert.Equal(0, sw.EffectiveResidentExperts);
        Assert.Equal(0, sw.PinnedExpertBytes);

        // 0 must be the historical streaming behaviour: every expert of a loaded layer exists.
        // Wait for the async layer-0 preload first, then check the loaded layer's dicts.
        for (int i = 0; i < weights.Blocks.Length; i++)
            sw.EnsureLayerLoadedSync(i);
        Assert.All(weights.Blocks, blk =>
            Assert.Equal(fixture.Config.NumExperts, blk.RawWgateExp?.Count ?? 0));
    }

    [Fact]
    public void Residency_EffectiveCount_IsClampedToExpertCount()
    {
        using var fixture = NewMoEModel();
        string path = ExportGguf(fixture);
        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);

        // Requesting far more experts than the model has must not wire more than exist; the
        // request stays recorded so the banner can show want/effective, the engine clamp wins.
        using var weights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming,
            desiredResidentExperts: fixture.Config.NumExperts + 64);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, fixture.SharpConfig);

        var sw = Assert.IsType<TransformerWeightsStreaming>(weights);
        Assert.Equal(fixture.Config.NumExperts + 64, sw.DesiredResidentExperts);
        Assert.True(sw.EffectiveResidentExperts >= 1 && sw.EffectiveResidentExperts <= fixture.Config.NumExperts,
            $"effective {sw.EffectiveResidentExperts} outside [1, {fixture.Config.NumExperts}]");
    }

    [Theory]
    [InlineData(4)] // pin every expert — no on-demand loads, pure pin-store round-trip
    [InlineData(2)] // pin half — the rest stream on demand per token
    [InlineData(1)] // pin one — maximal on-demand churn
    public void Streaming_Residency_IsBitIdenticalToPlainStreaming(int desired)
    {
        using var fixture = NewMoEModel();
        string path = ExportGguf(fixture);
        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);

        using var plainWeights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming);
        plainWeights.InitializeWeights();
        using var plain = ModelFactory.CreateTransformer(plainWeights, fixture.SharpConfig);

        using var resWeights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming,
            desiredResidentExperts: desired);
        resWeights.InitializeWeights();
        using var res = ModelFactory.CreateTransformer(resWeights, fixture.SharpConfig);

        var sw = Assert.IsType<TransformerWeightsStreaming>(resWeights);
        Assert.Equal(desired, sw.EffectiveResidentExperts);

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        // Two passes: the second exercises a FreeLayer/preload cycle so the pinned planes
        // must survive an unload and come back through the per-layer store.
        for (int pass = 0; pass < 2; pass++)
        {
            using var a = plain.Forward(tokens);
            using var b = res.Forward(tokens);
            Assert.Equal(a.ElementCount, b.ElementCount);
            for (int i = 0; i < a.ElementCount; i++)
                Assert.True(BitConverter.SingleToInt32Bits(a.Data[i]) == BitConverter.SingleToInt32Bits(b.Data[i]),
                    $"{nameof(desired)}={desired} pass {pass} logit {i}: plain={a.Data[i]} residency={b.Data[i]}");
        }

        if (desired > 0)
        {
            // Pin-store round-trip: force an unload of layer 0 and run once more. The pins
            // must be retained across the unload (not returned to the pool) and come back
            // through RehydratePins so the third pass stays bit-identical.
            sw.FreeLayer(0);
            Assert.True(sw.PinnedExpertBytes > 0, "pinned planes should be retained across the unload cycle");
            using var a = plain.Forward(tokens);
            using var b = res.Forward(tokens);
            Assert.Equal(a.ElementCount, b.ElementCount);
            for (int i = 0; i < a.ElementCount; i++)
                Assert.True(BitConverter.SingleToInt32Bits(a.Data[i]) == BitConverter.SingleToInt32Bits(b.Data[i]),
                    $"{nameof(desired)}={desired} pin-round-trip logit {i}: plain={a.Data[i]} residency={b.Data[i]}");
        }
    }

    [Fact]
    public void SmmLoader_Residency_IsBitIdenticalToPlainStreaming()
    {
        // SmmLoader serves training exports and GGUF→SMM conversions, so the residency
        // engine must wire just as it does for GgufLoader: SupportsExpertSlicing plus the
        // non-expert layer load and the on-demand expert loads. EffectiveResidentExperts
        // going non-zero proves the wiring; bit-identical logits prove the bytes agree.
        using var fixture = NewMoEModel();
        string smm = Path.Combine(_temp.Path, "moe-resid.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smm, new SmmWriteOptions { Source = "training" }, model: fixture.Model);
        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);

        using var plainWeights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, smm, LoadMode.Streaming);
        plainWeights.InitializeWeights();
        using var plain = ModelFactory.CreateTransformer(plainWeights, fixture.SharpConfig);

        using var resWeights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, smm, LoadMode.Streaming,
            desiredResidentExperts: 2);
        resWeights.InitializeWeights();
        using var res = ModelFactory.CreateTransformer(resWeights, fixture.SharpConfig);

        var sw = Assert.IsType<TransformerWeightsStreaming>(resWeights);
        Assert.Equal(2, sw.EffectiveResidentExperts);

        using var tokens = new Tensor<int>(1, 5);
        int[] ids = [3, 17, 42, 8, 63];
        ids.CopyTo(tokens.Data);

        for (int pass = 0; pass < 2; pass++)
        {
            using var a = plain.Forward(tokens);
            using var b = res.Forward(tokens);
            Assert.Equal(a.ElementCount, b.ElementCount);
            for (int i = 0; i < a.ElementCount; i++)
                Assert.True(BitConverter.SingleToInt32Bits(a.Data[i]) == BitConverter.SingleToInt32Bits(b.Data[i]),
                    $"smm residency pass {pass} logit {i}: plain={a.Data[i]} residency={b.Data[i]}");
        }

        // Pin-store round-trip on the SMM loader too: forced unload, retained pins, third pass.
        sw.FreeLayer(0);
        Assert.True(sw.PinnedExpertBytes > 0, "pinned planes should be retained across the unload cycle");
        using var a2 = plain.Forward(tokens);
        using var b2 = res.Forward(tokens);
        for (int i = 0; i < a2.ElementCount; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a2.Data[i]) == BitConverter.SingleToInt32Bits(b2.Data[i]),
                $"smm residency pin-round-trip logit {i}: plain={a2.Data[i]} residency={b2.Data[i]}");
    }

    [Fact]
    public void SmmLoader_FusedStack_IsRecognisedAndSlicesOnDemand()
    {
        // GgufToSmmConverter carries a qwen2moe-style fused 3D expert stack
        // ("blk.N.ffn_gate_exps.weight", no ".exps." segment) through verbatim. SmmLoader
        // must (a) recognise the file as MoE via IsMoEGguf and register the per-plane
        // TensorMeta the FfnLayer is built from, and (b) slice only the requested planes
        // on an on-demand LoadExpertWeights — the fused analog of the named-expert flow.
        using var fixture = NewMoEModel();
        string path = Path.Combine(_temp.Path, "fused.smm");
        WriteFusedSmm(path, fixture.Config);

        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);
        using var weights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming);
        weights.InitializeWeights();
        // No CreateTransformer here: that would start the async layer-0 preload and load every
        // plane, which is exactly what we want to prove the on-demand pass does NOT do.

        var block = weights.Blocks[0];
        Assert.Equal(QuantDType.F32, block.TensorMeta["RawWgateExp_1"].Dtype);
        Assert.Equal(QuantDType.F32, block.TensorMeta["RawWdownExp_3"].Dtype);

        var loader = new SmmLoader(qOps, path, fixture.Config);
        loader.LoadExpertWeights(0, new[] { 1, 3 }, weights);

        Assert.False(block.RawWgateExp?.ContainsKey(0) ?? false);
        Assert.True(block.RawWgateExp?.ContainsKey(1) ?? false);
        Assert.False(block.RawWgateExp?.ContainsKey(2) ?? false);
        Assert.True(block.RawWgateExp?.ContainsKey(3) ?? false);
        Assert.True(block.RawWupExp?.ContainsKey(1) ?? false);
        Assert.True(block.RawWdownExp?.ContainsKey(3) ?? false);

        int plane = fixture.Config.HiddenDim * fixture.Config.ResolvedExpertFfnDim * 4;
        Assert.Equal(plane, block.RawWgateExp![1].Length);
        Assert.Equal(plane, block.RawWdownExp![3].Length);
    }

    [Fact]
    public void Residency_DenseModel_IsIgnored()
    {
        // Dense (non-MoE) models must not wire residency even when configured.
        using var fixture = BuildFixture(ModelConfig.Learnable with { NumExperts = 0, TopKExperts = 0 }, 20260805);
        string path = ExportGguf(fixture);
        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);

        using var weights = ModelFactory.CreateWeights(fixture.Config, fixture.SharpConfig, qOps, path, LoadMode.Streaming,
            desiredResidentExperts: 2);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, fixture.SharpConfig);

        Assert.Equal(0, Assert.IsType<TransformerWeightsStreaming>(weights).EffectiveResidentExperts);

        using var tokens = new Tensor<int>(1, 3);
        new[] { 3, 17, 42 }.CopyTo(tokens.Data);
        using var logits = model.Forward(tokens);
        foreach (float f in logits.Data)
            Assert.True(float.IsFinite(f), "dense streaming forward must stay functional");
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private string ExportGguf(TinyModelFixture fixture)
    {
        string smm = Path.Combine(_temp.Path, "moe-resid.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smm, new SmmWriteOptions { Source = "training" }, model: fixture.Model);
        string gguf = Path.Combine(_temp.Path, "moe-resid.gguf");
        SmmToGufConverter.Convert(smm, gguf);
        return gguf;
    }

    /// <summary>
    /// Writes the .SMM a <c>GgufToSmmConverter</c> run over a qwen2moe-style GGUF would
    /// produce: each routed projection as one flat 3D <c>[in, out, experts]</c> tensor with
    /// the fused ("*_exps.weight") name — no ".exps." segment — plus the F32 router.
    /// The expert bytes never need to be mathematically valid: the test only loads, never
    /// forwards. F32 keeps the plane geometry trivially flat (<c>d0*d1*4</c>).
    /// </summary>
    private static void WriteFusedSmm(string path, ModelConfig config)
    {
        int hidden = config.HiddenDim;
        int ffn = config.ResolvedExpertFfnDim;
        int experts = config.NumExperts;
        var rng = new Random(20260805);
        var tensors = new List<SmmTensorData>();

        void AddFused(string name, int d0, int d1)
        {
            int plane = d0 * d1 * 4;
            var bytes = new byte[plane * experts];
            for (int e = 0; e < experts; e++)
            {
                var floats = new float[d0 * d1];
                for (int i = 0; i < floats.Length; i++) floats[i] = (float)rng.NextDouble() - 0.5f;
                Buffer.BlockCopy(floats, 0, bytes, e * plane, plane);
            }
            tensors.Add(new SmmTensorData
            {
                Name = $"blk.0.ffn_{name}_exps.weight",
                Shape = [d0, d1, experts],
                Dtype = QuantDType.F32,
                GetBytes = () => bytes,
            });
        }

        AddFused("gate", hidden, ffn);
        AddFused("up", hidden, ffn);
        AddFused("down", ffn, hidden);

        var router = new byte[hidden * experts * 4];
        new Random(7).NextBytes(router);
        tensors.Add(new SmmTensorData
        {
            Name = "blk.0.ffn_gate_inp.weight",
            Shape = [hidden, experts],
            Dtype = QuantDType.F32,
            GetBytes = () => router,
        });

        SmmWriter.Write(path, config, null, null, tensors, new SmmWriteOptions { Source = "training" });
    }

    private static TinyModelFixture NewMoEModel()
        => BuildFixture(ModelConfig.Learnable with { NumExperts = 4, TopKExperts = 2 }, 777);

    private static TinyModelFixture BuildFixture(ModelConfig config, int seed)
    {
        var sharpConfig = config.NumExperts > 0
            ? SharpMindConfig.ForModel(config.NumHeads, config.NumKvHeads, "mixtral") with { Hardware = HardwareTier.Scalar }
            : SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        if (config.NumExperts > 0)
            WeightInitializer.InitializeModelMoE(model, seed + 1013);
        return new TinyModelFixture { Weights = weights, Model = model, Config = config, SharpConfig = sharpConfig, Tokenizer = BuildTokenizer() };
    }

    private static Tokenizer BuildTokenizer()
    {
        var vocabObj = new JsonObject();
        int vocab = ModelConfig.Learnable.VocabSize;
        for (int i = 0; i < vocab - 4; i++) vocabObj[$"w{i}"] = i;
        vocabObj["<unk>"] = vocab - 4;
        vocabObj["<s>"] = vocab - 3;
        vocabObj["</s>"] = vocab - 2;
        vocabObj["<pad>"] = vocab - 1;

        var root = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokenizer"] = "whitespace",
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = "<unk>",
                ["bos"] = "<s>",
                ["eos"] = "</s>",
                ["pad"] = "<pad>",
                ["additional"] = new JsonArray(),
            },
            ["vocab"] = vocabObj,
            ["merges"] = new JsonArray(),
        };
        return Tokenizer.FromJson(root.ToJsonString());
    }

    private sealed class TinyModelFixture : IDisposable
    {
        public required TransformerWeights Weights { get; init; }
        public required SharpMind.Model.Transformer Model { get; init; }
        public required ModelConfig Config { get; init; }
        public required SharpMindConfig SharpConfig { get; init; }
        public required SharpMind.Tokenization.Tokenizer Tokenizer { get; init; }
        public void Dispose() { Model.Dispose(); Weights.Dispose(); }
    }
}