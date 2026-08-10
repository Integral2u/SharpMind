using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.CUI.App;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Samples.Examples;
using SharpMind.Tokenization;
using SharpMind.Training;
using System.Text;
using System.Text.Json.Nodes;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// End-to-end coverage of the .SMM container: train a tiny Learnable model,
/// export it via <see cref="SmmTrainingExporter"/>, reload via
/// <see cref="SmmLoader"/>, and assert tensor / logits parity plus generation.
/// Also covers F16/Q8_0 quantized exports, GGUF→SMM conversion of
/// already-quantized data, and plugin embedding.
/// </summary>
public class SmmExportLoadTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    public static IEnumerable<object[]> QuantLevels()
    {
        yield return [QuantDType.F16];
        yield return [QuantDType.Q8_0];
    }

    [Fact]
    public void TrainingExport_F32_ReloadsWithParity()
    {
        using var fixture = TrainFixture();

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions
        {
            Source = "training",
        });

        using var reloaded = Reload(smmPath, fixture.SharpConfig, out var reloadedConfig, out var reloadedTokenizer);

        // Config round-trip (exporter tags the default architecture)
        Assert.Equal("gpt2", reloadedConfig.Architecture);
        Assert.Equal(fixture.Config.VocabSize, reloadedConfig.VocabSize);
        Assert.Equal(fixture.Config.HiddenDim, reloadedConfig.HiddenDim);
        Assert.Equal(fixture.Config.NumLayers, reloadedConfig.NumLayers);
        Assert.Equal(fixture.Config.NumHeads, reloadedConfig.NumHeads);
        Assert.Equal(fixture.Config.NumKvHeads, reloadedConfig.NumKvHeads);
        Assert.Equal(fixture.Config.FfnDim, reloadedConfig.FfnDim);
        Assert.Equal(fixture.Config.MaxSeqLen, reloadedConfig.MaxSeqLen);

        // Tokenizer round-trip — same vocab size and word→id mapping
        Assert.NotNull(reloadedTokenizer);
        Assert.Equal(fixture.Tokenizer.VocabSize, reloadedTokenizer!.VocabSize);
        foreach (string word in fixture.Generator.Vocabulary)
            Assert.Equal(fixture.Tokenizer.TokenToId(word), reloadedTokenizer.TokenToId(word));

        // Tensor parity — F32 is an exact round trip
        AssertWeightsClose(fixture.Weights, reloaded, 1e-6f, "f32");

        // Logits parity — reloaded weights drive the exact same forward pass
        using var model = ModelFactory.CreateTransformer(reloaded, fixture.SharpConfig, null, false);
        AssertLogitsParity(fixture.Model, model, fixture.Generator, fixture.Config.VocabSize);
        AssertGenerationWorks(model, reloadedTokenizer, fixture.Config.VocabSize);
    }

    [Fact]
    public void TrainingExport_StoresAndReloadsChatTemplate()
    {
        using var fixture = TrainFixture();
        string chatTemplate =
            "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{{ '<|im_start|>assistant\\n' }}";

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions
        {
            Source = "training",
        }, chatTemplate: chatTemplate);

        // Round-trips through the raw metadata ...
        Assert.Equal(chatTemplate, SmmLoader.LoadMeta(smmPath).GetChatTemplate());

        // ... and through the training pipeline helper.
        Assert.Equal(chatTemplate, SmmTrainingPipeline.LoadChatTemplate(smmPath));
    }

    [Fact]
    public void TrainingExport_StoresAndReloadsSkillsAndSystemPrompt()
    {
        using var fixture = TrainFixture();
        string[] skills = ["## Skill one\nDo this thing.", "# Skill two\nDo that thing."];
        string systemPrompt = "You are a helpful research assistant embedded in a model file.";

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions
        {
            Source = "training",
            Skills = skills,
            SystemPrompt = systemPrompt,
        });

        // Absent keys stay null/empty for a container written without them.
        var meta = SmmLoader.LoadMeta(smmPath);
        Assert.Equal(systemPrompt, SmmLoader.LoadSystemPromptFromMeta(meta));
        Assert.Equal(skills, SmmLoader.LoadSkillsFromMeta(meta));

        // Convenience path-based accessors agree.
        Assert.Equal(systemPrompt, SmmLoader.LoadSystemPrompt(smmPath));
        Assert.Equal(skills, SmmLoader.LoadSkills(smmPath));
    }

    [Fact]
    public void TrainingExport_StoresSystemPromptSkillsAndPluginsTogether()
    {
        using var fixture = TrainFixture();
        byte[] asm = File.ReadAllBytes(typeof(SmmExportLoadTests).Assembly.Location);
        string[] skills = ["# Skill\nDo the thing."];
        string systemPrompt = "Default system prompt.";

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        // Same option shape TrainRunner.BuildEmbedOptions emits: plugins flip on
        // SmmOutputs.Plugins while skills/system prompt ride on the default set.
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions
        {
            Source = "training",
            Outputs = SmmOutputs.Default | SmmOutputs.Plugins,
            Skills = skills,
            SystemPrompt = systemPrompt,
            Plugins = [new SmmPluginEntry { Name = "SharpMind.Plugins.Embed.dll", AssemblyBytes = asm }],
        });

        var meta = SmmLoader.LoadMeta(smmPath);
        Assert.Equal(systemPrompt, SmmLoader.LoadSystemPromptFromMeta(meta));
        Assert.Equal(skills, SmmLoader.LoadSkillsFromMeta(meta));

        var plugins = SmmLoader.LoadPlugins(smmPath);
        Assert.Single(plugins);
        Assert.Equal("SharpMind.Plugins.Embed.dll", plugins[0].Name);
        Assert.Equal(asm, plugins[0].AssemblyBytes);
    }

    [Fact]
    public void TrainingExport_WithoutSkills_PromptStaysEmpty()
    {
        using var fixture = TrainFixture();

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions());

        var meta = SmmLoader.LoadMeta(smmPath);
        Assert.Null(SmmLoader.LoadSystemPromptFromMeta(meta));
        Assert.Empty(SmmLoader.LoadSkillsFromMeta(meta));
    }

    [Theory]
    [MemberData(nameof(QuantLevels))]
    public void TrainingExport_Quantized_ReloadsWithinTolerance(QuantDType dtype)
    {
        using var fixture = TrainFixture();

        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions
        {
            QuantizationLevel = dtype,
            Source = "training",
        });

        using var reloaded = Reload(smmPath, fixture.SharpConfig, out _, out var reloadedTokenizer);
        Assert.NotNull(reloadedTokenizer);

        float tolerance = dtype == QuantDType.F16 ? 1e-3f : 1e-2f;
        AssertWeightsClose(fixture.Weights, reloaded, tolerance, dtype.ToString());

        using var model = ModelFactory.CreateTransformer(reloaded, fixture.SharpConfig, null, false);
        AssertGenerationWorks(model, reloadedTokenizer!, fixture.Config.VocabSize);
    }

    [Fact]
    public void BlockQuantization_RejectsDimNotMultipleOf32()
    {
        var values = Enumerable.Range(0, 8).Select(i => (float)i).ToArray();
        Assert.Throws<InvalidOperationException>(() => TensorQuantizer.Quantize(values, [8], QuantDType.Q8_0));
        Assert.Throws<InvalidOperationException>(() => TensorQuantizer.Quantize(values, [4, 2], QuantDType.Q4_0));
    }

    [Fact]
    public void GgufConversion_RoundTripsThroughSmm()
    {
        string ggufPath = Path.Combine(_temp.Path, "tiny.gguf");
        int vocab = 64;
        var tokens = BuildGgufTokens(vocab);
        string chatTemplate = "{{- messages }}";
        WriteTinyGguf(ggufPath, vocab, tokens, chatTemplate);

        string smmPath = Path.Combine(_temp.Path, "tiny.smm");
        SharpMind.Model.Format.Conversion.GgufToSmmConverter.Convert(ggufPath, smmPath);

        // Embedded metadata round-trips
        SmmLoader.Load(smmPath, null, out _, out var config, out var tokenizer);
        Assert.Equal("gpt2", config.Architecture);
        Assert.Equal(64, config.VocabSize);
        Assert.NotNull(tokenizer);
        Assert.Equal(vocab, tokenizer!.VocabSize);
        Assert.Equal(4, tokenizer.TokenToId("king"));
        Assert.Equal("king", tokenizer.IdToToken(4));
        var meta = SmmLoader.LoadMeta(smmPath);
        Assert.Equal(chatTemplate, meta.GetChatTemplate());

        // GGUF-loaded and SMM-loaded weights must be bit-identical (raw bytes copied verbatim)
        using var ggufWeights = LoadWeightsFrom(ggufPath, config, out _);
        using var smmWeights = LoadWeightsFrom(smmPath, config, out _);
        AssertWeightsClose(ggufWeights, smmWeights, 1e-6f, "gguf");
    }

    [Fact]
    public void GgufConversion_EmbedsSkillsAndPrompt_ButKeepsGgufClean()
    {
        string ggufPath = Path.Combine(_temp.Path, "tiny.gguf");
        int vocab = 64;
        var tokens = BuildGgufTokens(vocab);
        string chatTemplate = "{{- messages }}";
        WriteTinyGguf(ggufPath, vocab, tokens, chatTemplate);

        string smmPath = Path.Combine(_temp.Path, "tiny.smm");
        SharpMind.Model.Format.Conversion.GgufToSmmConverter.Convert(ggufPath, smmPath, new SmmWriteOptions
        {
            Skills = ["# Skill\nDo the thing."],
            SystemPrompt = "Default system prompt.",
        });

        var meta = SmmLoader.LoadMeta(smmPath);
        Assert.Equal("Default system prompt.", SmmLoader.LoadSystemPromptFromMeta(meta));
        Assert.Equal(["# Skill\nDo the thing."], SmmLoader.LoadSkillsFromMeta(meta));

        // SMM-only: converting back to GGUF must NOT leak skills/system prompt
        // into the GGUF metadata (GGUF stays a clean, portable container).
        string roundTrip = Path.Combine(_temp.Path, "tiny.roundtrip.gguf");
        SmmToGufConverter.Convert(smmPath, roundTrip);
        var ggufMeta = GgufLoader.LoadMeta(roundTrip);
        Assert.Empty(ggufMeta.KvPairs.Where(k =>
            k.Key == SmmConstants.SystemPromptKey || k.Key == SmmConstants.SkillsKey));
    }

    [Fact]
    public void GgufConversion_CancelledBeforeStart_WritesNothing()
    {
        string ggufPath = Path.Combine(_temp.Path, "tiny.gguf");
        WriteTinyGguf(ggufPath, 64, BuildGgufTokens(64), "{{- messages }}");

        string smmPath = Path.Combine(_temp.Path, "tiny.smm");
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            GgufToSmmConverter.Convert(ggufPath, smmPath, new SmmWriteOptions(), cts.Token));

        // No partial output and no leftover temp file.
        Assert.False(File.Exists(smmPath));
        Assert.False(File.Exists(smmPath + ".tmp"));
    }

    [Fact]
    public void SmmConversion_CancelledBeforeStart_WritesNothing()
    {
        using var fixture = TrainFixture();
        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath, new SmmWriteOptions());

        string ggufPath = Path.Combine(_temp.Path, "model.gguf");
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() => SmmToGufConverter.Convert(smmPath, ggufPath, cts.Token));

        Assert.False(File.Exists(ggufPath));
        Assert.False(File.Exists(ggufPath + ".tmp"));
    }

    [Fact]
    public void GgufConversion_ReportsProgress_UpToComplete()
    {
        string ggufPath = Path.Combine(_temp.Path, "tiny.gguf");
        WriteTinyGguf(ggufPath, 64, BuildGgufTokens(64), "{{- messages }}");

        string smmPath = Path.Combine(_temp.Path, "tiny.smm");
        var reported = new List<float>();
        GgufToSmmConverter.Convert(ggufPath, smmPath, new SmmWriteOptions(), CancellationToken.None,
            new InlineProgress(reported.Add));

        Assert.True(File.Exists(smmPath));
        Assert.NotEmpty(reported);
        Assert.InRange(reported[^1], 0.99f, 1.0f);
        // Progress must be monotonic non-decreasing.
        for (int i = 1; i < reported.Count; i++)
            Assert.True(reported[i] >= reported[i - 1], "progress went backwards");
    }

    [Fact]
    public void AgentBuilder_SkillContentAndAdditionalPrompt()
    {
        var builder = new SharpMind.Inference.Agent.AgentBuilder()
            .WithSkillContent("# Skill A")
            .WithSkillContent("# Skill A") // idempotent
            .WithSkillContent("# Skill B")
            .WithAdditionalSystemPrompt("Embedded default.")
            .WithAdditionalSystemPrompt("Embedded default."); // idempotent

        string prompt = builder.BuildAgentPrompt();
        Assert.Contains("## Skills", prompt);
        Assert.Contains("# Skill A", prompt);
        Assert.Contains("# Skill B", prompt);
        Assert.Single(builder.AdditionalSystemPrompts);
        Assert.Equal("Embedded default.", builder.AdditionalSystemPrompts[0]);
    }

    [Fact]
    public void Plugins_EmbedAndLoad()
    {
        using var fixture = TrainFixture();
        byte[] asm = File.ReadAllBytes(typeof(SmmExportLoadTests).Assembly.Location);
        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, null, smmPath, new SmmWriteOptions
        {
            Outputs = SmmOutputs.Plugins,
            Plugins =
            [
                new SmmPluginEntry { Name = "SharpMind.Plugins.Weather.dll", AssemblyBytes = asm, Recommended = true },
                new SmmPluginEntry { Name = "SharpMind.Plugins.Calculator.dll", AssemblyBytes = asm, Recommended = false },
            ],
        });

        var plugins = SmmLoader.LoadPlugins(smmPath);
        Assert.Equal(2, plugins.Count);
        Assert.Equal("SharpMind.Plugins.Weather.dll", plugins[0].Name);
        Assert.True(plugins[0].Recommended);
        Assert.False(plugins[1].Recommended);
        Assert.Equal(asm, plugins[0].AssemblyBytes);
    }

    [Fact]
    public void EmbeddedPlugins_DiscoveredBySessionLauncher()
    {
        // An SMM container carrying a plugin whose assembly happens to be this test
        // assembly — SessionLauncher must materialize the embedded components and
        // derive the exact tool-name set the permission gate will restrict.
        using var fixture = TrainFixture();
        byte[] asm = File.ReadAllBytes(typeof(SmmExportLoadTests).Assembly.Location);
        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, null, smmPath, new SmmWriteOptions
        {
            Outputs = SmmOutputs.Plugins,
            Plugins =
            [
                new SmmPluginEntry { Name = "SharpMind.Plugins.Probe.dll", AssemblyBytes = asm, Recommended = true },
            ],
        });

        var embedded = SessionLauncher.LoadEmbeddedPlugins(smmPath);

        Assert.NotNull(embedded);
        Assert.Equal(["SharpMind.Plugins.Probe.dll"], embedded!.AssemblyNames);
        Assert.NotEmpty(embedded.Plugins.Tools);
        Assert.True(embedded.Plugins.Tools.Any(t => t.GetType().GetMethod("Probe") is not null),
            "The embedded assembly's [ToolDesc] class should be materialized as a tool.");
        Assert.Contains("Probe", embedded.ToolNames);
    }

    [Fact]
    public void EmbeddedPlugins_NoPluginSection_ReturnsNull()
    {
        using var fixture = TrainFixture();
        string smmPath = Path.Combine(_temp.Path, "model.smm");
        SmmTrainingExporter.Export(fixture.Weights, null, smmPath, new SmmWriteOptions());

        Assert.Null(SessionLauncher.LoadEmbeddedPlugins(smmPath));
    }

    /// <summary>Lives in the test assembly so plugin discovery can find it via raw assembly bytes.</summary>
    public class ProbeEmbeddedTool
    {
        [ToolDesc("Probe tool used to verify embedded plugin discovery.")]
        public string Probe() => "ok";
    }

    /// <summary>Calls back synchronously on the calling thread (no async marshalling).</summary>
    private sealed class InlineProgress : IProgress<float>
    {
        private readonly Action<float> _onReport;
        public InlineProgress(Action<float> onReport) => _onReport = onReport;

        public void Report(float value) => _onReport(value);
    }

    // ── Fixture ────────────────────────────────────────────────────────────

    private sealed class TrainedFixture : IDisposable
    {
        public required TransformerWeights Weights { get; init; }
        public required Transformer Model { get; init; }
        public required ModelConfig Config { get; init; }
        public required SharpMindConfig SharpConfig { get; init; }
        public required Tokenizer Tokenizer { get; init; }
        public required LearnableGenerator Generator { get; init; }

        public void Dispose()
        {
            Model.Dispose();
            Weights.Dispose();
            Generator.Dispose();
        }
    }

    private static TrainedFixture TrainFixture()
    {
        var modelConfig = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        var learnConfig = new LearnableConfig
        {
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
        };
        var generator = new LearnableGenerator(learnConfig, new Random(1234));
        var tokenizer = BuildTokenizer(generator);

        TrainEmbedding(model, weights, generator);

        return new TrainedFixture
        {
            Weights = weights,
            Model = model,
            Config = modelConfig,
            SharpConfig = sharpConfig,
            Tokenizer = tokenizer,
            Generator = generator,
        };
    }

    /// <summary>
    /// Builds a tokenizer whose vocab covers every model output ID (words at
    /// the generator's IDs, filler tokens for the unused rows) so decode is
    /// well-defined for any greedy argmax.
    /// </summary>
    private static Tokenizer BuildTokenizer(LearnableGenerator generator)
    {
        var vocabObj = new JsonObject();
        var words = generator.Vocabulary;
        for (int i = 0; i < words.Count; i++)
            vocabObj[words[i]] = i;
        for (int i = words.Count; i < ModelConfig.Learnable.VocabSize; i++)
            vocabObj[$"<t{i}>"] = i;
        int tail = ModelConfig.Learnable.VocabSize;
        vocabObj["<unk>"] = tail;
        vocabObj["<s>"] = tail + 1;
        vocabObj["</s>"] = tail + 2;
        vocabObj["<pad>"] = tail + 3;

        var root = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokeniser"] = "whitespace",
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

    private static void TrainEmbedding(Transformer model, TransformerWeights weights, LearnableGenerator generator)
    {
        const int batch = 2;
        const int seqLen = 3;
        const int steps = 25;
        const int gradSamples = 12;
        const float lr = 0.05f;
        const float h = 1e-3f;

        int vocab = model.Config.VocabSize;
        var embed = weights.EmbeddingWeight;
        var data = embed.Data;
        int gradCount = Math.Min(gradSamples, data.Length);
        var grad = new float[data.Length];
        var tokens = new int[batch * seqLen];
        var targets = new int[batch * seqLen];

        for (int step = 0; step < steps; step++)
        {
            for (int b = 0; b < batch; b++)
            {
                var sample = generator.GenerateTrainingSample();
                Array.Copy(sample.TokenIds, 0, tokens, b * seqLen, seqLen);
            }
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen - 1; s++)
                    targets[b * seqLen + s] = tokens[b * seqLen + s + 1];
                targets[b * seqLen + seqLen - 1] = -100;
            }

            using var t = Tensor<int>.From(tokens, batch, seqLen);
            using var logits = model.Forward(t);
            float loss = CrossEntropy(logits, targets, batch * seqLen, vocab);

            for (int i = 0; i < gradCount; i++)
            {
                float original = data[i];
                data[i] = original + h;
                float plus = LossFor(model, t, targets, batch * seqLen, vocab);
                data[i] = original - h;
                float minus = LossFor(model, t, targets, batch * seqLen, vocab);
                data[i] = original;
                grad[i] = (plus - minus) / (2 * h);
            }

            float norm = MathF.Sqrt(grad.AsSpan(0, gradCount).ToArray().Select(g => g * g).Sum() / gradCount + 1e-8f);
            float scale = norm > 1f ? 1f / norm : 1f;
            for (int i = 0; i < gradCount; i++)
                data[i] -= lr * scale * grad[i];

            if (loss is float.NaN or float.PositiveInfinity)
                throw new InvalidOperationException(
                    $"Training loss diverged at step {step}: loss={loss}, " +
                    $"max|grad|={grad.AsSpan(0, gradCount).ToArray().Max(MathF.Abs):G6}.");
        }
    }

    private static float LossFor(Transformer model, Tensor<int> tokens, int[] targets, int n, int vocab)
    {
        using var logits = model.Forward(tokens);
        return CrossEntropy(logits, targets, n, vocab);
    }

    private static float CrossEntropy(Tensor<float> logits, int[] targets, int n, int vocab)
    {
        float sum = 0;
        int total = 0;
        for (int i = 0; i < n; i++)
        {
            int t = targets[i];
            if (t < 0) continue;
            float max = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++) max = MathF.Max(max, logits.Data[i * vocab + v]);
            float s = 0;
            for (int v = 0; v < vocab; v++) s += MathF.Exp(logits.Data[i * vocab + v] - max);
            sum += max - logits.Data[i * vocab + t] + MathF.Log(s);
            total++;
        }
        return total > 0 ? sum / total : 0f;
    }

    // ── Reload helpers ─────────────────────────────────────────────────────

    private static TransformerWeights Reload(
        string path, SharpMindConfig sharpConfig, out ModelConfig config, out Tokenizer? tokenizer)
    {
        SmmLoader.Load(path, null, out _, out config, out tokenizer);
        return LoadWeightsFrom(path, config, sharpConfig);
    }

    private static TransformerWeights LoadWeightsFrom(
        string path, ModelConfig config, SharpMindConfig sharpConfig)
    {
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        var weights = ModelFactory.CreateWeights(config, sharpConfig, qOps, path, LoadMode.Full);
        weights.InitializeWeights();
        return weights;
    }

    private static TransformerWeights LoadWeightsFrom(string path, ModelConfig config, out SharpMindConfig sharpConfig)
    {
        sharpConfig = config.ForModel();
        return LoadWeightsFrom(path, config, sharpConfig);
    }

    private static void AssertWeightsClose(TransformerWeights a, TransformerWeights b, float tolerance, string label)
    {
        AssertTensorClose(a.EmbeddingWeight, b.EmbeddingWeight, tolerance, label + ".token_embd");
        AssertTensorClose(a.FinalNormWeight, b.FinalNormWeight, tolerance, label + ".output_norm");
        Assert.Equal(a.Blocks.Length, b.Blocks.Length);
        for (int i = 0; i < a.Blocks.Length; i++)
        {
            var x = a.Blocks[i];
            var y = b.Blocks[i];
            string blk = $"{label}.blk{i}";
            AssertTensorClose(x.Wq, y.Wq, tolerance, blk + ".attn_q");
            AssertTensorClose(x.Wk, y.Wk, tolerance, blk + ".attn_k");
            AssertTensorClose(x.Wv, y.Wv, tolerance, blk + ".attn_v");
            AssertTensorClose(x.Wo, y.Wo, tolerance, blk + ".attn_output");
            AssertTensorClose(x.Wf1, y.Wf1, tolerance, blk + ".ffn_up");
            AssertTensorClose(x.Wf2, y.Wf2, tolerance, blk + ".ffn_down");
            AssertTensorClose(x.WqBias, y.WqBias, tolerance, blk + ".attn_q.bias");
            AssertTensorClose(x.WkBias, y.WkBias, tolerance, blk + ".attn_k.bias");
            AssertTensorClose(x.WvBias, y.WvBias, tolerance, blk + ".attn_v.bias");
            AssertTensorClose(x.WoBias, y.WoBias, tolerance, blk + ".attn_output.bias");
            AssertTensorClose(x.Wf1Bias, y.Wf1Bias, tolerance, blk + ".ffn.bias");
            AssertTensorClose(x.Wf2Bias, y.Wf2Bias, tolerance, blk + ".ffn_down.bias");
            AssertTensorClose(x.Norm1W, y.Norm1W, tolerance, blk + ".attn_norm");
            AssertTensorClose(x.Norm2W, y.Norm2W, tolerance, blk + ".ffn_norm");
        }
    }

    private static void AssertTensorClose(Tensor<float> a, Tensor<float> b, float tolerance, string label)
    {
        Assert.NotNull(a);
        Assert.NotNull(b);
        Assert.Equal(a.Shape, b.Shape);
        Assert.Equal(a.ElementCount, b.ElementCount);
        float maxAbs = 0f;
        for (int i = 0; i < a.ElementCount; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(a.Data[i]));
        float bound = tolerance * MathF.Max(1f, maxAbs);
        for (int i = 0; i < a.ElementCount; i++)
        {
            float diff = MathF.Abs(a.Data[i] - b.Data[i]);
            Assert.True(diff <= bound, $"{label}: diff {diff} at [{i}] exceeds {bound}");
        }
    }

    private static void AssertLogitsParity(Transformer original, Transformer reloaded, LearnableGenerator generator, int vocab)
    {
        var sample = generator.GenerateTrainingSample();
        var prompt = sample.TokenIds;

        float[] a = ComputeLogits(original, prompt, vocab);
        float[] b = ComputeLogits(reloaded, prompt, vocab);
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            Assert.True(diff < 1e-3f, $"logits[{i}] differ: {a[i]} vs {b[i]} ({diff})");
        }
    }

    private static float[] ComputeLogits(Transformer model, int[] tokenIds, int vocab)
    {
        using var t = Tensor<int>.From(tokenIds, 1, tokenIds.Length);
        using var logits = model.Forward(t);
        var result = new float[tokenIds.Length * vocab];
        logits.Data.CopyTo(result);
        return result;
    }

    private static void AssertGenerationWorks(Transformer model, Tokenizer tokenizer, int vocab)
    {
        var generator = new LearnableGenerator(
            new LearnableConfig { IncludeNouns = true, IncludeVerbs = true, IncludeObjects = true },
            new Random(1));
        var prompt = generator.GenerateTrainingSample().TokenIds;
        var ids = new List<int>(prompt);

        for (int step = 0; step < 3; step++)
        {
            using var tokens = Tensor<int>.From(ids.ToArray(), 1, ids.Count);
            using var logits = model.Forward(tokens);
            int s = ids.Count;
            float max = float.NegativeInfinity;
            int best = -1;
            for (int v = 0; v < vocab; v++)
            {
                float l = logits.Data[(s - 1) * vocab + v];
                if (float.IsFinite(l) && l > max) { max = l; best = v; }
            }
            Assert.True(best >= 0 && best < vocab, $"argmax produced {best}");
            ids.Add(best);
        }

        string decoded = tokenizer.Decode([.. ids], skipSpecials: false);
        Assert.False(string.IsNullOrWhiteSpace(decoded));
    }

    // ── Tiny GGUF writer (mirrors what llama.cpp emits) ────────────────────

    private static string[] BuildGgufTokens(int vocab)
    {
        var tokens = new string[vocab];
        string[] baseTokens = ["<unk>", "<s>", "</s>", "<pad>", "king", "queen", "dog", "cat"];
        for (int i = 0; i < vocab; i++)
            tokens[i] = i < baseTokens.Length ? baseTokens[i] : $"<t{i}>";
        return tokens;
    }

    private static void WriteTinyGguf(string path, int vocab, string[] tokens, string chatTemplate)
    {
        int hidden = 32;
        using var fs = new FileStream(path, FileMode.Create);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(0x46554747u); // "GGUF"
        w.Write(3u);          // version
        w.Write(3L);          // tensor_count
        w.Write(11L);         // kv_count

        void KvString(string key, string value) { WriteGgufString(w, key); w.Write(8u); WriteGgufString(w, value); }
        void KvU32(string key, uint value) { WriteGgufString(w, key); w.Write(4u); w.Write(value); }
        void KvStringArray(string key, string[] values)
        {
            WriteGgufString(w, key);
            w.Write(9u);
            w.Write(8u); // element type: string
            w.Write((ulong)values.Length);
            foreach (string v in values) WriteGgufString(w, v);
        }

        KvString("general.architecture", "gpt2");
        KvU32("gpt2.embedding_length", (uint)hidden);
        KvU32("gpt2.block_count", 1);
        KvU32("gpt2.feed_forward_length", 64);
        KvU32("gpt2.context_length", 16);
        KvU32("gpt2.attention.head_count", 4);
        KvU32("gpt2.attention.head_count_kv", 4);
        KvStringArray("tokenizer.ggml.tokens", tokens);
        KvU32("tokenizer.ggml.bos_token_id", 1);
        KvU32("tokenizer.ggml.eos_token_id", 2);
        KvString("tokenizer.chat_template", chatTemplate);

        byte[] emb = MakeFloatData(vocab * hidden, seed: 7);
        byte[] norm = MakeFloatData(hidden, seed: 11);
        byte[] attn = MakeQ8_0Data(MakeFloatData(hidden * hidden, seed: 13));

        void TensorInfo(string name, int[] shape, uint dtype, long offset)
        {
            WriteGgufString(w, name);
            w.Write((uint)shape.Length);
            foreach (int d in shape) w.Write((ulong)d);
            w.Write(dtype);
            w.Write((ulong)offset);
        }

        TensorInfo("token_embd.weight", [vocab, hidden], 0u, 0);
        TensorInfo("output_norm.weight", [hidden], 0u, emb.Length);
        TensorInfo("blk.0.attn_q.weight", [hidden, hidden], 8u, emb.Length + norm.Length);

        long end = fs.Position;
        long aligned = (end + 31) & ~31L;
        if (aligned > end) fs.Write(new byte[aligned - end]);

        fs.Write(emb, 0, emb.Length);
        fs.Write(norm, 0, norm.Length);
        fs.Write(attn, 0, attn.Length);
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static byte[] MakeFloatData(int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++)
            floats[i] = (float)(rng.NextDouble() * 2 - 1);
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] MakeQ8_0Data(byte[] f32Bytes)
    {
        const int blockSize = 32;
        int count = f32Bytes.Length / 4;
        var floats = new float[count];
        Buffer.BlockCopy(f32Bytes, 0, floats, 0, f32Bytes.Length);

        int blocks = count / blockSize;
        var outBytes = new byte[blocks * 34];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < blockSize; i++)
                amax = MathF.Max(amax, MathF.Abs(floats[b * blockSize + i]));
            float d = amax == 0 ? 0f : amax / 127f;
            ushort d16 = QuantizationKernels.FloatToHalf_F16C(d);
            int o = b * 34;
            outBytes[o] = (byte)(d16 & 0xFF);
            outBytes[o + 1] = (byte)(d16 >> 8);
            for (int i = 0; i < blockSize; i++)
            {
                float q = d == 0 ? 0f : floats[b * blockSize + i] / d;
                outBytes[o + 2 + i] = (byte)Math.Clamp((int)MathF.Round(q), -128, 127);
            }
        }
        return outBytes;
    }
}
