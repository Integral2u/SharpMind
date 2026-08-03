using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training.Optimizers;
using System.Text.Json.Nodes;

namespace SharpMind.Samples.Training;

/// <summary>
/// End-to-end .SMM training example: trains a tiny Learnable (PseudoLanguage)
/// model, exports it to <c>c:\temp\pseudo.smm</c> via
/// <see cref="SmmTrainingExporter"/>, reloads it via <see cref="SmmLoader"/>,
/// and runs a greedy generation loop on the reloaded weights.
/// </summary>
public static class SmmPseudoLanguageExample
{
    private const string SaveDir = @"c:\temp";
    private const string SavePath = @"c:\temp\pseudo.smm";
    private const int Steps = 100;

    public static void Run()
    {
        Console.WriteLine("== SharpMind SMM training example ==");
        Console.WriteLine();

        // 1. Create the model (tiny Learnable-style config, SIMD kernels).
        var modelConfig = new ModelConfig
        {
            VocabSize = 64,
            HiddenDim = 16,
            NumLayers = 1,
            NumHeads = 2,
            NumKvHeads = 2,
            FfnDim = 32,
            MaxSeqLen = 512,
        };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(modelConfig, sharpConfig);
        InitializeRandomly(weights, seed: 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        var learnConfig = new LearnableConfig
        {
            BatchSize = 4,
            SeqLen = 3,
            TrainSamples = 200,
            TestSamples = 50,
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
        };
        var generator = new LearnableGenerator(learnConfig, new Random(1234));

        // 2. Train the model with AdamW over finite-difference gradients.
        var parameters = model.Parameters().ToList();
        using var optimizer = new AdamW(parameters, lr: 0.02f, weightDecay: 0f);
        Train(model, parameters, generator, optimizer);

        Console.WriteLine();
        Console.WriteLine($"Next-token accuracy: {NextTokenAccuracy(model, generator, modelConfig.VocabSize):P1}");

        // 3. Build the tokenizer and export the model to .SMM.
        var tokenizer = BuildTokenizer(generator, modelConfig.VocabSize);
        Directory.CreateDirectory(SaveDir);
        SmmTrainingExporter.Export(weights, tokenizer, SavePath, new SmmWriteOptions
        {
            Compression = CompressionMode.Auto,
            Source = "training",
        });
        Console.WriteLine($"Saved: {SavePath} ({new FileInfo(SavePath).Length:N0} bytes)");

        // 4. Reload the .SMM from disk and rebuild the inference transformer.
        using var reloaded = Reload(out var reloadedTokenizer, out var reloadedConfig);

        Console.WriteLine();
        Console.WriteLine($"Reloaded config: {reloadedConfig}");
        Console.WriteLine($"Reloaded tokenizer vocab size: {reloadedTokenizer.VocabSize}");
        Console.WriteLine();

        // 5. Run greedy generation on a fresh pseudo-language prompt.
        var probe = new LearnableGenerator(
            new LearnableConfig { IncludeNouns = true, IncludeVerbs = true, IncludeObjects = true },
            new Random(7));
        var prompt = probe.GenerateTrainingSample().TokenIds;
        var ids = new List<int>(prompt);

        for (int step = 0; step < 3; step++)
        {
            using var tokens = Tensor<int>.From(ids.ToArray(), 1, ids.Count);
            using var logits = reloaded.Forward(tokens);
            int s = ids.Count;
            float max = float.NegativeInfinity;
            int best = -1;
            for (int v = 0; v < reloadedConfig.VocabSize; v++)
            {
                float l = logits.Data[(s - 1) * reloadedConfig.VocabSize + v];
                if (float.IsFinite(l) && l > max) { max = l; best = v; }
            }
            if (best < 0 || best >= reloadedConfig.VocabSize)
                throw new InvalidOperationException($"Generation produced invalid id {best}.");
            ids.Add(best);
        }

        string decoded = reloadedTokenizer.Decode([.. ids], skipSpecials: false);
        Console.WriteLine($"Prompt ids : [{string.Join(", ", prompt)}]");
        Console.WriteLine($"Generated  : [{string.Join(", ", ids)}]");
        Console.WriteLine($"Decoded    : {decoded}");
    }

    /// <summary>
    /// Randomizes all float weights (GPT-2 style, std 0.02) so the network
    /// produces non-trivial logits and the finite-difference gradients below
    /// are meaningful. <see cref="ModelFactory.CreateForTraining"/> allocates
    /// zeroed tensors, which would otherwise pin the loss at log(V).
    /// </summary>
    private static void InitializeRandomly(TransformerWeights weights, int seed)
    {
        var rng = new Random(seed);
        const float std = 0.02f;
        FillNormal(weights.EmbeddingWeight.Data, rng, std);
        foreach (var block in weights.Blocks)
        {
            FillNormal(block.Wq!.Data, rng, std);
            FillNormal(block.Wk!.Data, rng, std);
            FillNormal(block.Wv!.Data, rng, std);
            FillNormal(block.Wo!.Data, rng, std);
            FillNormal(block.Wf1!.Data, rng, std);
            FillNormal(block.Wf2!.Data, rng, std);
        }
    }

    private static void FillNormal(Span<float> data, Random rng, float std)
    {
        for (int i = 0; i < data.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            data[i] = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }

    private static Transformer Reload(out Tokenizer tokenizer, out ModelConfig config)
    {
        SmmLoader.Load(SavePath, null, out _, out config, out var rawTokenizer);
        tokenizer = rawTokenizer ?? throw new InvalidDataException("SMM file is missing its tokenizer (smm.tokenizer).");
        var sharpConfig = config.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        var reloaded = ModelFactory.CreateWeights(config, sharpConfig, qOps, SavePath, LoadMode.Full);
        reloaded.InitializeWeights();
        return ModelFactory.CreateTransformer(reloaded, sharpConfig, null, false);
    }

    /// <summary>
    /// Trains every parameter for next-token prediction. Gradients are
    /// estimated by finite differences (the same approach SharpMind.Samples.Tests
    /// uses for the Learnable generator) and applied by AdamW.
    /// </summary>
    private static void Train(Transformer model, List<Parameter> parameters, LearnableGenerator generator, AdamW optimizer)
    {
        const int batchSize = 4;
        const float h = 1e-3f;

        int vocab = model.Config.VocabSize;

        for (int step = 0; step < Steps; step++)
        {
            var batch = generator.GenerateBatch(batchSize);
            int seqLen = batch.TokenIds.Length / batchSize;
            int n = batchSize * seqLen;
            var targets = new int[n];
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen - 1; s++)
                    targets[b * seqLen + s] = batch.TokenIds[b * seqLen + s + 1];
                targets[b * seqLen + seqLen - 1] = -100;
            }

            using var tokens = Tensor<int>.From(batch.TokenIds, batchSize, seqLen);
            using var logits = model.Forward(tokens);
            float loss = CrossEntropy(logits, targets, n, vocab);

            optimizer.ZeroGrad();
            foreach (var p in parameters)
            {
                var data = p.Data.Data;
                var grad = p.Grad.Data;
                for (int i = 0; i < data.Length; i++)
                {
                    float original = data[i];
                    data[i] = original + h;
                    float plus = LossFor(model, tokens, targets, n, vocab);
                    data[i] = original - h;
                    float minus = LossFor(model, tokens, targets, n, vocab);
                    data[i] = original;
                    grad[i] = (plus - minus) / (2 * h);
                }
            }

            optimizer.Update();

            if (loss is float.NaN or float.PositiveInfinity)
                throw new InvalidOperationException($"Training loss diverged at step {step}: loss={loss}.");

            if (step % 20 == 0)
                Console.WriteLine($"step {step + 1,3}/{Steps}: loss = {loss:F4}");
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

    private static float NextTokenAccuracy(Transformer model, LearnableGenerator generator, int vocab)
    {
        int total = 0;
        int correct = 0;
        for (int i = 0; i < 20; i++)
        {
            var ids = generator.GenerateTrainingSample().TokenIds;
            using var tokens = Tensor<int>.From(ids, 1, ids.Length);
            using var logits = model.Forward(tokens);
            for (int s = 0; s < ids.Length - 1; s++)
            {
                int target = ids[s + 1];
                float max = float.NegativeInfinity;
                int best = -1;
                for (int v = 0; v < vocab; v++)
                {
                    float l = logits.Data[s * vocab + v];
                    if (float.IsFinite(l) && l > max) { max = l; best = v; }
                }
                total++;
                if (best == target) correct++;
            }
        }
        return total > 0 ? (float)correct / total : 0f;
    }

    /// <summary>
    /// Builds a tokenizer whose vocab covers every model output ID (words at
    /// the generator's IDs, filler tokens for the unused rows) so decode is
    /// well-defined for any greedy argmax.
    /// </summary>
    private static Tokenizer BuildTokenizer(LearnableGenerator generator, int vocabSize)
    {
        var vocabObj = new JsonObject();
        var words = generator.Vocabulary;
        for (int i = 0; i < words.Count; i++)
            vocabObj[words[i]] = i;
        // The four specials must live inside [0, vocabSize) so every token id
        // the tokenizer can emit maps to a valid embedding row. Untrained filler
        // rows fill the gap between the words and the specials.
        for (int i = words.Count; i < vocabSize - 4; i++)
            vocabObj[$"<t{i}>"] = i;
        vocabObj["<unk>"] = vocabSize - 4;
        vocabObj["<s>"] = vocabSize - 3;
        vocabObj["</s>"] = vocabSize - 2;
        vocabObj["<pad>"] = vocabSize - 1;

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
}
