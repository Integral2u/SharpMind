using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Samples.Examples;

public static class InteractiveChat
{
    private const string GgufFileName = "qwen2-0_5b-instruct-q4_k_m.gguf";// "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"; // qwen2-0_5b-instruct-q4_k_m.gguf";// DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf";

    public static Task RunAsync()
    {
        Console.WriteLine("=== SharpMind Interactive Chat ===");
        Console.WriteLine("Type 'quit' to exit");
        Console.WriteLine();

        var hardware = DetectBestHardware();
        Console.WriteLine($"Hardware: {hardware}");
        Console.WriteLine();

        var baseDir = "C:\\Integral2u\\source\\repos\\SharpMind\\ExternalAssets";
        var ggufPath = Path.Combine(baseDir, GgufFileName);

        int vocabSize = 128256;
        int hiddenDim = 1536;
        int numLayers = 24;
        int numHeads = 12;
        int numKvHeads = 12;
        int ffnDim = 6144;
        int maxSeqLen = 2048;

        GgufLoader.GgufMeta meta = null!;

        if (!File.Exists(ggufPath))
        {
            Console.WriteLine($"GGUF not found: {ggufPath}");
            Console.WriteLine();
            return Task.CompletedTask;
        }

        var fileInfo = new FileInfo(ggufPath);
        Console.WriteLine($"GGUF: {ggufPath}");
        Console.WriteLine($"  Size: {fileInfo.Length / 1_000_000.0:F1} MB");

        try
        {
            meta = GgufLoader.LoadMeta(ggufPath);
            Console.WriteLine($"  Loaded metadata: {meta.TensorCount} tensors");

            // Find embedding layer to get vocab size/hidden dim
            var embdInfo = meta.Tensors.FirstOrDefault(t => t.Name.Contains("token_embd") && t.Name.Contains("weight"));
            if (embdInfo.Shape != null && embdInfo.Shape.Length >= 2)
            {
                // GGUF stores weight as [hidden_dim, vocab_size] or [vocab_size, hidden_dim]
                var dim0 = embdInfo.Shape[0];
                var dim1 = embdInfo.Shape[1];
                if (dim0 > dim1) { vocabSize = dim0; hiddenDim = dim1; }
                else { vocabSize = dim1; hiddenDim = dim0; }
                Console.WriteLine($"  Detected from tensor: vocab={vocabSize}, hidden={hiddenDim}");
            }
            else
            {
                hiddenDim = (int)meta.GetLong("llama.embedding_length", 1536);
                vocabSize = (int)meta.GetLong("vocab_size", 32000); // Fallback
            }

            numLayers = (int)meta.GetLong("llama.block_count", 24);
            numHeads = (int)meta.GetLong("llama.attention.head_count", 12);
            numKvHeads = (int)meta.GetLong("llama.attention.head_count_kv", 12);
            ffnDim = (int)meta.GetLong("llama.feed_forward_length", 6144);
            maxSeqLen = (int)meta.GetLong("llama.context_length", 2048);

            Console.WriteLine($"  Config: vocab={vocabSize}, hidden={hiddenDim}, layers={numLayers}");
            Console.WriteLine($"  Heads: {numHeads}, kv={numKvHeads}, ffn={ffnDim}");

            Console.WriteLine();
            Console.WriteLine("Loading weights...");
        }

        catch (Exception ex)
        {
            Console.WriteLine($"  Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }

        // Fixup: Ensure HiddenDim is divisible by NumHeads
        if (hiddenDim % numHeads != 0)
        {
            Console.WriteLine($"[Warning] HiddenDim ({hiddenDim}) not divisible by NumHeads ({numHeads}). Adjusting NumHeads...");
            for (int h = numHeads; h > 0; h--)
            {
                if (hiddenDim % h == 0)
                {
                    numHeads = h;
                    break;
                }
            }
            Console.WriteLine($"[Warning] Adjusted NumHeads to: {numHeads}");
        }

        // Fixup: Ensure NumHeads is divisible by NumKvHeads
        if (numHeads % numKvHeads != 0)
        {
            Console.WriteLine($"[Warning] NumHeads ({numHeads}) not divisible by NumKvHeads ({numKvHeads}). Adjusting NumKvHeads...");
            for (int kv = numKvHeads; kv > 0; kv--)
            {
                if (numHeads % kv == 0)
                {
                    numKvHeads = kv;
                    break;
                }
            }
            Console.WriteLine($"[Warning] Adjusted NumKvHeads to: {numKvHeads}");
        }


        Console.WriteLine();

        var modelConfig = new ModelConfig
        {
            VocabSize = vocabSize,
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            NumHeads = numHeads,
            NumKvHeads = numKvHeads,
            FfnDim = ffnDim,
            MaxSeqLen = maxSeqLen,
        };

        var sharpConfig = new SharpMindConfig
        {
            Activation = ActivationKind.SiLU,
            Gate = GateKind.SwiGLU,
            Ffn = FfnKind.Gated,
            Attention = AttentionKind.MQA,
            Norm = NormKind.RMSNorm,
            Arch = ArchKind.Decoder,
            Hardware = hardware,
        };

        Console.WriteLine("Building model...");
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model: {model.ParameterCount / 1_000_000.0:F1}M params");

        Console.WriteLine("Loading weights into model...");
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
        sw.Stop();
        Console.WriteLine($"Weights loaded in {sw.ElapsedMilliseconds}ms!");

        string systemPrompt = File.Exists("System.md") ? File.ReadAllText("System.md") : "";
        string agentPrompt = File.Exists("Agent.md") ? File.ReadAllText("Agent.md") : "";

        var session = new SimpleChatSession(model, vocabSize, systemPrompt, agentPrompt);

        Console.WriteLine();
        Console.WriteLine("Chat ready! Say hello.");
        Console.WriteLine();

        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input)) continue;
            if (input.ToLower() == "quit") break;

            var response = session.Chat(input);
            Console.WriteLine($"Bot: {response}");
            Console.WriteLine();
        }

        Console.WriteLine("Chat ended.");
        return Task.CompletedTask;
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported) return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }
}

public sealed class SimpleChatSession
{
    private readonly Transformer _model;
    private readonly int _vocabSize;
    private readonly List<int> _context = new();
    private readonly string _systemPrompt;
    private readonly string _agentPrompt;

    private static readonly Dictionary<string, int> CharToId = new();
    private static readonly List<string> IdToChar = new();

    static SimpleChatSession()
    {
        for (int i = 0; i < 128; i++)
        {
            var c = (char)i;
            IdToChar.Add(c.ToString());
            CharToId[c.ToString()] = i;
        }
    }

    public SimpleChatSession(Transformer model, int vocabSize, string systemPrompt, string agentPrompt)
    {
        _model = model;
        _vocabSize = vocabSize;
        _systemPrompt = systemPrompt;
        _agentPrompt = agentPrompt;
    }

    public string Chat(string input, int maxTokens = 64)
    {
        var inputIds = Encode(input);
        _context.AddRange(inputIds);

        if (_context.Count > _model.Config.MaxSeqLen - maxTokens)
            _context.RemoveRange(0, _context.Count - (_model.Config.MaxSeqLen - maxTokens));

        var generated = Generate(maxTokens);

        var newTokens = generated.Skip(_context.Count - inputIds.Length).ToList();
        _context.AddRange(newTokens);

        return Decode(newTokens);
    }

    private int[] Encode(string text)
    {
        var result = new List<int> { 1 };
        foreach (var c in text)
        {
            var s = c.ToString();
            result.Add(CharToId.TryGetValue(s, out var id) ? id : 0);
        }
        result.Add(2);
        return result.ToArray();
    }

    private string Decode(List<int> tokens)
    {
        var result = new System.Text.StringBuilder();
        foreach (var t in tokens.Take(200))
        {
            if (t < IdToChar.Count)
                result.Append(IdToChar[t][0]);
        }
        return result.ToString();
    }

    private List<int> Generate(int maxTokens)
    {
        var result = new List<int>(_context);

        for (int i = 0; i < maxTokens; i++)
        {
            using var input = Tensor<int>.From(result.ToArray(), 1, result.Count);
            using var output = _model.Forward(input);

            var logits = output.Data;
            var offset = (result.Count - 1) * _vocabSize;
            var lastLogits = logits.Slice(offset, _vocabSize);

            var nextToken = SampleGreedy(lastLogits);

            if (nextToken <= 3) break;

            result.Add(nextToken);

            if (result.Count >= _model.Config.MaxSeqLen) break;

            if (nextToken == 2 || nextToken == 3) break;
        }

        return result;
    }

    private int SampleGreedy(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float maxLog = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLog)
            {
                maxLog = logits[i];
                best = i;
            }
        }
        return best;
    }
}