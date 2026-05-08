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
    private const string GgufFileName = "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf";
    
    public static Task RunAsync()
    {
        Console.WriteLine("=== SharpMind Interactive Chat ===");
        Console.WriteLine("Type 'quit' to exit");
        Console.WriteLine();

        var hardware = DetectBestHardware();
        Console.WriteLine($"Hardware: {hardware}");
        Console.WriteLine();

        var baseDir = "ExternalAssets";
        var ggufPath = Path.Combine(baseDir, GgufFileName);
        
        int vocabSize = 128256;
        int hiddenDim = 1536;
        int numLayers = 24;
        int numHeads = 12;
        int numKvHeads = 12;
        int ffnDim = 6144;
        int maxSeqLen = 2048;
        
        GgufLoader.GgufMeta meta = null!;
        
        if (File.Exists(ggufPath))
        {
            var fileInfo = new FileInfo(ggufPath);
            Console.WriteLine($"GGUF: {ggufPath}");
            Console.WriteLine($"  Size: {fileInfo.Length / 1_000_000.0:F1} MB");
            
            try
            {
                meta = GgufLoader.LoadMeta(ggufPath);
                Console.WriteLine($"  Loaded metadata: {meta.TensorCount} tensors");
                
                hiddenDim = (int)meta.GetLong("llama.embedding_length", 1536);
                numLayers = (int)meta.GetLong("llama.block_count", 24);
                numHeads = (int)meta.GetLong("llama.attention.head_count", 12);
                numKvHeads = (int)meta.GetLong("llama.attention.head_count_kv", 12);
                ffnDim = (int)meta.GetLong("llama.feed_forward_length", 6144);
                maxSeqLen = (int)meta.GetLong("llama.context_length", 2048);
                
                // Find embedding layer to get vocab size - shape may be [vocab, hidden] or [hidden, vocab]
                var embdInfo = meta.Tensors.FirstOrDefault(t => t.Name.Contains("token_embd") && t.Name.Contains("weight"));
                if (!string.IsNullOrEmpty(embdInfo.Name))
                {
                    var shape = embdInfo.Shape;
                    if (shape != null && shape.Length >= 2)
                    {
                        var dim0 = shape[0];
                        var dim1 = shape[1];
                        var hiddenFromMeta = (int)meta.GetLong("llama.embedding_length", 0);
                        
                        if (dim0 * dim1 == (long)dim0 * dim1) // Just a sanity check
                        {
                            if (dim0 == hiddenFromMeta)
                            {
                                vocabSize = dim1;
                                hiddenDim = dim0;
                            }
                            else if (dim1 == hiddenFromMeta)
                            {
                                vocabSize = dim0;
                                hiddenDim = dim1;
                            }
                            else
                            {
                                vocabSize = Math.Max(dim0, dim1);
                                hiddenDim = Math.Min(dim0, dim1);
                            }
                            Console.WriteLine($"  Detected: vocab={vocabSize}, hidden={hiddenDim}");
                        }
                    }
                }
                
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
        }
        else
        {
            Console.WriteLine($"GGUF not found: {ggufPath}");
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
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model: {model.ParameterCount / 1_000_000.0:F1}M params");
        
        Console.WriteLine("Loading weights into model...");
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