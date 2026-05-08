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
        
        Dictionary<string, Tensor<float>> weights = new();
        
        if (File.Exists(ggufPath))
        {
            var fileInfo = new FileInfo(ggufPath);
            Console.WriteLine($"GGUF: {ggufPath}");
            Console.WriteLine($"  Size: {fileInfo.Length / 1_000_000.0:F1} MB");
            
            try
            {
                var meta = GgufLoader.LoadMeta(ggufPath);
                Console.WriteLine($"  Loaded metadata: {meta.TensorCount} tensors");
                
                hiddenDim = (int)meta.GetLong("llama.embedding_length", 1536);
                numLayers = (int)meta.GetLong("llama.block_count", 24);
                numHeads = (int)meta.GetLong("llama.attention.head_count", 12);
                numKvHeads = (int)meta.GetLong("llama.attention.head_count_kv", 12);
                ffnDim = (int)meta.GetLong("llama.feed_forward_length", 6144);
                maxSeqLen = (int)meta.GetLong("llama.context_length", 2048);
                
                // Need to scan tensors to find vocab size from embedding shape
                weights = GgufLoader.LoadWeights(ggufPath);
                Console.WriteLine($"  Loaded {weights.Count} tensors");
                
                // Find embedding layer to get vocab size - shape may be [vocab, hidden] or [hidden, vocab]
                var embd = weights.FirstOrDefault(t => t.Key.Contains("token_embd") && t.Key.Contains("weight"));
                if (embd.Value is not null)
                {
                    var shape = embd.Value.Shape;
                    var total = embd.Value.ElementCount;
                    if (shape.Length >= 2)
                    {
                        var dim0 = shape[0];
                        var dim1 = shape[1];
                        var hiddenFromMeta = (int)meta.GetLong("llama.embedding_length", 0);
                        
                        // dim0 * dim1 should equal total
                        if (dim0 * dim1 == total)
                        {
                            // If hidden from metadata matches one of dims, use the other as vocab
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
                                // Fallback: larger dim is vocab (vocab >> hidden for most models)
                                vocabSize = Math.Max(dim0, dim1);
                                hiddenDim = Math.Min(dim0, dim1);
                            }
                            Console.WriteLine($"  Detected: vocab={vocabSize}, hidden={hiddenDim}, elements={total}");
                        }
                    }
                }
                
                Console.WriteLine($"  Config: vocab={vocabSize}, hidden={hiddenDim}, layers={numLayers}");
                
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
        
        if (weights.Count > 0)
        {
            Console.WriteLine("Loading weights into model...");
            var sw = System.Diagnostics.Stopwatch.StartNew();
            LoadWeightsToModel(model, weights);
            sw.Stop();
            Console.WriteLine($"Weights loaded in {sw.ElapsedMilliseconds}ms!");
        }
        else
        {
            Console.WriteLine("WARNING: No weights loaded - using random initialization");
        }
        
        var session = new SimpleChatSession(model, vocabSize);
        
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

        foreach (var w in weights.Values)
            w.Dispose();
        
        Console.WriteLine("Chat ended.");
        return Task.CompletedTask;
    }

    private static void LoadWeightsToModel(Transformer model, Dictionary<string, Tensor<float>> weights)
    {
        var loaded = 0;
        var missing = 0;
        var sampleMissing = new List<string>();
        var sampleLoaded = new List<string>();
        
        foreach (var kvp in weights)
        {
            var name = kvp.Key;
            var tensor = kvp.Value;
            
            try
            {
                model.LoadWeight(name, tensor.Data);
                loaded++;
                if (loaded <= 3)
                    sampleLoaded.Add(name);
            }
            catch
            {
                missing++;
                if (missing <= 5)
                    sampleMissing.Add(name);
            }
        }
        
        Console.WriteLine($"  Loaded {loaded} weights, {missing} not matched");
        if (sampleLoaded.Count > 0)
            Console.WriteLine($"  Sample loaded: {string.Join(", ", sampleLoaded)}");
        if (sampleMissing.Count > 0)
            Console.WriteLine($"  Sample missing: {string.Join(", ", sampleMissing)}");
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
    
    public SimpleChatSession(Transformer model, int vocabSize)
    {
        _model = model;
        _vocabSize = vocabSize;
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