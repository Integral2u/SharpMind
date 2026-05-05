using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Format;

/// <summary>
/// Model converter - converts between external formats and SharpMind native format.
/// </summary>
public static class ModelConverter
{
    private static readonly JsonSerializerOptions IndentedJsonSerializerOptions = new() { WriteIndented = true };
    /// <summary>Supported external model formats.</summary>
    public enum ModelFormat
    {
        Unknown,
        SafeTensors,  // HuggingFace
        Gguf,         // llama.cpp
        Pytorch,      // PyTorch checkpoint
    }
    
    /// <summary>Conversion result with parameters and config.</summary>
    public sealed class ConversionResult
    {
        public required List<Parameter> Parameters { get; init; }
        public required SharpMindConfig Config { get; init; }
        public string? Warning { get; init; }
    }
    
    /// <summary>Detect format from file extension.</summary>
    public static ModelFormat DetectFormat(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
            ".safetensors" => ModelFormat.SafeTensors,
            ".gguf" => ModelFormat.Gguf,
            ".bin" => ModelFormat.Gguf,
            ".pt" or ".pth" => ModelFormat.Pytorch,
            _ => ModelFormat.Unknown
        };
    }
    
    /// <summary>
    /// Load and convert a model from external format to SharpMind format.
    /// </summary>
    /// <param name="path">Path to model file or directory.</param>
    /// <param name="mapper">Weight name mapper (LlamaMapper, Gpt2Mapper, etc.).</param>
    public static ConversionResult Load(string path, WeightMapper mapper)
    {
        var format = DetectFormat(path);
        
        return format switch
        {
            ModelFormat.SafeTensors => LoadSafeTensors(path, mapper),
            ModelFormat.Gguf => LoadGguf(path, mapper),
            _ => throw new NotSupportedException($"Unknown format: {path}")
        };
    }
    
    /// <summary>Load from GGUF - auto-detects config from metadata.</summary>
    private static ConversionResult LoadGguf(string path, WeightMapper mapper)
    {
        var meta = GgufLoader.LoadMeta(path);
        var weights = GgufLoader.LoadWeights(path);
        
        var config = new SharpMindConfig
        {
            VocabSize = (int)meta.GetLong("token_embd.weight", 32000),
            HiddenDim = (int)meta.GetLong("embedding", 4096),
            NumLayers = (int)meta.GetLong("block_count", 32),
            NumHeads = (int)meta.GetLong("attention.head_count", 32),
            NumKvHeads = (int)meta.GetLong("attention.head_count_kv", 32),
            FfnDim = (int)meta.GetLong("ffn_dim", 11008),
            MaxSeqLen = (int)meta.GetLong("context_length", 2048),
            RopeTheta = meta.GetFloat("rope_theta", 10000f),
            Source = meta.GetString("arch", null),
        };
        
        return ConvertWeights(weights, mapper, config);
    }
    
    /// <summary>Load from Safetensors - requires config.json in same directory.</summary>
    private static ConversionResult LoadSafeTensors(string path, WeightMapper mapper)
    {
        var dir = Path.GetDirectoryName(path) ?? ".";
        var configPath = Path.Combine(dir, "config.json");
        
        var weights = SafetensorsLoader.LoadWeights(path);
        
        SharpMindConfig config;
        if (File.Exists(configPath))
        {
            var hfConfig = JsonNode.Parse(File.ReadAllText(configPath));
            
            config = new SharpMindConfig
            {
                VocabSize = hfConfig?["vocab_size"]?.GetValue<int>() ?? 32000,
                HiddenDim = hfConfig?["hidden_size"]?.GetValue<int>() ?? 4096,
                NumLayers = hfConfig?["num_hidden_layers"]?.GetValue<int>() ?? 32,
                NumHeads = hfConfig?["num_attention_heads"]?.GetValue<int>() ?? 32,
                NumKvHeads = hfConfig?["num_key_value_heads"]?.GetValue<int>() ?? 32,
                FfnDim = hfConfig?["intermediate_size"]?.GetValue<int>() ?? 11008,
                MaxSeqLen = hfConfig?["max_position_embeddings"]?.GetValue<int>() ?? 2048,
                RopeTheta = (float)(hfConfig?["rope_theta"]?.GetValue<double>() ?? 10000.0),
            };
        }
        else
        {
            config = InferConfig(weights);
        }
        
        return ConvertWeights(weights, mapper, config);
    }
    
    private static SharpMindConfig InferConfig(Dictionary<string, Tensor<float>> weights)
    {
        int maxLayer = 0;
        
        foreach (var name in weights.Keys)
        {
            if (name.Contains(".mlp.gate_proj.weight"))
            {
                var parts = name.Split('.');
                if (parts.Length >= 3 && parts[1] == "layers" && int.TryParse(parts[2], out int layer))
                    maxLayer = Math.Max(maxLayer, layer);
            }
        }
        
        return new SharpMindConfig
        {
            VocabSize = 32000,
            HiddenDim = 4096,
            NumLayers = maxLayer + 1,
            NumHeads = 32,
            NumKvHeads = 32,
            FfnDim = 11008,
            MaxSeqLen = 2048,
            Tokenizer = new TokenizerInfo { Type = "bpe" },
        };
    }
    
    private static ConversionResult ConvertWeights(
        Dictionary<string, Tensor<float>> weights,
        WeightMapper mapper,
        SharpMindConfig config)
    {
        var parameters = new List<Parameter>();
        var missing = new List<string>();
        
        foreach (var kvp in weights)
        {
            var smName = mapper.MapWeight(kvp.Key, [.. kvp.Value.Shape.Dims]);
            if (smName != null)
            {
                var param = new Parameter(smName, kvp.Value);
                parameters.Add(param);
            }
            else
            {
                missing.Add(kvp.Key);
            }
        }
        
        string? warning = missing.Count > 0 
            ? $"Skipped {missing.Count} weights: {string.Join(", ", missing.Take(5))}..." 
            : null;
        
        return new ConversionResult
        {
            Parameters = parameters,
            Config = config,
            Warning = warning
        };
    }
    
    /// <summary>
    /// Save model to SharpMind native format.
    /// Creates: model.sharpmind/weights.bin + config.json
    /// </summary>
    /// <param name="parameters">Model parameters.</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="outputDir">Output directory.</param>
    public static void SaveSharpMind(IEnumerable<Parameter> parameters, SharpMindConfig config, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        
        config.Save(Path.Combine(outputDir, "config.json"));
        
        string weightsPath = Path.Combine(outputDir, "weights.bin");
        SaveWeightsBinary(parameters, weightsPath);
        
        var manifest = new
        {
            version = "1.0",
            format = "sharpmind",
            paramCount = parameters.Sum(p => p.Data.ElementCount),
            checksum = ComputeChecksum(weightsPath),
        };
        
        string manifestPath = Path.Combine(outputDir, "manifest.json");
        File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, IndentedJsonSerializerOptions));
    }
    
    private static void SaveWeightsBinary(IEnumerable<Parameter> parameters, string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        
        var paramList = parameters.ToList();
        writer.Write(paramList.Count);
        
        foreach (var p in paramList)
        {
            writer.Write(p.Name);
            writer.Write(p.Data.Shape.Rank);
            foreach (int dim in p.Data.Shape.Dims)
                writer.Write(dim);
            
            var data = p.Data.Data;
            for (int i = 0; i < data.Length; i++)
                writer.Write(data[i]);
        }
    }
    
    private static string ComputeChecksum(string path)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(path);
        var hash = sha256.ComputeHash(stream);
        return Convert.ToHexString(hash);
    }
    
    /// <summary>
    /// Load model from SharpMind native format.
    /// </summary>
    public static ConversionResult LoadSharpMind(string modelDir)
    {
        var config = SharpMindConfig.Load(Path.Combine(modelDir, "config.json"));
        string weightsPath = Path.Combine(modelDir, "weights.bin");
        
        var parameters = LoadWeightsBinary(weightsPath);
        
        return new ConversionResult
        {
            Parameters = parameters,
            Config = config
        };
    }
    
    private static List<Parameter> LoadWeightsBinary(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        
        var result = new List<Parameter>();
        int count = reader.ReadInt32();
        
        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            int rank = reader.ReadInt32();
            var dims = new int[rank];
            for (int j = 0; j < rank; j++)
                dims[j] = reader.ReadInt32();
            
            var tensor = new Tensor<float>(dims);
            var data = tensor.Data;
            for (int j = 0; j < data.Length; j++)
                data[j] = reader.ReadSingle();
            
            result.Add(new Parameter(name, tensor));
        }
        
        return result;
    }
}