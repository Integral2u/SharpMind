using System.Security.Cryptography;
using System.Text.Json;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Format;
/// <summary>
/// Model converter - converts between external formats and SharpMind native format.
/// </summary>
public static partial class ModelConverter
{
    private static readonly JsonSerializerOptions IndentedJsonSerializerOptions = new() { WriteIndented = true };
    
    /// <summary>Detect format from file extension.</summary>
    public static ModelFormat DetectFormat(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext switch
        {
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
            ModelFormat.Gguf => LoadGguf(path, mapper),
            _ => throw new NotSupportedException($"Unknown format: {path}")
        };
    }
    
    /// <summary>Load from GGUF - auto-detects config from metadata.</summary>
    private static ConversionResult LoadGguf(string path, WeightMapper mapper)
    {
        var meta = GgufLoader.LoadMeta(path);
        var weights = GgufLoader.LoadWeights(path);
        string arch = meta.GetString("general.architecture", "llama");

        // Derive vocab size from embedding tensor shape (most reliable)
        int vocabSize = 32000;
        var embdInfo = meta.Tensors.FirstOrDefault(
            t => t.Name.Contains("token_embd") && t.Name.Contains("weight"));
        if (embdInfo.Shape is { Length: >= 2 })
        {
            long d0 = embdInfo.Shape[0], d1 = embdInfo.Shape[1];
            vocabSize = (int)(d0 > d1 ? d0 : d1);
        }
        // Override with explicit metadata keys
        vocabSize = (int)meta.GetLong($"{arch}.vocab_size",
                    meta.GetLong("tokenizer.ggml.token_count",
                    meta.GetLong("vocab_size", vocabSize)));

        var config = new SharpMindModelConfig
        {
            VocabSize = vocabSize,
            HiddenDim = (int)meta.GetLong($"{arch}.embedding_length", 4096),
            NumLayers = (int)meta.GetLong($"{arch}.block_count", 32),
            NumHeads = (int)meta.GetLong($"{arch}.attention.head_count", 32),
            NumKvHeads = (int)meta.GetLong($"{arch}.attention.head_count_kv", 32),
            FfnDim = (int)meta.GetLong($"{arch}.feed_forward_length", 11008),
            MaxSeqLen = (int)meta.GetLong($"{arch}.context_length", 2048),
            RopeTheta = meta.GetFloat($"{arch}.rope.freq_base",
                        meta.GetFloat("rope_theta", 10000f)),
            Source = $"{arch}/{meta.GetString("general.name", "model")}",
        };

        return ConvertWeights(weights, mapper, config);
    }   
    private static ConversionResult ConvertWeights(
        Dictionary<string, Tensor<float>> weights,
        WeightMapper mapper,
        SharpMindModelConfig config)
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
    public static void SaveSharpMind(IEnumerable<Parameter> parameters, SharpMindModelConfig config, string outputDir)
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
        return System.Convert.ToHexStringLower(hash);
    }
    
    /// <summary>
    /// Load model from SharpMind native format.
    /// </summary>
    public static ConversionResult LoadSharpMind(string modelDir)
    {
        var config = SharpMindModelConfig.Load(Path.Combine(modelDir, "config.json"));
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