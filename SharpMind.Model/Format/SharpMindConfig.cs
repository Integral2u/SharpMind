using System.Text.Json;
using System.Text.Json.Serialization;
using SharpMind.Model.Config;

namespace SharpMind.Model.Format;

/// <summary>
/// SharpMind native model format configuration.
/// Stored as config.json in model.sharpmind/ directory.
/// </summary>
public sealed class SharpMindConfig
{
    /// <summary>Format version for compatibility.</summary>
    public string Version { get; set; } = "1.0";
    
    /// <summary>Architecture type: decoder, encoder, encoder-decoder.</summary>
    public string Architecture { get; set; } = "decoder";
    
    // ── Model dimensions (mirrors ModelConfig) ───────────────────────────────
    
    public int VocabSize { get; set; }
    public int HiddenDim { get; set; }
    public int NumLayers { get; set; }
    public int NumHeads { get; set; }
    public int NumKvHeads { get; set; }
    public int FfnDim { get; set; }
    public int MaxSeqLen { get; set; }
    public float RopeTheta { get; set; } = 10000f;
    
    // ── Quantization settings ────────────────────────────────────────────
    
    public QuantConfig? Quantization { get; set; }
    
    // ── Metadata ──────────────────────────────────────────────────────
    
    /// <summary>Original model source (e.g. "llama-3.1-8b", "gpt2").</summary>
    public string? Source { get; set; }
    
    /// <summary>SHA256 hash of weights.bin for integrity.</summary>
    public string? Checksum { get; set; }
    
    /// <summary>Tokenizer type and vocab file.</summary>
    public TokenizerInfo? Tokenizer { get; set; }
    
    /// <summary>Convert from ModelConfig.</summary>
    public static SharpMindConfig FromModelConfig(ModelConfig config, string? source = null)
    {
        return new SharpMindConfig
        {
            VocabSize = config.VocabSize,
            HiddenDim = config.HiddenDim,
            NumLayers = config.NumLayers,
            NumHeads = config.NumHeads,
            NumKvHeads = config.NumKvHeads,
            FfnDim = config.FfnDim,
            MaxSeqLen = config.MaxSeqLen,
            RopeTheta = config.RopeTheta,
            Source = source,
        };
    }
    
    /// <summary>Convert to ModelConfig for model creation.</summary>
    public ModelConfig ToModelConfig()
    {
        return new ModelConfig
        {
            VocabSize = VocabSize,
            HiddenDim = HiddenDim,
            NumLayers = NumLayers,
            NumHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            FfnDim = FfnDim,
            MaxSeqLen = MaxSeqLen,
            RopeTheta = RopeTheta,
        };
    }
    
    /// <summary>Save config to JSON file.</summary>
    public void Save(string path)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        };
        string json = JsonSerializer.Serialize(this, options);
        File.WriteAllText(path, json);
    }
    
    /// <summary>Load config from JSON file.</summary>
    public static SharpMindConfig Load(string path)
    {
        string json = File.ReadAllText(path);
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        };
        return JsonSerializer.Deserialize<SharpMindConfig>(json, options) 
            ?? throw new InvalidDataException("Failed to deserialize config.json");
    }
}

/// <summary>
/// Tokenizer information in SharpMind format.
/// </summary>
public sealed class TokenizerInfo
{
    /// <summary>Tokenizer type: bpe, unigram, wordpiece.</summary>
    public string Type { get; set; } = "bpe";
    
    /// <summary>Path to vocab file (relative to model dir).</summary>
    public string? VocabFile { get; set; }
    
    /// <summary>Special tokens map.</summary>
    public Dictionary<string, string>? SpecialTokens { get; set; }
}