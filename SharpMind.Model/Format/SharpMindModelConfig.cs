using System.Text.Json;
using System.Text.Json.Serialization;
using SharpMind.Model.Config;

namespace SharpMind.Model.Format;

/// <summary>
/// SharpMind native model format configuration.
/// Stored as config.json in model.sharpmind/ directory.
/// </summary>
public sealed class SharpMindModelConfig
{
    /// <summary>Format version for compatibility.</summary>
    public string Version { get; set; } = "1.0";
    
    /// <summary>Architecture type: decoder, encoder, encoder-decoder.</summary>
    public string Architecture { get; set; } = "decoder";
    
    // Model dimensions (mirrors ModelConfig)
    public int VocabSize { get; set; }
    public int HiddenDim { get; set; }
    public int NumLayers { get; set; }
    public int NumHeads { get; set; }
    public int NumKvHeads { get; set; }
    public int FfnDim { get; set; }
    public int MaxSeqLen { get; set; }
    public float RopeTheta { get; set; } = 10000f;
    
    // Activation settings
    public string Activation { get; set; } = "silu";
    public string Gate { get; set; } = "swiglu";
    public string Ffn { get; set; } = "gated";
    public string Norm { get; set; } = "rmsnorm";
    public string Attention { get; set; } = "gqa";
    
    // Quantization settings
    public QuantConfig? Quantization { get; set; }
    
    // Metadata
    public string? Source { get; set; }
    public string? Checksum { get; set; }
    private static readonly JsonSerializerOptions JsonSerializerOptionsSavePolicy = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };
    private static readonly JsonSerializerOptions JsonSerializerOptionsLoadPolicy = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    public static SharpMindModelConfig FromModelConfig(ModelConfig config, string? source = null)
    {
        return new SharpMindModelConfig
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

    public SharpMindConfig ToJigSawConfig()
    {
        return new SharpMindConfig
        {
            Activation = Activation?.ToLowerInvariant() switch
            {
                "silu" => ActivationKind.SiLU,
                "gelu" => ActivationKind.GELU,
                "relu" => ActivationKind.ReLU,
                _ => ActivationKind.SiLU
            },
            Gate = Gate?.ToLowerInvariant() switch
            {
                "swiglu" => GateKind.SwiGLU,
                "geglu" => GateKind.GeGLU,
                _ => GateKind.None
            },
            Ffn = Ffn?.ToLowerInvariant() switch
            {
                "gated" => FfnKind.Gated,
                "moe" => FfnKind.MoE,
                _ => FfnKind.Dense
            },
            Attention = Attention?.ToLowerInvariant() switch
            {
                "gqa" => AttentionKind.GQA,
                "mqa" => AttentionKind.MQA,
                _ => AttentionKind.MHA
            },
            Norm = Norm?.ToLowerInvariant() switch
            {
                "rmsnorm" => NormKind.RMSNorm,
                "layernorm" => NormKind.LayerNorm,
                _ => NormKind.RMSNorm
            },
            Arch = Architecture?.ToLowerInvariant() switch
            {
                "encoder" => ArchKind.Encoder,
                _ => ArchKind.Decoder
            },
        };
    }

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonSerializerOptionsSavePolicy));

    public static SharpMindModelConfig Load(string path) => JsonSerializer.Deserialize<SharpMindModelConfig>(File.ReadAllText(path), JsonSerializerOptionsLoadPolicy)
            ?? throw new InvalidDataException("Failed to deserialize config.json");
}
