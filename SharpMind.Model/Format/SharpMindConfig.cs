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
    
    // ── Model dimensions (mirrors ModelConfig) ─────────────────────────────
    public int VocabSize { get; set; }
    public int HiddenDim { get; set; }
    public int NumLayers { get; set; }
    public int NumHeads { get; set; }
    public int NumKvHeads { get; set; }
    public int FfnDim { get; set; }
    public int MaxSeqLen { get; set; }
    public float RopeTheta { get; set; } = 10000f;
    
    // ── Activation settings ─────────────────────────────────────────────
    public string Activation { get; set; } = "silu";
    public string Gate { get; set; } = "swiglu";
    public string Ffn { get; set; } = "gated";
    public string Norm { get; set; } = "rmsnorm";
    public string Attention { get; set; } = "gqa";
    
    // ── Quantization settings ────────────────────────────────────────────
    public QuantConfig? Quantization { get; set; }
    
    // ── Metadata ─────────────────────────────────────────────────────
    public string? Source { get; set; }
    public string? Checksum { get; set; }
    public TokenizerInfo? Tokenizer { get; set; }

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

    public global::SharpMind.SharpMindConfig ToJigSawConfig()
    {
        return new global::SharpMind.SharpMindConfig
        {
            Activation = Activation?.ToLowerInvariant() switch
            {
                "silu" => global::SharpMind.ActivationKind.SiLU,
                "gelu" => global::SharpMind.ActivationKind.GELU,
                "relu" => global::SharpMind.ActivationKind.ReLU,
                _ => global::SharpMind.ActivationKind.SiLU
            },
            Gate = Gate?.ToLowerInvariant() switch
            {
                "swiglu" => global::SharpMind.GateKind.SwiGLU,
                "geglu" => global::SharpMind.GateKind.GeGLU,
                _ => global::SharpMind.GateKind.None
            },
            Ffn = Ffn?.ToLowerInvariant() switch
            {
                "gated" => global::SharpMind.FfnKind.Gated,
                "moe" => global::SharpMind.FfnKind.MoE,
                _ => global::SharpMind.FfnKind.Dense
            },
            Attention = Attention?.ToLowerInvariant() switch
            {
                "gqa" => global::SharpMind.AttentionKind.GQA,
                "mqa" => global::SharpMind.AttentionKind.MQA,
                _ => global::SharpMind.AttentionKind.MHA
            },
            Norm = Norm?.ToLowerInvariant() switch
            {
                "rmsnorm" => global::SharpMind.NormKind.RMSNorm,
                "layernorm" => global::SharpMind.NormKind.LayerNorm,
                _ => global::SharpMind.NormKind.RMSNorm
            },
            Arch = Architecture?.ToLowerInvariant() switch
            {
                "encoder" => global::SharpMind.ArchKind.Encoder,
                _ => global::SharpMind.ArchKind.Decoder
            },
        };
    }

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, JsonSerializerOptionsSavePolicy));

    public static SharpMindConfig Load(string path) => JsonSerializer.Deserialize<SharpMindConfig>(File.ReadAllText(path), JsonSerializerOptionsLoadPolicy)
            ?? throw new InvalidDataException("Failed to deserialize config.json");
}

public sealed class TokenizerInfo
{
    public string Type { get; set; } = "bpe";
    public string? VocabFile { get; set; }
    public Dictionary<string, string>? SpecialTokens { get; set; }
}