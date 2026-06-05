namespace SharpMind;

/// <summary>
/// Orchestrates the assembly of JigSawDotNet mappings.
/// Allows for base presets with hardware-specific overrides and manual adjustments.
/// </summary>
public class MappingBuilder(HardwareTier hardware = HardwareTier.Auto)
{
    private readonly Dictionary<string, string> _mapping = [];
    private readonly HardwareTier _hardware = hardware;

    /// <summary>
    /// Applies a base configuration's default mappings.
    /// </summary>
    public MappingBuilder ApplyPreset(SharpMindConfig config)
    {
        string hw = GetHwSuffix();
        string flash = config.FlashAttention ? "flash" : "";
        string act = config.Activation.ToString().ToLowerInvariant();
        string gate = config.Gate.ToString().ToLowerInvariant();

        _mapping[SharpMindConfig.KeyPointWise] = $"{act}{hw}";
        _mapping[KeyGate] = $"{gate}{hw}";
        _mapping[SharpMindConfig.KeySoftmax] = hw;
        _mapping[SharpMindConfig.KeyRMSNorm] = hw;
        _mapping[SharpMindConfig.KeyMatMul] = GetMatMulHwKey();
        _mapping[SharpMindConfig.KeyAttention] = string.IsNullOrEmpty(hw)
            ? $"{config.Attention.ToString().ToLowerInvariant()}{flash}scalar"
            : $"{config.Attention.ToString().ToLowerInvariant()}{flash}{hw}";
        _mapping[SharpMindConfig.KeyFfn] = config.Ffn.ToString().ToLowerInvariant();
        _mapping[SharpMindConfig.KeyNorm] = config.Norm == NormKind.RMSNorm ? SharpMindConfig.ValNormRMS : SharpMindConfig.ValNormLayer;
        _mapping[SharpMindConfig.KeyArch] = config.Arch == ArchKind.Decoder ? SharpMindConfig.ValDecoder : SharpMindConfig.ValEncoder;
        _mapping[SharpMindConfig.KeyAdamW] = _hardware == HardwareTier.Scalar ? SharpMindConfig.ValScalar : SharpMindConfig.ValAvx2;
        _mapping[SharpMindConfig.KeyGradNorm] = _hardware == HardwareTier.Scalar ? SharpMindConfig.ValScalar : SharpMindConfig.ValAvx2;

        return this;
    }

    /// <summary>
    /// Explicitly overrides a mapping slot.
    /// </summary>
    public MappingBuilder Override(string key, string value)
    {
        _mapping[key] = value;
        return this;
    }

    /// <summary>
    /// Reads the current value for a mapping key, if set.
    /// </summary>
    public bool TryGetValue(string key, out string? value) => _mapping.TryGetValue(key, out value);

    public Dictionary<string, string> Build() => new(_mapping);

    private string GetHwSuffix() => _hardware switch
    {
        HardwareTier.FMA => "fma",
        HardwareTier.AVX2 => "avx2",
        _ => ""
    };
    
    private string GetMatMulHwKey() => _hardware switch
    {
        HardwareTier.FMA => "fma",
        HardwareTier.AVX2 => "avx2",
        _ => "scalar" // MatMul typically needs an explicit scalar key
    };

    // Internal helper for keys to avoid repetition
    public const string KeyGate = "gate";
}