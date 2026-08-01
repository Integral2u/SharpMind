namespace SharpMind.Core;

public class MappingBuilder(HardwareTier hardware = HardwareTier.Auto)
{
    private readonly Dictionary<string, string> _mapping = [];
    private readonly HardwareTier _hardware = hardware;

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
        _mapping[SharpMindConfig.KeyAttention] = string.IsNullOrEmpty(hw)
            ? $"{config.Attention.ToString().ToLowerInvariant()}{flash}scalar"
            : $"{config.Attention.ToString().ToLowerInvariant()}{flash}{hw}";
        _mapping[SharpMindConfig.KeyAttentionQ8] = string.IsNullOrEmpty(hw)
            ? $"{config.Attention.ToString().ToLowerInvariant()}flashq8_0scalar"
            : $"{config.Attention.ToString().ToLowerInvariant()}flashq8_0{hw}";
        _mapping[SharpMindConfig.KeyAttentionQ4] = string.IsNullOrEmpty(hw)
            ? $"{config.Attention.ToString().ToLowerInvariant()}flashq4_0scalar"
            : $"{config.Attention.ToString().ToLowerInvariant()}flashq4_0{hw}";
        _mapping[SharpMindConfig.KeyFfn] = config.Ffn.ToString().ToLowerInvariant();
        _mapping[SharpMindConfig.KeyLinear] = SharpMindConfig.ValLinearAuto;
        _mapping[SharpMindConfig.KeyLogit] = SharpMindConfig.ValLinearAuto;

        string trainTier = _hardware switch
        {
            HardwareTier.Scalar => SharpMindConfig.ValScalar,
            HardwareTier.FMA    => SharpMindConfig.ValFma,
            _                   => SharpMindConfig.ValAvx2
        };
        string gradLinearTier = _hardware switch
        {
            HardwareTier.Scalar => SharpMindConfig.ValScalar,
            HardwareTier.FMA    => SharpMindConfig.ValFma,
            _                   => SharpMindConfig.ValAvx2
        };
        string gradActTier = _hardware == HardwareTier.Scalar
            ? SharpMindConfig.ValScalar
            : SharpMindConfig.ValAvx2;

        _mapping[SharpMindConfig.KeyAdamW] = trainTier;
        _mapping[SharpMindConfig.KeyGradNorm] = trainTier;
        _mapping[SharpMindConfig.KeyLayerNormRow] = gradActTier;

        // Gradient backward kernels. Only linear currently has dedicated AVX2/FMA
        // tiers; the rest resolve to the scalar kernel for every tier.
        _mapping[SharpMindConfig.KeyGradLinear] = gradLinearTier;
        _mapping[SharpMindConfig.KeyGradRMSNorm] = SharpMindConfig.ValScalar;
        _mapping[SharpMindConfig.KeyGradLayerNorm] = SharpMindConfig.ValScalar;
        _mapping[SharpMindConfig.KeyGradAttention] = SharpMindConfig.ValScalar;
        _mapping[SharpMindConfig.KeyGradEmbedding] = SharpMindConfig.ValScalar;
        _mapping[SharpMindConfig.KeyGradActivationSiLU] = gradActTier;
        _mapping[SharpMindConfig.KeyGradActivationGELU] = gradActTier;

        return this;
    }

    public MappingBuilder ApplyQuantPreset(SharpMindConfig config) => ApplyQuantPreset(config.Parallel);

    public MappingBuilder ApplyQuantPreset(bool parallel = true)
    {
        string hwSuffix = QuantHwSuffix();
        string qmmSuffix = QuantQmmSuffix(parallel);

        _mapping["vecdot_q3k"]  = $"q3k{hwSuffix}";
        _mapping["vecdot_q4k"]  = $"q4k{hwSuffix}";
        _mapping["vecdot_q5k"]  = $"q5k{hwSuffix}";
        _mapping["vecdot_q6k"]  = $"q6k{hwSuffix}";
        _mapping["vecdot_q8_0"] = $"q8_0{hwSuffix}";
        _mapping["vecdot_q4_0"] = $"q4_0{hwSuffix}";
        _mapping["vecdot_q4_1"] = $"q4_1{hwSuffix}";
        _mapping["vecdot_q5_0"] = $"q5_0{hwSuffix}";
        _mapping["vecdot_q5_1"] = $"q5_1{hwSuffix}";
        _mapping["vecdot_q8_1"] = $"q8_1{hwSuffix}";
        _mapping["vecdot_q2k"]  = $"q2k{hwSuffix}";
        _mapping["vecdot_q8k"]  = $"q8k{hwSuffix}";
        _mapping["vecdot_q4_nl"]= $"q4_nl{hwSuffix}";
        _mapping["vecdot_f32"]  = $"f32{hwSuffix}";
        _mapping["vecdot_f16"]  = $"f16{hwSuffix}";
        _mapping["vecdot_i8"]   = $"i8{hwSuffix}";
        _mapping["vecdot_i16"]  = $"i16{hwSuffix}";
        _mapping["vecdot_i32"]  = $"i32{hwSuffix}";
        _mapping["vecdot_tq2_0"]= $"tq2_0{hwSuffix}";
        _mapping["vecdot_tq1_0"]= $"tq1_0{hwSuffix}";
        _mapping["vecdot_iq1_s"]= $"iq1_s{hwSuffix}";
        _mapping["vecdot_iq1_m"]= $"iq1_m{hwSuffix}";

        _mapping["qmatmul_q8_0"] = $"qmatmul_q8_0{qmmSuffix}";
        _mapping["qmatmul_q5_0"] = $"qmatmul_q5_0{qmmSuffix}";
        _mapping["qmatmul_q6k"]  = $"qmatmul_q6k{qmmSuffix}";
        _mapping["qmatmul_q4_0"] = $"qmatmul_q4_0{qmmSuffix}";
        _mapping["qmatmul_q4_1"] = $"qmatmul_q4_1{qmmSuffix}";
        _mapping["qmatmul_q2k"]  = $"qmatmul_q2k{qmmSuffix}";
        _mapping["qmatmul_q3k"]  = $"qmatmul_q3k{qmmSuffix}";
        _mapping["qmatmul_q4k"]  = $"qmatmul_q4k{qmmSuffix}";
        _mapping["qmatmul_q5k"]  = $"qmatmul_q5k{qmmSuffix}";
        _mapping["qmatmul_q8k"]  = $"qmatmul_q8k{qmmSuffix}";
        _mapping["qmatmul_q8_1"] = $"qmatmul_q8_1{qmmSuffix}";
        _mapping["qmatmul_q5_1"] = $"qmatmul_q5_1{qmmSuffix}";
        _mapping["qmatmul_q4_nl"]= $"qmatmul_q4_nl{qmmSuffix}";
        _mapping["qmatmul_f32"]  = $"qmatmul_f32{qmmSuffix}";
        _mapping["qmatmul_f16"]  = $"qmatmul_f16{qmmSuffix}";
        _mapping["qmatmul_i8"]   = $"qmatmul_i8{qmmSuffix}";
        _mapping["qmatmul_i16"]  = $"qmatmul_i16{qmmSuffix}";
        _mapping["qmatmul_i32"]  = $"qmatmul_i32{qmmSuffix}";
        _mapping["qmatmul_tq2_0"]= $"qmatmul_tq2_0{qmmSuffix}";
        _mapping["qmatmul_tq1_0"]= $"qmatmul_tq1_0{qmmSuffix}";
        _mapping["qmatmul_iq1_s"]= $"qmatmul_iq1_s{qmmSuffix}";
        _mapping["qmatmul_iq1_m"]= $"qmatmul_iq1_m{qmmSuffix}";

        _mapping["read_q8_0"] = "read_q8_0_scalar";
        _mapping["read_q4_0"] = "read_q4_0_scalar";
        _mapping["read_q4_1"] = "read_q4_1_scalar";
        _mapping["read_q5_0"] = "read_q5_0_scalar";
        _mapping["read_q5_1"] = "read_q5_1_scalar";
        _mapping["read_q8_1"] = "read_q8_1_scalar";
        _mapping["read_q4_nl"] = "read_q4_nl_scalar";
        _mapping["read_q2k"]  = "read_q2k_scalar";
        _mapping["read_q3k"]  = "read_q3k_scalar";
        _mapping["read_q4k"]  = "read_q4k_scalar";
        _mapping["read_q5k"]  = "read_q5k_scalar";
        _mapping["read_q6k"]  = "read_q6k_scalar";
        _mapping["read_q8k"]  = "read_q8k_scalar";
        _mapping["read_f32"]  = "read_f32_scalar";
        _mapping["read_f16"]  = "read_f16_scalar";
        _mapping["read_i8"]   = "read_i8_scalar";
        _mapping["read_i16"]  = "read_i16_scalar";
        _mapping["read_i32"]  = "read_i32_scalar";
        _mapping["read_tq2_0"]= "read_tq2_0_scalar";
        _mapping["read_tq1_0"]= "read_tq1_0_scalar";
        _mapping["read_iq1_s"]= "read_iq1_s_scalar";
        _mapping["read_iq1_m"]= "read_iq1_m_scalar";

        _mapping["hsum256"]             = $"hsum{hwSuffix}";
        _mapping["halftofloat"]         = $"halftofloat{hwSuffix}";
        _mapping["floattohalf"]         = $"floattohalf{hwSuffix}";
        _mapping["getscalemink4_scale"] = $"getscalemink4_scale{hwSuffix}";
        _mapping["getscalemink4_min"]   = $"getscalemink4_min{hwSuffix}";

        return this;
    }

    public MappingBuilder Override(string key, string value)
    {
        _mapping[key] = value;
        return this;
    }

    public bool TryGetValue(string key, out string? value) => _mapping.TryGetValue(key, out value);

    public Dictionary<string, string> Build() => new(_mapping);

    private string GetHwSuffix() => _hardware switch
    {
        HardwareTier.FMA => "fma",
        HardwareTier.AVX2 => "avx2",
        _ => ""
    };

    private string QuantHwSuffix() => _hardware switch
    {
        HardwareTier.FMA  => "_fma",
        HardwareTier.AVX2 => "_avx2",
        HardwareTier.SSE  => "_sse",
        _                 => "_scalar"
    };

    private string QuantQmmSuffix(bool parallel)
    {
        string mode = parallel ? "parallel" : "serial";
        string hw = _hardware switch
        {
            HardwareTier.FMA  => "fma",
            HardwareTier.AVX2 => "avx2",
            _                 => "scalar"
        };
        return $"_{mode}_{hw}";
    }

    public const string KeyGate = "gate";
}
