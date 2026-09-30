using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

/// <summary>
/// The authoritative bridge between canonical GGML/GGUF tensor type ids and
/// SharpMind's internal <see cref="QuantDType"/>.
///
/// The internal enum is a convenience index, not the GGUF wire format: several
/// members (I8=16, I16=17, I32=18, IQ1_M=21, TQ1_0=22, TQ2_0=23) deliberately
/// differ from the canonical <c>ggml_type</c> values, and BF16 was absent
/// entirely. Casting a raw GGUF id to <see cref="QuantDType"/> decoded real IQ2/
/// IQ3 tensors as raw integer types (silent garbage), and mis-mapped the rest.
/// All GGUF reads and writes must go through this map so the canonical ids are
/// stable and unsupported-but-real types fail loudly instead of mis-decoding.
/// </summary>
public static class GgufTypeMap
{
    /// <summary>Canonical <c>ggml_type</c> ids SharpMind can load, and their internal dtype.</summary>
    private static readonly Dictionary<uint, QuantDType> Supported = new()
    {
        [0] = QuantDType.F32,
        [1] = QuantDType.F16,
        [2] = QuantDType.Q4_0,
        [3] = QuantDType.Q4_1,
        [6] = QuantDType.Q5_0,
        [7] = QuantDType.Q5_1,
        [8] = QuantDType.Q8_0,
        [9] = QuantDType.Q8_1,
        [10] = QuantDType.Q2_K,
        [11] = QuantDType.Q3_K,
        [12] = QuantDType.Q4_K,
        [13] = QuantDType.Q5_K,
        [14] = QuantDType.Q6_K,
        [15] = QuantDType.Q8_K,
        [19] = QuantDType.IQ1_S,
        [20] = QuantDType.IQ4_NL,
        [24] = QuantDType.I8,
        [25] = QuantDType.I16,
        [26] = QuantDType.I32,
        [29] = QuantDType.IQ1_M,
        [30] = QuantDType.BF16,
        [34] = QuantDType.TQ1_0,
        [35] = QuantDType.TQ2_0,
        [41] = QuantDType.Q1_0,
    };

    private static readonly Dictionary<QuantDType, uint> ByDtype = Supported.ToDictionary(p => p.Value, p => p.Key);

    /// <summary>Canonical <c>ggml_type</c> names (for error messages), covering every id SharpMind may ever see.</summary>
    private static readonly Dictionary<uint, string> Names = new()
    {
        [0] = "F32", [1] = "F16", [2] = "Q4_0", [3] = "Q4_1",
        [6] = "Q5_0", [7] = "Q5_1", [8] = "Q8_0", [9] = "Q8_1",
        [10] = "Q2_K", [11] = "Q3_K", [12] = "Q4_K", [13] = "Q5_K", [14] = "Q6_K", [15] = "Q8_K",
        [16] = "IQ2_XXS", [17] = "IQ2_XS", [18] = "IQ3_XXS",
        [19] = "IQ1_S", [20] = "IQ4_NL",
        [21] = "IQ3_S", [22] = "IQ2_S", [23] = "IQ4_XS",
        [24] = "I8", [25] = "I16", [26] = "I32",
        [27] = "I64", [28] = "F64",
        [29] = "IQ1_M", [30] = "BF16",
        [31] = "Q4_0_4_4", [32] = "Q4_0_4_8", [33] = "Q4_0_8_8",
        [34] = "TQ1_0", [35] = "TQ2_0",
        [36] = "IQ4_NL_4_4", [37] = "IQ4_NL_4_8", [38] = "IQ4_NL_8_8",
        [39] = "MXFP4", [40] = "NVFP4", [41] = "Q1_0",
    };

    /// <summary>
    /// Maps a canonical GGUF type id to an internal <see cref="QuantDType"/>.
    /// Throws <see cref="NotSupportedException"/> for real GGML types SharpMind
    /// does not implement (e.g. IQ2_XXS, MXFP4) and for unknown ids, so a
    /// mismatched dtype can never silently produce garbage weights.
    /// </summary>
    public static QuantDType FromGgufId(uint id, string? tensorName = null)
    {
        if (Supported.TryGetValue(id, out var dtype))
            return dtype;

        string where = string.IsNullOrEmpty(tensorName) ? "" : $" for tensor '{tensorName}'";
        string what = Names.TryGetValue(id, out var name)
            ? $"GGML type {name} (id {id})"
            : $"unknown GGML type id {id}";
        throw new NotSupportedException($"{what}{where} is not supported by SharpMind. " +
            "Convert the model to a supported format (e.g. Q8_0, F16, Q4_K_M) and try again.");
    }

    /// <summary>
    /// Maps an internal <see cref="QuantDType"/> back to its canonical GGUF type
    /// id. SharpMind's S/M/L K-quant aliases (100-108) share their base type's
    /// block layout and are written as the base id.
    /// </summary>
    public static uint ToGgmlType(QuantDType dtype)
    {
        var canonical = dtype switch
        {
            QuantDType.Q2_K_S => QuantDType.Q2_K,
            QuantDType.Q3_K_S => QuantDType.Q3_K,
            QuantDType.Q3_K_M => QuantDType.Q3_K,
            QuantDType.Q3_K_L => QuantDType.Q3_K,
            QuantDType.Q4_K_S => QuantDType.Q4_K,
            QuantDType.Q4_K_M => QuantDType.Q4_K,
            QuantDType.Q5_K_S => QuantDType.Q5_K,
            QuantDType.Q5_K_M => QuantDType.Q5_K,
            QuantDType.Q6_K_S => QuantDType.Q6_K,
            _ => dtype
        };

        if (ByDtype.TryGetValue(canonical, out uint id)) return id;
        throw new ArgumentOutOfRangeException(nameof(dtype), dtype, $"No canonical GGML type id for {dtype}");
    }
}