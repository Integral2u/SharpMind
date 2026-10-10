using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

/// <summary>
/// Geometry of a fused MoE expert stack. qwen2moe packs all experts of one FFN projection
/// into a single 3D tensor with no per-expert index in the name, e.g.
/// <c>blk.N.ffn_gate_exps.weight</c> with GGUF shape <c>[in, out, num_experts]</c>. GGUF lists
/// dims fastest-varying first, so the expert axis is the LAST dim and expert <c>e</c> owns the
/// contiguous byte range <c>[e*plane, (e+1)*plane)</c> — exactly the <c>[in, out]</c> chunk each
/// <c>InferenceLinearLayer</c> size guard expects.
///
/// This decision lives in one place because it is consumed twice: <see cref="GgufLoader"/>
/// slices the file bytes for each expert, and <see cref="TransformerWeightsStreaming"/>
/// registers per-expert <c>TensorMeta</c> before those bytes are ever read (the FfnLayer is
/// built from that metadata, so a wrong plane size silently falls back to F32 and then rejects
/// the quantized bytes at load time). Keeping the two in sync is the point of the helper.
/// </summary>
internal readonly record struct FusedExpertLayout(string FieldPrefix, int NumExperts, int PlaneSizeBytes)
{
    /// <summary>True when <paramref name="rawField"/> is one of the three fused expert fields
    /// <see cref="TransformerWeights.ResolveTarget"/> returns for an unsplit expert stack.</summary>
    public static bool IsFusedField(string rawField) =>
        rawField.StartsWith("RawWgateExpFused", StringComparison.Ordinal) ||
        rawField.StartsWith("RawWupExpFused", StringComparison.Ordinal) ||
        rawField.StartsWith("RawWdownExpFused", StringComparison.Ordinal);

    /// <summary>Per-expert field prefix (<c>RawWgateExp_</c>, …) for the given fused field.</summary>
    public static string PrefixFor(string rawField) => rawField switch
    {
        var f when f.StartsWith("RawWgateExpFused", StringComparison.Ordinal) => "RawWgateExp_",
        var f when f.StartsWith("RawWupExpFused", StringComparison.Ordinal) => "RawWupExp_",
        _ => "RawWdownExp_",
    };

    /// <summary>
    /// Computes the layout for a fused expert tensor. The plane size is derived from the 2D
    /// sub-shape rather than by dividing the total byte count: <see cref="QuantizationOps.GetRawTensorByteCount"/>
    /// lays blocks along the LAST dim, which for the 3D stack is the expert axis and over-counts
    /// (<c>ceil(experts/32)</c> blocks per row) unless the expert count is block-aligned. The 2D
    /// call is correct because <c>shape[0]</c> and <c>shape[1]</c> are both block-aligned.
    /// </summary>
    public static FusedExpertLayout Create(string rawField, int[] shape, QuantDType dtype)
    {
        if (shape.Length != 3)
            throw new InvalidDataException(
                $"Fused MoE expert field '{rawField}' has rank {shape.Length}; expected a 3D [in, out, experts] tensor.");

        int numExperts = shape[2];
        int planeSize = TensorLoadHelper.CheckedInt(
            QuantizationOps.GetRawTensorByteCount([shape[0], shape[1]], dtype),
            "fused MoE expert plane size");

        return new FusedExpertLayout(PrefixFor(rawField), numExperts, planeSize);
    }

    /// <summary>Byte offset of expert <paramref name="expertIndex"/> within the 3D tensor.</summary>
    public long OffsetOf(int expertIndex) => (long)expertIndex * PlaneSizeBytes;
}
