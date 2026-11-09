using JigSawDotNet;
using SharpMind.Core;

namespace SharpMind.Model.Layers;

public abstract class NormOps
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(NormKernels)}";

    /// <summary>
    /// Auto-tier kernels for a <see cref="LayerNormLayer"/> constructed without explicit ops.
    /// Never reassigned: a model's layers carry the ops resolved from that model's own
    /// <see cref="SharpMindConfig.Hardware"/>, so building one model cannot change another's kernels.
    /// </summary>
    public static NormOps Default { get; } = NormOpsFactory.Create(new SharpMindConfig());

    [PuzzleCornerPiece(SharpMindConfig.KeyLayerNormRow,
        SharpMindConfig.ValAvx2, NS + "." + nameof(NormKernels.LayerNormRowAVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(NormKernels.LayerNormRowScalar))]
    public abstract void ApplyLayerNormRow(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> dst, float eps);
}
