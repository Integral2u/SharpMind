using JigSawDotNet;
using SharpMind.Core;
 
namespace SharpMind.Model.Layers;

public abstract class NormOps
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(NormKernels)}";

    private static volatile NormOps? _default;

    public static NormOps Default => _default
        ?? throw new InvalidOperationException(
            $"{nameof(NormOps)}.{nameof(Default)} has not been initialised. " +
            $"Call {nameof(NormOpsFactory)}.{nameof(NormOpsFactory.SetDefault)} at application startup.");

    internal static void SetDefault(NormOps instance) => _default = instance;

    [PuzzleCornerPiece(SharpMindConfig.KeyLayerNormRow,
        SharpMindConfig.ValAvx2, NS + "." + nameof(NormKernels.LayerNormRowAVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(NormKernels.LayerNormRowScalar))]
    public abstract void ApplyLayerNormRow(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> dst, float eps);
}
