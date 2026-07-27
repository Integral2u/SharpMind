using JigSawDotNet;
using SharpMind.Core;
 
namespace SharpMind.Training;

/// <summary>
/// JigSaw-assembled training kernels. Hardware tier is selected once at
/// factory time — no runtime capability checks inside any kernel.
///
/// Two slots:
///   "adamw"   → per-element AdamW update (AVX2 or scalar)
///   "gradnorm" → L2 norm-squared accumulation (AVX2 or scalar)
/// </summary>
public abstract class TrainingOps
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Training)}.{nameof(TrainingKernels)}";
    [PuzzleCornerPiece(SharpMindConfig.KeyAdamW,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(TrainingKernels.AdamWUpdate_AVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(TrainingKernels.AdamWUpdate_Scalar))]
    public abstract void AdamWUpdate(
        Span<float> data, Span<float> grad,
        Span<float> m,    Span<float> v,
        float beta1,   float beta2,
        float bc1,     float bc2,
        float lr,      float epsilon,
        float decay);

    [PuzzleCornerPiece(SharpMindConfig.KeyGradNorm,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(TrainingKernels.L2NormSq_AVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(TrainingKernels.L2NormSq_Scalar))]
    public abstract float L2NormSq(ReadOnlySpan<float> data);
}
