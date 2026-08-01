using SharpMind.Core.Tensors;
using SharpMind.Core.Training.Kernels;
using JigSawDotNet;

namespace SharpMind.Core.Training;

/// <summary>
/// JigSaw-assembled gradient kernels for training. Hardware tier is selected
/// once at factory time — no runtime capability checks inside any kernel.
///
/// Each slot maps to a set of <see cref="GradientKernels"/> tiers and is
/// dispatched by <see cref="GradientMappingFactory"/>.
///
/// Every method returns the downstream gradient while <em>accumulating</em>
/// parameter gradients into the supplied <see cref="Parameter"/> objects, so
/// gradient accumulation across multiple batches works correctly.
/// </summary>
public abstract class GradientMapping
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Training)}.{nameof(Kernels)}.{nameof(GradientKernels)}";

    /// <summary>
    /// Linear backward: returns dInput [B, InFeatures] and accumulates
    /// gradients into weight/bias parameters.
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradLinear,
        SharpMindConfig.ValFma,    NS + "." + nameof(GradientKernels.Linear_FMA),
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.Linear_AVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.Linear_Scalar))]
    public abstract Tensor<float> Linear(
        Tensor<float> dOutput,   // [B, OutFeatures]
        Tensor<float> input,     // [B, InFeatures]
        Parameter     weight,    // [OutFeatures, InFeatures]
        Parameter?    bias = null);

    /// <summary>
    /// RMSNorm backward: returns dInput [T, D] and accumulates into weight
    /// gradient. <paramref name="rmsInv"/> is the per-row 1/rms value and
    /// <paramref name="xNorm"/> the normalised input, both saved during forward.
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradRMSNorm,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.RMSNorm_Scalar),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.RMSNorm_Scalar))]
    public abstract Tensor<float> RMSNorm(
        Tensor<float> dOutput,  // [T, D]
        Tensor<float> xNorm,    // [T, D]  x * rmsInv (saved from forward)
        float[]       rmsInv,   // [T]
        Parameter     weight);  // [D]

    /// <summary>
    /// LayerNorm backward: returns dInput [T, D] and accumulates into
    /// weight/bias gradients.
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradLayerNorm,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.LayerNorm_Scalar),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.LayerNorm_Scalar))]
    public abstract Tensor<float> LayerNorm(
        Tensor<float> dOutput,
        Tensor<float> input,
        Parameter     weight,
        Parameter     bias,
        float         eps = 1e-5f);

    /// <summary>
    /// Scaled dot-product attention backward: returns (dQ, dK, dV) for one head.
    /// Q,K,V: [SeqLen, HeadDim]   probs: [SeqLen, SeqLen]   dOut: [SeqLen, HeadDim]
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradAttention,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.Attention_Scalar),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.Attention_Scalar))]
    public abstract AttentionGradients Attention(
        Tensor<float> dOut,   // [S, HeadDim]
        Tensor<float> q,      // [S, HeadDim]
        Tensor<float> k,      // [S, HeadDim]
        Tensor<float> v,      // [S, HeadDim]
        Tensor<float> probs,  // [S, S]
        float         scale);

    /// <summary>
    /// Embedding backward: accumulates dOutput into the rows of the embedding
    /// weight gradient selected by tokenIds. There is no "dInput" for an
    /// embedding lookup — the integer indices have no gradient.
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradEmbedding,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.Embedding_Scalar),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.Embedding_Scalar))]
    public abstract void Embedding(
        Tensor<float> dOutput,   // [T, EmbedDim] flat
        Tensor<int>   tokenIds,  // [T] flat
        Parameter     weight);   // [VocabSize, EmbedDim]

    /// <summary>
    /// SiLU backward: returns dInput, d/dx [x * sigmoid(x)].
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradActivationSiLU,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.ActivationSiLU_AVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.ActivationSiLU_Scalar))]
    public abstract Tensor<float> ActivationSiLU(Tensor<float> dOutput, Tensor<float> preAct);

    /// <summary>
    /// GELU backward (tanh approximation derivative): returns dInput.
    /// </summary>
    [PuzzleCornerPiece(SharpMindConfig.KeyGradActivationGELU,
        SharpMindConfig.ValAvx2,   NS + "." + nameof(GradientKernels.ActivationGELU_AVX2),
        SharpMindConfig.ValScalar, NS + "." + nameof(GradientKernels.ActivationGELU_Scalar))]
    public abstract Tensor<float> ActivationGELU(Tensor<float> dOutput, Tensor<float> preAct);
}
