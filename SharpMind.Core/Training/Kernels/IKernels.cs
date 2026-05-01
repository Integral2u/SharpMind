using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.Kernels;

public interface ILinearBackward : IGradientKernel
{
    /// <summary>
    /// Returns dInput [B, InFeatures] and accumulates gradients into weight/bias parameters.
    /// </summary>
    Tensor<float> Compute(
        Tensor<float> dOutput,   // [B, OutFeatures]
        Tensor<float> input,     // [B, InFeatures]
        Parameter     weight,    // [OutFeatures, InFeatures]
        Parameter?    bias = null);
}

public interface IRMSNormBackward : IGradientKernel
{
    /// <summary>
    /// Returns dInput [T, D] and accumulates into weight gradient.
    /// </summary>
    Tensor<float> Compute(
        Tensor<float> dOutput,  // [T, D]
        Tensor<float> xNorm,    // [T, D]  x * rmsInv (saved from forward)
        float[]       rmsInv,   // [T]
        Parameter     weight);  // [D]
}

public interface ILayerNormBackward : IGradientKernel
{
    Tensor<float> Compute(
        Tensor<float> dOutput,
        Tensor<float> input,
        Parameter     weight,
        Parameter     bias,
        float         eps = 1e-5f);
}

public interface IAttentionBackward : IGradientKernel
{
    (Tensor<float> dQ, Tensor<float> dK, Tensor<float> dV) Compute(
        Tensor<float> dOut,   // [S, HeadDim]
        Tensor<float> q,      // [S, HeadDim]
        Tensor<float> k,      // [S, HeadDim]
        Tensor<float> v,      // [S, HeadDim]
        Tensor<float> probs,  // [S, S]
        float         scale);
}

public interface IEmbeddingBackward : IGradientKernel
{
    void Compute(
        Tensor<float> dOutput,   // [T, EmbedDim] flat
        Tensor<int>   tokenIds,  // [T] flat
        Parameter     weight);   // [VocabSize, EmbedDim]
}

public interface IActivationBackward : IGradientKernel
{
    Tensor<float> Compute(Tensor<float> dOutput, Tensor<float> preAct, ActivationType type);
}

public enum ActivationType
{
    SiLU,
    GELU
}
