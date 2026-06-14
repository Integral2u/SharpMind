using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public interface IActivationHook
{
    void OnPreAttention(int layer, Tensor<float> hiddenStates);
    void OnPostAttention(int layer, Tensor<float> attnOut);
    void OnPostFFN(int layer, Tensor<float> ffnOut);
}
