using SharpMind.Core.Memory;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public sealed class LogitOps(Tensor<float> projectionWeight, byte[]? rawWeight, QuantDType? rawDtype, TensorOps ops, QuantizationOps? qOps)
{
    private readonly Tensor<float> projectionWeight = projectionWeight;
    private readonly byte[]? rawWeight = rawWeight;
    private readonly TensorOps ops = ops;
    private readonly QuantizationOps? qOps = qOps;
    public readonly bool UseQuantized = rawWeight != null && qOps != null && (rawDtype == QuantDType.Q8_0 || rawDtype == QuantDType.Q5_0 || rawDtype == QuantDType.Q6_K || rawDtype == QuantDType.Q6_K_S);

    public unsafe Tensor<float> Project(Tensor<float> input, int M, int K, int N, Workspace? workspace = null)
    {
        if (UseQuantized)
        {
            Tensor<float> result = workspace != null
                ? workspace.Rent<float>([M, N])
                : new Tensor<float>(M, N);
            fixed (float* pInput = input.Data)
            fixed (float* pOutput = result.Data)
            fixed (byte* pRaw = rawWeight!)
            {
                switch (rawDtype)
                {
                    case QuantDType.Q8_0:
                        qOps!.QuantizedMatMulQ8_0(pInput, pRaw, pOutput, M, K, N);
                        break;
                    case QuantDType.Q5_0:
                        qOps!.QuantizedMatMulQ5_0(pInput, pRaw, pOutput, M, K, N);
                        break;
                    case QuantDType.Q6_K:
                    case QuantDType.Q6_K_S:
                        qOps!.QuantizedMatMulQ6K(pInput, pRaw, pOutput, M, K, N);
                        break;
                }
            }
            return result;
        }

        if (workspace != null)
        {
            var result = workspace.Rent<float>([M, N]);
            ops.MatMulWithBTInto(input, projectionWeight, result);
            return result;
        }
        return ops.MatMulWithBT(input, projectionWeight);
    }
}
