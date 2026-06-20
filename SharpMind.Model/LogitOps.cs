using SharpMind.Core.Memory;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;

namespace SharpMind.Model;

public sealed class LogitOps(Tensor<float> projectionWeight, byte[]? rawWeight, GgufDtype? rawDtype, TensorOps ops, QuantizationOps? qOps)
{
    public bool UseQuantized => rawWeight != null && rawDtype == GgufDtype.Q8_0 && qOps != null;

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
                qOps!.QuantizedMatMulQ8_0(pInput, pRaw, pOutput, M, K, N);
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
