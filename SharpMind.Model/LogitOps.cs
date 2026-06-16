using SharpMind.Core.Memory;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;

namespace SharpMind.Model;

public sealed class LogitOps
{
    private readonly Tensor<float> _projectionWeight;
    private readonly byte[]? _rawWeight;
    private readonly GgufDtype? _rawDtype;
    private readonly TensorOps _ops;
    private readonly QuantizationOps? _qOps;

    public bool UseQuantized => _rawWeight != null && _rawDtype == GgufDtype.Q8_0 && _qOps != null;

    public LogitOps(Tensor<float> projectionWeight, byte[]? rawWeight, GgufDtype? rawDtype, TensorOps ops, QuantizationOps? qOps)
    {
        _projectionWeight = projectionWeight;
        _rawWeight = rawWeight;
        _rawDtype = rawDtype;
        _ops = ops;
        _qOps = qOps;
    }

    public unsafe Tensor<float> Project(Tensor<float> input, int M, int K, int N, Workspace? workspace = null)
    {
        if (UseQuantized)
        {
            Tensor<float> result = workspace != null
                ? workspace.Rent<float>([M, N])
                : new Tensor<float>(M, N);
            fixed (float* pInput = input.Data)
            fixed (float* pOutput = result.Data)
            fixed (byte* pRaw = _rawWeight!)
            {
                _qOps!.QuantizedMatMulQ8_0(pInput, pRaw, pOutput, M, K, N);
            }
            return result;
        }

        if (workspace != null)
        {
            var result = workspace.Rent<float>([M, N]);
            _ops.MatMulWithBTInto(input, _projectionWeight, result);
            return result;
        }
        return _ops.MatMulWithBT(input, _projectionWeight);
    }
}
