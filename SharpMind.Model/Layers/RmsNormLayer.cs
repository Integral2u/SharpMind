using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public sealed class RmsNormLayer(int dim, float eps = 1e-5f, Tensor<float>? weight = null, Tensor<float>? bias = null) : NormLayer(dim, false, eps, weight, bias)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        if (Avx2.IsSupported) 
            NormKernels.RMSNormRowAVX2(src, weight, dst, rmsInv);
        else
            NormKernels.RMSNormRowScalar(src, weight, dst, rmsInv);
        
    }

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
        => Avx2.IsSupported
            ? NormKernels.RMSNormParamAVX2(row, Eps)
            : NormKernels.RMSNormParamScalar(row, Eps);

    protected override void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float rmsInv)
    {
        float rms = 1f / rmsInv;
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i] * rms;
    }
}
