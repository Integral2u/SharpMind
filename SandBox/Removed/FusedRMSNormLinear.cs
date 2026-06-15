using JigSawDotNet;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

// Fused RmsNorm + Linear layer — reduces memory bandwidth by fusing operations

public sealed class FusedRMSNormLinear : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(FusedKernels)}";

    private readonly int _hiddenDim;
    private readonly Tensor<float> _normWeight;
    private readonly Tensor<float> _weight;
    private readonly Tensor<float>? _bias;
    private bool _disposed;

    public FusedRMSNormLinear(int hiddenDim, bool hasBias = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);

        _hiddenDim = hiddenDim;
        _normWeight = Tensor<float>.Ones(hiddenDim);
        _weight = new Tensor<float>(hiddenDim, hiddenDim);
        _bias = hasBias ? new Tensor<float>(hiddenDim) : null;
    }

    public int HiddenDim => _hiddenDim;
    public Tensor<float> NormWeight => _normWeight;
    public Tensor<float> Weight => _weight;
    public Tensor<float>? Bias => _bias;

    [PuzzleCornerPiece(SharpMindConfig.KeyFusedNormLinear,
        SharpMindConfig.ValFusedNormLinearAVX2, NS + "." + nameof(FusedKernels.FusedRMSNormLinearAVX2),
        SharpMindConfig.ValFusedNormLinearScalar, NS + "." + nameof(FusedKernels.FusedRMSNormLinearScalar))]
    public unsafe void Apply(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> weight,
        float* bias,
        Span<float> dst,
        float rmsInv)
    {
        throw new NotImplementedException("JigSaw invokes via reflection - implementation in FusedKernels");
    }

    public Tensor<float> Forward(Tensor<float> input, float rmsInv)
    {
        ThrowIfDisposed();
        int rows = input.Shape.Rows;
        int cols = input.Shape.Cols;

        var output = new Tensor<float>(rows, cols, _hiddenDim);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var src = input.RowSpan(r * cols + c);
                var dst = output.RowSpan(r * cols + c);

                unsafe
                {
                    float* biasPtr = _bias is not null ? _bias.DataPtr : null;
                    fixed (float* pDst = dst)
                    {
                        FusedKernels.FusedRMSNormLinearScalar(
                            src, _normWeight.Data, _weight.Data, biasPtr, dst, rmsInv);
                    }
                }
            }
        }

        return output;
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter("fused.normweight", _normWeight);
        yield return new Parameter("fused.weight", _weight);
        if (_bias is not null)
            yield return new Parameter("fused.bias", _bias);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _normWeight.Dispose();
        _weight.Dispose();
        _bias?.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(FusedRMSNormLinear));
}