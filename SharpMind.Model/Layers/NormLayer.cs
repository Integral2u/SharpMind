using JigSawDotNet;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

/// <summary>
/// Normalisation layer assembled by JigSawDotNet.
/// The "norm" mapping key selects the implementation:
///   "rmsnorm"   → RMSNorm (LLaMA, Mistral)
///   "layernorm" → LayerNorm (GPT-2, BERT)
///
/// Weight and optional bias are owned by this instance.
/// </summary>
public abstract class NormLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(NormKernels)}";
    protected readonly Tensor<float> Weight;
    protected readonly Tensor<float>? Bias;  // LayerNorm only
    protected readonly float Eps;
    private bool _disposed;

    protected NormLayer(int dim, bool hasBias = false, float eps = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        Dim = dim;
        Eps = eps;
        Weight = Tensor<float>.Ones(dim);
        Bias = hasBias ? Tensor<float>.Zeros(dim) : null;
    }

    public int Dim { get; }

    // ── PuzzleCornerPieces ────────────────────────────────────────────────

    [PuzzleCornerPiece(SharpMindConfig.KeyNorm,
        SharpMindConfig.ValNormRMSAvx2,
            NS + "." + nameof(NormKernels.RMSNormRowAVX2),
        SharpMindConfig.ValNormRMSScalar,
            NS + "." + nameof(NormKernels.RMSNormRowScalar),
        SharpMindConfig.ValNormLayerAvx2,
            NS + "." + nameof(NormKernels.LayerNormRowAVX2),
        SharpMindConfig.ValNormLayerScalar,
            NS + "." + nameof(NormKernels.LayerNormRowScalar))]
    public abstract void ApplyRow(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> weight,
        Span<float> dst,
        float scalarParam);  // rmsInv for RMSNorm, eps for LayerNorm

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>Normalises x row-wise and returns a new tensor of the same shape.</summary>
    public Tensor<float> Forward(Tensor<float> x)
    {
        ThrowIfDisposed();
        if (x.Shape[^1] != Dim)
            throw new ArgumentException(
                $"NormLayer expects last dim {Dim}, got {x.Shape[^1]}.");

        var result = new Tensor<float>(x.Shape);
        int rows = x.ElementCount / Dim;

        for (int i = 0; i < rows; i++)
        {
            float param = ComputeScalarParam(x.RowSpan(i));
            ApplyRow(x.RowSpan(i), Weight.Data, result.RowSpan(i), param);
        }
        return result;
    }

    /// <summary>Computes the per-row scalar parameter passed to <see cref="ApplyRow"/>.</summary>
    protected abstract float ComputeScalarParam(ReadOnlySpan<float> row);

    public void LoadWeight(ReadOnlySpan<float> data) => data.CopyTo(Weight.Data);
    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (Bias is null) throw new InvalidOperationException("This norm has no bias.");
        data.CopyTo(Bias.Data);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            Weight.Dispose();
            Bias?.Dispose();
        }
        _disposed = true;
    }

    ~NormLayer() => Dispose(false);

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(NormLayer));
}
