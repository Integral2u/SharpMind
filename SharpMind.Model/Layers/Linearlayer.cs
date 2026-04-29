using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

/// <summary>
/// A single linear (fully-connected) projection: out = xW^T + b.
/// Weights are stored as [OutFeatures, InFeatures] — the transpose is handled
/// by <see cref="TensorOps.MatMul"/> which transposes B before the kernel.
///
/// Used by all QKV projections, output projections, and FFN weight matrices.
/// </summary>
public sealed class LinearLayer : IDisposable
{
    private readonly Tensor<float> _weight; // [OutFeatures, InFeatures]
    private readonly Tensor<float>? _bias;  // [OutFeatures] or null
    private bool _disposed;

    // ── Construction ──────────────────────────────────────────────────────

    public LinearLayer(int inFeatures, int outFeatures, bool bias = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);

        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = new Tensor<float>(outFeatures, inFeatures);
        _bias = bias ? new Tensor<float>(outFeatures) : null;
    }

    // ── Properties ────────────────────────────────────────────────────────

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public bool HasBias => _bias is not null;

    public Tensor<float> Weight => _weight;
    public Tensor<float>? Bias => _bias;

    // ── Forward pass ──────────────────────────────────────────────────────

    /// <summary>
    /// Applies the linear projection.
    /// Input:  [*, InFeatures]  — any leading batch dimensions.
    /// Output: [*, OutFeatures]
    /// </summary>
    public Tensor<float> Forward(Tensor<float> input, TensorOps ops)
    {
        ThrowIfDisposed();

        // Flatten all batch dims to 2D for matmul, then restore shape
        bool needReshape = input.Rank > 2;
        int batchSize = input.ElementCount / input.Shape[^1];

        var flat = needReshape ? input.Reshape(batchSize, InFeatures) : input;

        // out = input @ weight^T  (weight is [OutFeatures, InFeatures])
        // TensorOps.MatMul transposes B internally so this gives [batch, OutFeatures]
        var output = ops.MatMul(flat, _weight);

        if (_bias is not null)
            TensorOps.AddInPlace(output, BroadcastBias(batchSize));

        if (needReshape)
        {
            int[] outDims = [.. input.Shape.Dims.ToArray()[..^1], OutFeatures];
            var reshaped = output.Reshape(outDims);
            output.Dispose();
            return reshaped;
        }

        return output;
    }

    // ── Weight initialisation ─────────────────────────────────────────────

    /// <summary>Loads pre-trained weights. Validates shape before copying.</summary>
    public void LoadWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException(
                $"Expected {_weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(_weight.Data);
    }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (_bias is null)
            throw new InvalidOperationException("This LinearLayer has no bias.");
        if (data.Length != _bias.ElementCount)
            throw new ArgumentException(
                $"Expected {_bias.ElementCount} bias values, got {data.Length}.");
        data.CopyTo(_bias.Data);
    }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weight.Dispose();
        _bias?.Dispose();
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// <summary>
    /// Broadcasts bias [OutFeatures] to [BatchSize, OutFeatures] for addition.
    /// Zero-copy: slices the existing bias tensor row by row into a new view.
    /// For small batch sizes this is cheaper than a full allocation.
    /// </summary>
    private Tensor<float> BroadcastBias(int batchSize)
    {
        var broadcast = new Tensor<float>(batchSize, OutFeatures);
        for (int i = 0; i < batchSize; i++)
            _bias!.Data.CopyTo(broadcast.RowSpan(i));
        return broadcast;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LinearLayer));
}