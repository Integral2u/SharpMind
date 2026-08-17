using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public abstract class LinearLayer : IDisposable
{
    protected Tensor<float> _weight;
    protected Tensor<float>? _bias;
    protected bool _ownsWeight;
    protected bool _ownsBias;
    private bool _disposed;

    /// <param name="allocateFullWeight">
    /// When <paramref name="weight"/> is null, whether to materialise a full
    /// <c>[inFeatures, outFeatures]</c> float weight (true, the training default)
    /// or only a one-column placeholder (false). Quantized inference layers run
    /// off raw bytes and never read this float weight, so on a large model the
    /// full copy is pure dead memory — ~28 GB on a 7B — and passing false here is
    /// what lets quantized-resident loading actually stay resident.
    /// </param>
    protected LinearLayer(string name, int inFeatures, int outFeatures, bool bias, Tensor<float>? weight, Tensor<float>? biasTensor, bool allocateFullWeight = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inFeatures);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outFeatures);
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _weight = weight ?? new Tensor<float>(inFeatures, allocateFullWeight ? outFeatures : 1);
        _bias = biasTensor ?? (bias ? new Tensor<float>(outFeatures) : null);
        _ownsWeight = weight == null;
        _ownsBias = biasTensor == null && _bias != null;
    }

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public bool HasBias => _bias is not null;
    public string Name { get; }
    public Tensor<float> Weight => _weight;
    public Tensor<float>? Bias => _bias;

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter($"{Name}.weight", _weight);
        if (_bias is not null)
            yield return new Parameter($"{Name}.bias", _bias);
    }

    public abstract Tensor<float> Forward(Tensor<float> input, Workspace? workspace = null);

    public virtual (Tensor<float> Output, LinearLayerState State) ForwardWithState(Tensor<float> input)
        => throw new NotSupportedException($"{GetType().Name} does not support ForwardWithState");
    public virtual Tensor<float> Backward(Tensor<float> gradOutput, LinearLayerState state)
        => throw new NotSupportedException($"{GetType().Name} does not support Backward");

    public virtual void FreeFloatWeight() { }
    public virtual void SetRawWeight(byte[]? rawData) { }

    /// <summary>
    /// Enables/disables quantization-aware training for this layer. Base
    /// implementation is a no-op: only <see cref="TrainingLinearLayer"/>
    /// (and equips) fake-quantize their forward pass. A null or
    /// <see cref="QuantDType.F32"/> target restores pure-float forward.
    /// </summary>
    public virtual void EnableQuantAwareTraining(QuantDType? target) { }

    public void ReplaceWeights(Tensor<float> weight, Tensor<float>? biasTensor)
    {
        ThrowIfDisposed();

        if (_ownsWeight) _weight.Dispose();
        if (_ownsBias) _bias?.Dispose();

        _weight = weight;
        _bias = biasTensor;
        _ownsWeight = false;
        _ownsBias = false;
        InvalidateCache();
    }

    public void ReplaceBias(Tensor<float> biasTensor)
    {
        ThrowIfDisposed();
        if (_ownsBias) _bias?.Dispose();
        _bias = biasTensor;
        _ownsBias = false;
    }

    public void LoadWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");
        data.CopyTo(_weight.Data);
        InvalidateCache();
    }

    public void LoadWeightTransposed(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _weight.ElementCount)
            throw new ArgumentException($"Expected {_weight.ElementCount} weight values, got {data.Length}.");

        int inF = InFeatures;
        int outF = OutFeatures;
        for (int o = 0; o < outF; o++)
            for (int i = 0; i < inF; i++)
                _weight.Data[i * outF + o] = data[o * inF + i];
        InvalidateCache();
    }

    protected virtual void InvalidateCache() { }

    public void LoadBias(ReadOnlySpan<float> data)
    {
        if (_bias is null) throw new InvalidOperationException("No bias.");
        if (data.Length != _bias.ElementCount)
            throw new ArgumentException($"Expected {_bias.ElementCount} bias values, got {data.Length}.");
        data.CopyTo(_bias.Data);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsWeight) _weight.Dispose();
        if (_ownsBias) _bias?.Dispose();
    }

    /// <summary>
    /// Adds the bias to every row of <paramref name="result"/> in place.
    ///
    /// This used to materialise the bias broadcast into a whole second
    /// [batchSize, OutFeatures] tensor and then add tensor-to-tensor. For the
    /// fused gate/up projection that duplicate is the single largest
    /// allocation in a layer, and during prefill it pushed the forward pass
    /// past the size Workspace.CalculateRequiredSize budgets for it — a long
    /// prompt died with "Workspace capacity exceeded" in the last layers.
    /// The add is O(batchSize × OutFeatures) against a matmul that is
    /// O(batchSize × OutFeatures × InFeatures), so doing it row-wise costs
    /// nothing measurable and allocates nothing at all.
    /// </summary>
    protected void AddBiasInPlace(Tensor<float> result, int batchSize)
    {
        var bias = _bias!.Data;
        for (int i = 0; i < batchSize; i++)
        {
            var row = result.RowSpan(i);
            for (int j = 0; j < bias.Length; j++)
                row[j] += bias[j];
        }
    }
    private protected void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LinearLayer));
}
