using SharpMind.Core.Ops;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Layers;

public abstract class LinearLayer : IDisposable
{
    private bool _disposed;

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public string Name { get; }
    public bool HasBias => _bias is not null;

    protected Tensor<float>? _bias;

    protected LinearLayer(string name, int inFeatures, int outFeatures, Tensor<float>? biasTensor)
    {
        Name = name;
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        _bias = biasTensor;
    }

    public abstract Tensor<float> Forward(Tensor<float> input, TensorOps ops, Workspace? workspace = null);
    public abstract IEnumerable<Parameter> Parameters();

    public virtual void FreeFloatWeight() { }

    protected Tensor<float> BroadcastBias(int batchSize)
    {
        var broadcast = new Tensor<float>(batchSize, OutFeatures);
        for (int i = 0; i < batchSize; i++)
            _bias!.Data.CopyTo(broadcast.RowSpan(i));
        return broadcast;
    }

    protected void AddBias(Tensor<float> output, int batchSize, Workspace? workspace)
    {
        if (_bias is null) return;
        if (workspace != null)
        {
            var biasB = workspace.Rent<float>([batchSize, OutFeatures]);
            for (int i = 0; i < batchSize; i++)
                _bias!.Data.CopyTo(biasB.RowSpan(i));
            TensorOps.AddInPlace<float>(output, biasB);
        }
        else
        {
            TensorOps.AddInPlace<float>(output, BroadcastBias(batchSize));
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        DisposeCore();
    }

    protected abstract void DisposeCore();
}
