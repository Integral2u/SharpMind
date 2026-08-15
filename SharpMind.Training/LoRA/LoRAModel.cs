using SharpMind.Core.Training;
using SharpMind.Model;

namespace SharpMind.Training.LoRA;

/// <summary>
/// Full model with LoRA adaptation.
/// </summary>
public class LoRAModel : IDisposable
{
    private readonly Transformer _baseModel;
    private readonly LoRAAttention _attention;
    private readonly LoRAFFN? _ffn;
    private readonly LoRAConfig _config;
    private bool _disposed;

    public LoRAModel(Transformer baseModel, LoRAConfig config)
    {
        _baseModel = baseModel;
        _config = config;

        _attention = new LoRAAttention(
            baseModel.Config.HiddenDim,
            baseModel.Config.NumHeads,
            baseModel.Config.HeadDim,
            config);

        if (baseModel.Config.FfnDim > 0)
        {
            _ffn = new LoRAFFN(
                baseModel.Config.HiddenDim,
                baseModel.Config.FfnDim,
                config);
        }
    }

    public IEnumerable<Parameter> LoRAParameters()
    {
        foreach (var p in _attention.Parameters())
            yield return p;
        if (_ffn is not null)
            foreach (var p in _ffn.Parameters())
                yield return p;
    }

    /// <summary>
    /// Number of trainable parameters vs original.
    /// </summary>
    public double TrainableRatio()
    {
        long baseParams = _baseModel.ParameterCount;
        long loraParams = LoRAParameters().Sum(p => p.Data.ElementCount);
        return (double)loraParams / baseParams;
    }

    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { _attention.Dispose(); _ffn?.Dispose(); }
        _disposed = true;
    }
}