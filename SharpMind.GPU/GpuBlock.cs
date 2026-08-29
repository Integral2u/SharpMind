using SharpMind.Core.Training;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.GPU;

/// <summary>
/// One transformer block's device-resident weights and the activations its backward needs.
///
/// The weights are uploaded once and live for the engine's lifetime; the activation fields
/// are arena tensors written by <see cref="GpuBackpropEngine"/>'s forward and valid only
/// until the arena is reset, i.e. for the rest of that step.
/// </summary>
internal sealed class GpuBlock : IDisposable
{
    public readonly DeviceBuffer Norm1W, Norm2W;
    public readonly GpuLinear Wq, Wk, Wv, Wo, WGated, WDown;
    // saved in forward (arena tensors, valid until Reset)
    public DeviceTensor X, Norm1Out, RInv1, Q, K, V, Probs, AttnOut, X1, Norm2Out, RInv2, Fused, Act;
    /// <summary>Flash path only: [B·H·S, 3] softmax statistics in place of <see cref="Probs"/>.</summary>
    public DeviceTensor Stats;

    /// <summary>
    /// A throw partway through leaves the earlier device buffers unreachable — the constructor
    /// never returns, so no caller can dispose them and DeviceBuffer has no finalizer. Unwind.
    /// </summary>
    public GpuBlock(GpuDevice dev, TransformerBlock block, Func<Core.Tensors.Tensor<float>, Parameter?> param)
    {
        var owned = new List<IDisposable>();
        try
        {
            Norm1W = DeviceBuffer.From(dev, block.Norm1.NormWeight); owned.Add(Norm1W);
            Norm2W = DeviceBuffer.From(dev, block.Norm2.NormWeight); owned.Add(Norm2W);
            var a = block.Attention; var f = (GatedFfnLayer)block.Ffn;
            Wq = Lin(dev, a.Wq, param, owned); Wk = Lin(dev, a.Wk, param, owned); Wv = Lin(dev, a.Wv, param, owned); Wo = Lin(dev, a.Wo, param, owned);
            WGated = Lin(dev, f.WGated!, param, owned); WDown = Lin(dev, f.WDown!, param, owned);
        }
        catch
        {
            foreach (var d in owned) d.Dispose();
            throw;
        }
    }

    private static GpuLinear Lin(GpuDevice dev, LinearLayer l, Func<Core.Tensors.Tensor<float>, Parameter?> param, List<IDisposable> owned)
    {
        var gl = l is TrainingLinearLayer { HasLoRA: true } t ? new GpuLinear(dev, l, param(t.LoRAA!), param(t.LoRAB!)) : new GpuLinear(dev, l, null, null);
        owned.Add(gl);
        return gl;
    }

    public IEnumerable<GpuLinear> Linears() { yield return Wq; yield return Wk; yield return Wv; yield return Wo; yield return WGated; yield return WDown; }

    public void Dispose() { Norm1W.Dispose(); Norm2W.Dispose(); foreach (var l in Linears()) l.Dispose(); }
}
