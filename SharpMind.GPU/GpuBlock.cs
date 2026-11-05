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
    public readonly DeviceBuffer? Norm1B, Norm2B;

    public readonly GpuLinear Wq, Wk, Wv, Wo;

    /// <summary>Gated FFN weights (SwiGLU/GeGLU).</summary>
    public readonly GpuLinear? WGated, WDown;

    /// <summary>Dense FFN weights (GPT-2/BERT-style two-layer FFN).</summary>
    public readonly GpuLinear? W1, W2;

    public readonly bool LayerNorm, Dense;

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
            if (block.Norm1.NormBias is { } b1) { Norm1B = DeviceBuffer.From(dev, b1); owned.Add(Norm1B); }
            if (block.Norm2.NormBias is { } b2) { Norm2B = DeviceBuffer.From(dev, b2); owned.Add(Norm2B); }
            LayerNorm = block.Norm1 is LayerNormLayer;

            var a = block.Attention;
            Wq = Lin(dev, a.Wq, param, owned); Wk = Lin(dev, a.Wk, param, owned); Wv = Lin(dev, a.Wv, param, owned); Wo = Lin(dev, a.Wo, param, owned);

            switch (block.Ffn)
            {
                case GatedFfnLayer g:
                    WGated = Lin(dev, g.WGated!, param, owned); WDown = Lin(dev, g.WDown!, param, owned);
                    break;
                case DenseFfnLayer d:
                    W1 = Lin(dev, d.W1Layer!, param, owned); W2 = Lin(dev, d.W2Layer!, param, owned);
                    Dense = true;
                    break;
                default:
                    throw new NotSupportedException($"GPU block does not support FFN kind {block.Ffn.GetType().Name}.");
            }
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

    public IEnumerable<GpuLinear> Linears()
    {
        yield return Wq; yield return Wk; yield return Wv; yield return Wo;
        if (WGated is not null) yield return WGated;
        if (WDown is not null) yield return WDown;
        if (W1 is not null) yield return W1;
        if (W2 is not null) yield return W2;
    }

    public void Dispose()
    {
        Norm1W.Dispose(); Norm2W.Dispose();
        Norm1B?.Dispose(); Norm2B?.Dispose();
        foreach (var l in Linears()) l.Dispose();
    }
}
