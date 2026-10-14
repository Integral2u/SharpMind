using SharpMind.Core;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

/// <summary>
/// Gemma-4 per-layer embedding (PLE) block tail. Consumes this block's slice of the
/// per-layer input tensor built by <see cref="PerLayerInputs"/> and adds a projection
/// of it back onto the residual stream, matching llama.cpp's <c>gemma4.cpp</c>:
/// <code>
/// cur = gelu(inp_gate · x); cur *= per_layer_inputs[layer];
/// cur = proj · cur; cur = rmsnorm(cur, post_norm); x += cur
/// </code>
/// <c>inp_gate</c> and <c>proj</c> are always F32, so they are float (training-style)
/// linear layers whose weights are held by reference and refilled in place on a
/// streaming reload.
/// </summary>
public sealed class PerLayerEmbedding : IDisposable
{
    private readonly LinearLayer _inpGate;
    private readonly LinearLayer _proj;
    private readonly NormLayer _postNorm;
    private readonly int _layerIdx;
    private bool _disposed;

    public PerLayerEmbedding(int layerIdx, LinearLayer inpGate, LinearLayer proj, NormLayer postNorm)
    {
        ArgumentNullException.ThrowIfNull(inpGate);
        ArgumentNullException.ThrowIfNull(proj);
        ArgumentNullException.ThrowIfNull(postNorm);
        _layerIdx = layerIdx;
        _inpGate = inpGate;
        _proj = proj;
        _postNorm = postNorm;
    }

    /// <summary>
    /// Applies the PLE tail to <paramref name="x"/> in place. <paramref name="perLayerInputs"/>
    /// is the <c>[NumLayers, tokens, PerLayerEmbdDim]</c> tensor; the block reads its own slice.
    /// </summary>
    public void ApplyInPlace(Tensor<float> x, Tensor<float> perLayerInputs, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        int hidden = x.Shape[^1];
        int tokens = x.ElementCount / hidden;
        var flat = x.Rank > 2 ? x.Reshape(tokens, hidden) : x;

        var layerInput = perLayerInputs.Slice(_layerIdx);   // [tokens, PerLayerEmbdDim]

        var gate = _inpGate.Forward(flat, workspace);        // [tokens, PerLayerEmbdDim]
        var g = gate.Data;
        var li = layerInput.Data;
        for (int i = 0; i < g.Length; i++)
            g[i] = MathEx.Gelu(g[i]) * li[i];

        var cur = _proj.Forward(gate, workspace);            // [tokens, HiddenDim]
        gate.Dispose();

        _postNorm.ForwardInPlace(cur);
        flat.AddInPlace(cur);
        cur.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _inpGate.Dispose();
        _proj.Dispose();
        _postNorm.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(PerLayerEmbedding));
}
