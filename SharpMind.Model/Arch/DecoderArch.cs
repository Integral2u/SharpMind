using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Arch;

// ─────────────────────────────────────────────────────────────────────────────
// DecoderArch — causal, autoregressive (GPT, LLaMA, Mistral)
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Decoder-only architecture. Each token attends only to itself and
/// tokens before it (causal mask). Used by all autoregressive LLMs.
/// </summary>
public sealed class DecoderArch : IArchitecture
{
    private readonly TransformerBlock[] _blocks;
    private bool _disposed;
    
    private readonly List<Tensor<float>> _cachedInputs = [];

    public DecoderArch(IEnumerable<TransformerBlock> blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        _blocks = [.. blocks];
        if (_blocks.Length == 0)
            throw new ArgumentException("DecoderArch requires at least one block.", nameof(blocks));
    }

    public int NumLayers => _blocks.Length;
    public TransformerBlock[] Blocks => _blocks;

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var block in _blocks)
            foreach (var p in block.Parameters())
                yield return p;
    }

    /// <summary>
    /// Passes hidden states through all blocks with causal masking.
    /// <paramref name="positionOffset"/> supports KV-cache decode:
    /// set to the current cache length to correctly encode positions.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0)
    {
        ThrowIfDisposed();
        DisposeCache();
        
        var current = hiddenStates;

        for (int i = 0; i < _blocks.Length; i++)
        {
            var next = _blocks[i].Forward(current, positionOffset, causal: true);
            if (i > 0) current.Dispose();
            _cachedInputs.Add(current);
            current = next;
        }
        _cachedInputs.Add(current);

        return current;
    }
    
    public void Backward(Tensor<float> dOutput)
    {
        ThrowIfDisposed();
        DisposeCache();
    }
    
    private void DisposeCache()
    {
        foreach (var t in _cachedInputs)
            t.Dispose();
        _cachedInputs.Clear();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        DisposeCache();
        foreach (var b in _blocks) b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(DecoderArch));
}
