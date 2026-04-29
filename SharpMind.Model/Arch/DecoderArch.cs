using SharpMind.Core.Tensors;
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

    public DecoderArch(IEnumerable<TransformerBlock> blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        _blocks = [.. blocks];
        if (_blocks.Length == 0)
            throw new ArgumentException("DecoderArch requires at least one block.", nameof(blocks));
    }

    public int NumLayers => _blocks.Length;

    /// <summary>
    /// Passes hidden states through all blocks with causal masking.
    /// <paramref name="positionOffset"/> supports KV-cache decode:
    /// set to the current cache length to correctly encode positions.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0)
    {
        ThrowIfDisposed();
        var current = hiddenStates;
        bool ownsLast = false;

        for (int i = 0; i < _blocks.Length; i++)
        {
            var next = _blocks[i].Forward(current, positionOffset, causal: true);
            if (ownsLast) current.Dispose();
            current = next;
            ownsLast = true;
        }

        return current;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _blocks) b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(DecoderArch));
}
