using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Arch;

// ─────────────────────────────────────────────────────────────────────────────
// EncoderArch — bidirectional (BERT, RoBERTa)
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Encoder-only architecture. Every token attends to every other token
/// (no causal mask). Used by BERT-style models for classification,
/// embeddings, and masked language modelling.
/// </summary>
public sealed class EncoderArch : IArchitecture
{
    private readonly TransformerBlock[] _blocks;
    private bool _disposed;

    public EncoderArch(IEnumerable<TransformerBlock> blocks)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        _blocks = [.. blocks];
        if (_blocks.Length == 0)
            throw new ArgumentException("EncoderArch requires at least one block.", nameof(blocks));
    }

    public int NumLayers => _blocks.Length;

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var block in _blocks)
            foreach (var p in block.Parameters())
                yield return p;
    }

    /// <summary>
    /// Passes hidden states through all blocks without causal masking.
    /// positionOffset is accepted for API compatibility but is typically 0
    /// for encoders since they see the full sequence in one pass.
    /// </summary>
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0)
    {
        return Forward(hiddenStates, null, positionOffset);
    }

    public Tensor<float> Forward(Tensor<float> hiddenStates, KVCache[] caches, int positionOffset = 0)
    {
        ThrowIfDisposed();
        var current = hiddenStates;
        bool ownsLast = false;

        for (int i = 0; i < _blocks.Length; i++)
        {
            // Encoders typically don't use KV caches, but we support the API
            var next = _blocks[i].Forward(current, caches != null ? caches[i] : null, positionOffset, causal: false);
            if (ownsLast) current.Dispose();
            current = next;
            ownsLast = true;
        }

        return current;
    }

    public void Backward(Tensor<float> dOutput)
    {
        ThrowIfDisposed();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _blocks) b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(EncoderArch));

}
