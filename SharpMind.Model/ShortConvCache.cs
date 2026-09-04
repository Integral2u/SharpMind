using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Recurrent-state holder for an LFM2 short-conv (no-attention) layer. Sits in the
/// same per-layer cache array as the attention KV caches so generators' reset/trim/
/// truncate loops, the chat-session snapshot/restore path, and speculative/medusa
/// rollback all treat every layer uniformly.
///
/// The layer keeps the most recent <c>l_cache - 1</c> rows of its gated conv input
/// (<c>b ⊙ x</c>) as state; that is the entire "cache" this type needs. It is
/// deliberately tiny: <c>(l_cache - 1) × HiddenDim</c> floats per sequence.
/// </summary>
public sealed class ShortConvCache : IKVCache
{
    private Tensor<float> _state;
    private int _length;
    private bool _disposed;

    /// <param name="stateRows">Number of retained rows (l_cache - 1).</param>
    /// <param name="channels">Hidden dimension.</param>
    /// <param name="batch">Sequences sharing this cache (1 for a single conversation).</param>
    /// <param name="maxSeqLen">Effective context bound used by the generator's IsFull check.</param>
    public ShortConvCache(int stateRows, int channels, int batch = 1, int maxSeqLen = 0)
    {
        if (stateRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(stateRows));
        StateRows = stateRows;
        Channels = channels;
        _state = new Tensor<float>(batch, stateRows, channels);
        MaxSeqLen = Math.Max(0, maxSeqLen);
    }

    /// <summary>Number of retained rows (l_cache - 1).</summary>
    public int StateRows { get; }

    /// <summary>Hidden dimension of the gated state rows.</summary>
    public int Channels { get; }

    /// <summary>Backing state tensor [batch, StateRows, Channels], updated in place by the layer.</summary>
    public Tensor<float> State => _state;

    /// <summary>No-op: short-conv layers never call the KV Update path.</summary>
    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
    }

    public void Reset()
    {
        ThrowIfDisposed();
        _state.Data.Clear();
        _length = 0;
    }

    /// <summary>The conv is local (kernel width l_cache), so trimming older KV rows never
    /// invalidates the retained state, which always holds the most recent rows.</summary>
    public void TrimToLast(int keep)
    {
        ThrowIfDisposed();
        if (keep < _length) _length = keep;
    }

    /// <summary>Truncating the conversation to its first <paramref name="length"/> tokens
    /// cannot reconstruct the retained state by itself — the caller must restore or
    /// re-prefill for the state to match the kept prefix.</summary>
    public void Truncate(int length)
    {
        ThrowIfDisposed();
        if (length < 0) length = 0;
        if (length < _length) _length = length;
    }

    public int Length => _length;
    public int MaxSeqLen { get; }
    public bool IsFull => _length >= MaxSeqLen;
    public bool IsContiguous => false;

    /// <summary>Applies <paramref name="seqLen"/> newly processed tokens to the position bookkeeping.</summary>
    public void Advance(int seqLen)
    {
        ThrowIfDisposed();
        if (seqLen >= int.MaxValue - _length) _length = int.MaxValue;
        else _length += seqLen;
    }

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead) => null;
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) => null;

    public object? Snapshot() => _length == 0 ? null : ((int)_length, _state.Data.ToArray());

    public void Restore(object? snapshot)
    {
        ThrowIfDisposed();
        if (snapshot is not (int pos, float[] data))
            throw new ArgumentException($"Invalid ShortConvCache snapshot: {snapshot?.GetType().Name ?? "null"}");
        if (data.Length != _state.ElementCount)
            throw new ArgumentException(
                $"ShortConvCache snapshot length {data.Length} != state element count {_state.ElementCount}.");
        data.CopyTo(_state.Data);
        _length = pos;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _state.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(ShortConvCache));
}