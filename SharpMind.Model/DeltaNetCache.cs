using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Recurrent-state holder for a Qwen3.5 Gated DeltaNet layer. It carries the
/// conv window (the previous <c>kernel - 1</c> fused Q/K/V projection rows) and
/// the delta-rule memory matrix, one per value head. It slots into the same
/// per-layer cache array as the attention KV caches so reset/trim/truncate and
/// snapshot/restore loops treat every layer uniformly.
/// </summary>
public sealed class DeltaNetCache : IKVCache
{
    private readonly Tensor<float> _convState;
    private readonly Tensor<float> _deltaState;
    private int _length;
    private bool _disposed;

    /// <param name="convKernel">Causal conv kernel width (e.g. 4). The state keeps <c>convKernel - 1</c> rows.</param>
    /// <param name="convDim">Fused Q/K/V channel width (<c>2*keyDim + valueDim</c>).</param>
    /// <param name="numValueHeads">Number of value heads (the delta state is one matrix per head).</param>
    /// <param name="keyHeadDim">Key dimension of one head.</param>
    /// <param name="valueHeadDim">Value dimension of one head.</param>
    /// <param name="batch">Sequences sharing this cache (1 for a single conversation).</param>
    /// <param name="maxSeqLen">Effective context bound used by the generator's IsFull check.</param>
    public DeltaNetCache(int convKernel, int convDim, int numValueHeads, int keyHeadDim, int valueHeadDim,
        int batch = 1, int maxSeqLen = 0)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(convKernel, 2);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(convDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numValueHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyHeadDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(valueHeadDim);

        ConvRows = convKernel - 1;
        ConvDim = convDim;
        NumValueHeads = numValueHeads;
        KeyHeadDim = keyHeadDim;
        ValueHeadDim = valueHeadDim;

        _convState = new Tensor<float>(batch, ConvRows, convDim);
        _deltaState = new Tensor<float>(batch, numValueHeads, keyHeadDim * valueHeadDim);
        MaxSeqLen = Math.Max(0, maxSeqLen);
    }

    public int ConvRows { get; }
    public int ConvDim { get; }
    public int NumValueHeads { get; }
    public int KeyHeadDim { get; }
    public int ValueHeadDim { get; }

    /// <summary>Conv history [batch, ConvRows, ConvDim], updated in place by the layer.</summary>
    public Tensor<float> ConvState => _convState;

    /// <summary>Delta memory [batch, NumValueHeads, KeyHeadDim * ValueHeadDim], key-major per head.</summary>
    public Tensor<float> DeltaState => _deltaState;

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
    }

    public void Reset()
    {
        ThrowIfDisposed();
        _convState.Data.Clear();
        _deltaState.Data.Clear();
        _length = 0;
    }

    public void TrimToLast(int keep)
    {
        ThrowIfDisposed();
        if (keep < _length) _length = keep;
    }

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

    /// <summary>The delta state summarises the whole prefix; dropping the tail cannot rewind it.</summary>
    public bool SupportsTruncate => false;

    public void Advance(int seqLen)
    {
        ThrowIfDisposed();
        if (seqLen >= int.MaxValue - _length) _length = int.MaxValue;
        else _length += seqLen;
    }

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead) => null;
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) => null;

    public object? Snapshot()
        => _length == 0 ? null : ((int)_length, _convState.Data.ToArray(), _deltaState.Data.ToArray());

    public void Restore(object? snapshot)
    {
        ThrowIfDisposed();
        if (snapshot is not (int pos, float[] conv, float[] delta))
            throw new ArgumentException($"Invalid DeltaNetCache snapshot: {snapshot?.GetType().Name ?? "null"}");
        if (conv.Length != _convState.ElementCount || delta.Length != _deltaState.ElementCount)
            throw new ArgumentException("DeltaNetCache snapshot does not match the state size.");
        conv.CopyTo(_convState.Data);
        delta.CopyTo(_deltaState.Data);
        _length = pos;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _convState.Dispose();
        _deltaState.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(DeltaNetCache));
}
