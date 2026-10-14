using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

/// <summary>
/// Builds the gemma-4 per-layer input tensor consumed by every block's PLE tail.
/// Mirrors llama.cpp's <c>gemma4.cpp</c>:
/// <code>
/// inp_per_layer = per_layer_token_embd[tokens] * sqrt(n_embd_per_layer)
/// proj          = per_layer_model_proj · inp_batch * (1 / sqrt(n_embd))
/// inp_per_layer = (rmsnorm(proj, per_layer_proj_norm) + inp_per_layer) * (1 / sqrt(2))
/// </code>
/// The token table is kept as raw quantized bytes (a float copy of the Q5_K table
/// is ~11 GB); only the rows for the current token ids are dequantized. The result
/// is laid out <c>[NumLayers, tokens, PerLayerEmbdDim]</c>, layer-major, so each
/// block reads a single contiguous slice.
/// </summary>
public sealed class PerLayerInputs : IDisposable
{
    private readonly LinearLayer _modelProj;
    private readonly NormLayer _projNorm;
    private readonly byte[] _rawTokEmbd;
    private readonly QuantDType _tokDtype;
    private readonly QuantizationOps _qOps;
    private readonly int _numLayers;
    private readonly int _perLayerDim;
    private readonly int _hiddenDim;
    private readonly int _vocab;
    private readonly int _totalDim;
    private readonly int _rowBytes;
    private readonly float _tokScale;
    private readonly float _projScale;
    private readonly float _inputScale;
    private bool _disposed;

    public PerLayerInputs(
        LinearLayer modelProj,
        NormLayer projNorm,
        byte[] rawTokEmbd,
        QuantDType tokDtype,
        QuantizationOps qOps,
        int numLayers,
        int perLayerDim,
        int hiddenDim,
        int vocab)
    {
        ArgumentNullException.ThrowIfNull(modelProj);
        ArgumentNullException.ThrowIfNull(projNorm);
        ArgumentNullException.ThrowIfNull(rawTokEmbd);
        ArgumentNullException.ThrowIfNull(qOps);
        if (vocab <= 0) throw new ArgumentOutOfRangeException(nameof(vocab));
        if (rawTokEmbd.Length % vocab != 0)
            throw new ArgumentException(
                $"per-layer token table has {rawTokEmbd.Length} bytes, not a multiple of the vocabulary {vocab}.");

        _modelProj = modelProj;
        _projNorm = projNorm;
        _rawTokEmbd = rawTokEmbd;
        _tokDtype = tokDtype;
        _qOps = qOps;
        _numLayers = numLayers;
        _perLayerDim = perLayerDim;
        _hiddenDim = hiddenDim;
        _vocab = vocab;
        _totalDim = perLayerDim * numLayers;
        _rowBytes = rawTokEmbd.Length / vocab;
        _tokScale = MathF.Sqrt(perLayerDim);
        _projScale = 1f / MathF.Sqrt(hiddenDim);
        _inputScale = 1f / MathF.Sqrt(2f);
    }

    /// <summary>
    /// Builds the per-layer input tensor <c>[NumLayers, tokens, PerLayerEmbdDim]</c>
    /// for <paramref name="tokenIds"/>. The caller owns the returned tensor.
    /// </summary>
    public Tensor<float> Build(Tensor<float> scaledEmbedding, ReadOnlySpan<int> tokenIds, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();
        int tokens = tokenIds.Length;
        if (tokens == 0)
            throw new ArgumentException("PerLayerInputs.Build requires at least one token.", nameof(tokenIds));
        if (scaledEmbedding.ElementCount != (long)tokens * _hiddenDim)
            throw new ArgumentException(
                $"scaledEmbedding holds {scaledEmbedding.ElementCount} values but expected {tokens} x {_hiddenDim}.");

        // 1. Per-layer token rows [tokens, totalDim]; within a row the layout is
        //    layer-major (layer L occupies [L*perLayerDim, (L+1)*perLayerDim)).
        var tokEmb = new Tensor<float>(tokens, _totalDim);
        using (var ms = new MemoryStream(_rawTokEmbd, writable: false))
        using (var reader = new BinaryReader(ms))
        {
            var dest = tokEmb.Data;
            for (int r = 0; r < tokens; r++)
            {
                int id = tokenIds[r];
                if ((uint)id >= (uint)_vocab)
                    throw new InvalidDataException($"Token id {id} is out of range for the per-layer token table.");
                ms.Position = (long)id * _rowBytes;
                _qOps.ReadFor(_tokDtype, reader, dest.Slice(r * _totalDim, _totalDim), _totalDim);
            }
            for (int i = 0; i < dest.Length; i++)
                dest[i] *= _tokScale;
        }

        // 2. Model projection of the (already embedding-scaled) hidden states.
        var flatEmb = scaledEmbedding.Rank > 2
            ? scaledEmbedding.Reshape(tokens, _hiddenDim)
            : scaledEmbedding;
        using var projRaw = _modelProj.Forward(flatEmb, workspace);   // [tokens, totalDim]
        var p = projRaw.Data;
        for (int i = 0; i < p.Length; i++)
            p[i] *= _projScale;

        // 3. Weighted RMSNorm over each layer slice of the projection. Reshaping to
        //    [tokens*numLayers, perLayerDim] gives one row per (token, layer), which
        //    is exactly the norm's last-dim contract.
        using var projView = projRaw.Reshape(tokens * _numLayers, _perLayerDim);
        using var projNormed = _projNorm.Forward(projView, workspace);

        // 4. Combine and permute to [NumLayers, tokens, perLayerDim].
        var result = new Tensor<float>(_numLayers, tokens, _perLayerDim);
        var rd = result.Data;
        var pn = projNormed.Data;
        var td = tokEmb.Data;
        for (int r = 0; r < tokens; r++)
        {
            for (int l = 0; l < _numLayers; l++)
            {
                int src = (r * _numLayers + l) * _perLayerDim;
                int dst = (l * tokens + r) * _perLayerDim;
                for (int d = 0; d < _perLayerDim; d++)
                    rd[dst + d] = (pn[src + d] + td[r * _totalDim + l * _perLayerDim + d]) * _inputScale;
            }
        }

        tokEmb.Dispose();
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _modelProj.Dispose();
        _projNorm.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(PerLayerInputs));
}
