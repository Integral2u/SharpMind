using System.Numerics;
using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.DeltaNet;

/// <summary>
/// Qwen3.5 Gated DeltaNet linear-attention branch. Replaces the softmax attention
/// branch on the 3-of-4 hybrid layers, maintaining a fixed-size recurrent state
/// instead of a KV cache:
///
///   [q | k | v] = conv1d(qkv_projection(x)); q,k = l2norm(q,k)
///   S = exp(g) * S + k ⊗ (beta * (v - Sᵀk))
///   o = Sᵀq
///   out = W_out(rmsnorm(o) * silu(z))
///
/// The conv window and the delta state live in <see cref="DeltaNetCache"/> so the
/// generators' reset/trim/snapshot loops treat this layer like any other.
/// </summary>
public sealed class DeltaNetLayer : IDisposable
{
    private readonly ModelConfig _config;
    private readonly LinearLayer _wqkv;
    private readonly LinearLayer _wgate;
    private readonly LinearLayer _walpha;
    private readonly LinearLayer _wbeta;
    private readonly LinearLayer _wout;

    private Tensor<float> _convKernel;
    private Tensor<float> _ssmA;
    private Tensor<float> _ssmDt;
    private Tensor<float> _ssmNorm;
    private bool _ownsConvKernel;
    private bool _ownsSsmA;
    private bool _ownsSsmDt;
    private bool _ownsSsmNorm;

    private readonly Tensor<float> _fallbackConvState;
    private readonly Tensor<float> _fallbackDeltaState;
    private bool _disposed;

    public DeltaNetLayer(ModelConfig config, QuantizationOps qOps,
        TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        int hidden = config.HiddenDim;
        int convDim = config.SsmConvDim;
        int valueDim = config.SsmValueDim;
        int nVHeads = config.SsmTimeStepRank;

        var tm = weights?.TensorMeta;
        var qkvDtype = weights?.QuantDtypeWqkv ?? tm?.GetValueOrDefault("RawWqkvFull").Dtype ?? QuantDType.F32;
        var gateDtype = weights?.QuantDtypeWGate ?? tm?.GetValueOrDefault("RawWGate").Dtype ?? QuantDType.F32;
        // alpha/beta are 16-wide and always run on the float path; see
        // TransformerWeights.ResolveFloatTarget for why.
        const QuantDType alphaDtype = QuantDType.F32;
        const QuantDType betaDtype = QuantDType.F32;
        var outDtype = weights?.QuantDtypeWOut ?? tm?.GetValueOrDefault("RawWOut").Dtype ?? QuantDType.F32;

        _wqkv = LinearLayerFactory.Create("attn_qkv", hidden, convDim, false, weights?.Wqkv, null, qkvDtype, mapping);
        _wgate = LinearLayerFactory.Create("attn_gate", hidden, valueDim, false, weights?.WGate, null, gateDtype, mapping);
        _walpha = LinearLayerFactory.Create("ssm_alpha", hidden, nVHeads, false, weights?.WAlpha, null, alphaDtype, mapping);
        _wbeta = LinearLayerFactory.Create("ssm_beta", hidden, nVHeads, false, weights?.WBeta, null, betaDtype, mapping);
        _wout = LinearLayerFactory.Create("ssm_out", valueDim, hidden, false, weights?.WOut, null, outDtype, mapping);

        _convKernel = weights?.WConv ?? new Tensor<float>(config.SsmConvKernel, convDim);
        _ownsConvKernel = weights?.WConv is null;
        _ssmA = weights?.SsmA ?? new Tensor<float>(nVHeads);
        _ownsSsmA = weights?.SsmA is null;
        _ssmDt = weights?.SsmDt ?? new Tensor<float>(nVHeads);
        _ownsSsmDt = weights?.SsmDt is null;
        _ssmNorm = weights?.SsmNorm ?? new Tensor<float>(config.SsmValueHeadDim);
        _ownsSsmNorm = weights?.SsmNorm is null;

        _fallbackConvState = new Tensor<float>(1, config.SsmConvKernel - 1, convDim);
        _fallbackDeltaState = new Tensor<float>(1, nVHeads, config.SsmStateSize * config.SsmValueHeadDim);
    }

    public LinearLayer Wqkv => _wqkv;
    public LinearLayer WGate => _wgate;
    public LinearLayer WAlpha => _walpha;
    public LinearLayer WBeta => _wbeta;
    public LinearLayer WOut => _wout;

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (name.Contains("ssm_alpha", StringComparison.OrdinalIgnoreCase))
            _walpha.LoadWeightTransposed(data);
        else if (name.Contains("ssm_beta", StringComparison.OrdinalIgnoreCase))
            _wbeta.LoadWeightTransposed(data);
        else if (name.Contains("ssm_conv1d", StringComparison.OrdinalIgnoreCase))
            CopyInto(_convKernel, data, "ssm_conv1d");
        else if (name.Contains("ssm_dt", StringComparison.OrdinalIgnoreCase))
            CopyInto(_ssmDt, data, "ssm_dt");
        else if (name.Contains("ssm_norm", StringComparison.OrdinalIgnoreCase))
            CopyInto(_ssmNorm, data, "ssm_norm");
        else if (name.Contains("ssm_a", StringComparison.OrdinalIgnoreCase))
            CopyInto(_ssmA, data, "ssm_a");
        else if (name.Contains("ssm_out", StringComparison.OrdinalIgnoreCase))
            _wout.LoadWeightTransposed(data);
        else if (name.Contains("attn_qkv", StringComparison.OrdinalIgnoreCase))
            _wqkv.LoadWeightTransposed(data);
        else if (name.Contains("attn_gate", StringComparison.OrdinalIgnoreCase))
            _wgate.LoadWeightTransposed(data);
    }

    public bool SetRawWeight(string name, byte[] rawData, QuantDType _)
    {
        ThrowIfDisposed();
        if (name.EndsWith(".bias", StringComparison.OrdinalIgnoreCase)) return false;
        if (name.Contains("attn_qkv", StringComparison.OrdinalIgnoreCase))
        {
            _wqkv.SetRawWeight(rawData);
            return true;
        }
        if (name.Contains("attn_gate", StringComparison.OrdinalIgnoreCase))
        {
            _wgate.SetRawWeight(rawData);
            return true;
        }
        if (name.Contains("ssm_alpha", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("ssm_beta", StringComparison.OrdinalIgnoreCase))
            return false;
        if (name.Contains("ssm_out", StringComparison.OrdinalIgnoreCase))
        {
            _wout.SetRawWeight(rawData);
            return true;
        }
        return false;
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        ThrowIfDisposed();
        if (weights.Wqkv != null) _wqkv.ReplaceWeights(weights.Wqkv, null);
        _wqkv.SetRawWeight(weights.RawWqkvFull);
        if (weights.WGate != null) _wgate.ReplaceWeights(weights.WGate, null);
        _wgate.SetRawWeight(weights.RawWGate);
        if (weights.WAlpha != null) _walpha.ReplaceWeights(weights.WAlpha, null);
        _walpha.SetRawWeight(null);
        if (weights.WBeta != null) _wbeta.ReplaceWeights(weights.WBeta, null);
        _wbeta.SetRawWeight(null);
        if (weights.WOut != null) _wout.ReplaceWeights(weights.WOut, null);
        _wout.SetRawWeight(weights.RawWOut);

        ReplaceSmall(ref _convKernel, ref _ownsConvKernel, weights.WConv);
        ReplaceSmall(ref _ssmA, ref _ownsSsmA, weights.SsmA);
        ReplaceSmall(ref _ssmDt, ref _ownsSsmDt, weights.SsmDt);
        ReplaceSmall(ref _ssmNorm, ref _ownsSsmNorm, weights.SsmNorm);
    }

    private static void ReplaceSmall(ref Tensor<float> current, ref bool owns, Tensor<float>? replacement)
    {
        if (replacement is null) return;
        if (owns) current.Dispose();
        current = replacement;
        owns = false;
    }

    public Tensor<float> Forward(Tensor<float> x, IKVCache? cache, IWorkspace? workspace = null)
    {
        ThrowIfDisposed();

        int batch = 1;
        int seq = x.Shape[0];
        if (x.Rank == 3)
        {
            batch = x.Shape[0];
            seq = x.Shape[1];
        }

        using var qkvProjected = _wqkv.Forward(x, workspace);
        using var z = _wgate.Forward(x, workspace);
        using var betaProjected = _wbeta.Forward(x, workspace);
        using var alphaProjected = _walpha.Forward(x, workspace);

        int hidden = _config.HiddenDim;
        int keyDim = _config.SsmKeyDim;
        int valueDim = _config.SsmValueDim;
        int convDim = _config.SsmConvDim;
        int nKHeads = _config.SsmGroupCount;
        int nVHeads = _config.SsmTimeStepRank;
        int keyHeadDim = _config.SsmStateSize;
        int valueHeadDim = _config.SsmValueHeadDim;
        int taps = _config.SsmConvKernel;
        int stateRows = taps - 1;
        int groupSize = nVHeads / nKHeads;
        float eps = _config.NormEps;
        float qScale = 1f / MathF.Sqrt(keyHeadDim);

        var deltaCache = cache as DeltaNetCache;
        var convState = deltaCache?.ConvState ?? _fallbackConvState;
        var deltaState = deltaCache?.DeltaState ?? _fallbackDeltaState;
        if (convState.Shape[0] < batch || deltaState.Shape[0] < batch)
            throw new InvalidOperationException(
                $"DeltaNetCache holds {convState.Shape[0]} sequence(s) but this forward sees {batch}.");

        var output = x.Rank == 3
            ? new Tensor<float>(batch, seq, valueDim)
            : new Tensor<float>(seq, valueDim);

        var qkvData = qkvProjected.Data;
        var zData = z.Data;
        var betaData = betaProjected.Data;
        var alphaData = alphaProjected.Data;
        var convKernel = _convKernel.Data;
        var ssmA = _ssmA.Data;
        var ssmDt = _ssmDt.Data;
        var ssmNorm = _ssmNorm.Data;
        var convStateData = convState.Data;
        var deltaStateData = deltaState.Data;
        var outputData = output.Data;

        int convStateStride = stateRows * convDim;
        int deltaLayerStride = keyHeadDim * valueHeadDim;
        var convRow = new float[convDim];
        var residual = new float[valueHeadDim];

        for (int b = 0; b < batch; b++)
        {
            int convStateBase = b * convStateStride;
            int deltaStateBase = b * nVHeads * deltaLayerStride;

            for (int t = 0; t < seq; t++)
            {
                int qkvRowOffset = (b * seq + t) * convDim;
                var qkvRow = qkvData.Slice(qkvRowOffset, convDim);

                // Causal depthwise conv over [state | current], then SiLU. The
                // state holds the previous (kernel - 1) rows per channel.
                if (stateRows > 0)
                {
                    for (int c = 0; c < convDim; c++)
                    {
                        float acc = 0f;
                        for (int k = 0; k < stateRows; k++)
                            acc += convStateData[convStateBase + k * convDim + c] * convKernel[k * convDim + c];
                        float cur = qkvRow[c];
                        acc += cur * convKernel[(taps - 1) * convDim + c];
                        convRow[c] = acc * Sigmoid(acc);
                    }

                    for (int k = 0; k < stateRows; k++)
                    {
                        int src = k + 1;
                        int dstOffset = convStateBase + k * convDim;
                        if (src < stateRows)
                        {
                            int srcOffset = convStateBase + src * convDim;
                            convStateData.Slice(srcOffset, convDim).CopyTo(convStateData.Slice(dstOffset, convDim));
                        }
                        else
                        {
                            qkvRow.CopyTo(convStateData.Slice(dstOffset, convDim));
                        }
                    }
                }
                else
                {
                    for (int c = 0; c < convDim; c++)
                    {
                        float acc = qkvRow[c] * convKernel[c];
                        convRow[c] = acc * Sigmoid(acc);
                    }
                }

                // L2-normalise Q and K per key head.
                for (int h = 0; h < nKHeads; h++)
                {
                    L2Normalise(convRow.AsSpan(h * keyHeadDim, keyHeadDim), eps);
                    L2Normalise(convRow.AsSpan(keyDim + h * keyHeadDim, keyHeadDim), eps);
                }

                int zRowOffset = (b * seq + t) * valueDim;
                int betaRowOffset = (b * seq + t) * nVHeads;
                int alphaRowOffset = (b * seq + t) * nVHeads;
                int outRowOffset = (b * seq + t) * valueDim;

                for (int hv = 0; hv < nVHeads; hv++)
                {
                    int kh = hv / groupSize;
                    int qBase = kh * keyHeadDim;
                    int kBase = keyDim + kh * keyHeadDim;
                    int vBase = 2 * keyDim + hv * valueHeadDim;

                    float xAlpha = alphaData[alphaRowOffset + hv] + ssmDt[hv];
                    float softplus = xAlpha > 20f ? xAlpha : MathF.Log(1f + MathF.Exp(xAlpha));
                    float decay = MathF.Exp(softplus * ssmA[hv]);
                    float beta = Sigmoid(betaData[betaRowOffset + hv]);

                    var state = deltaStateData.Slice(
                        deltaStateBase + hv * deltaLayerStride, deltaLayerStride);

                    ScaleInPlace(state, decay);

                    // pred[v] = sum_k state[k,v] * key[k]. Reduced with k as the outer
                    // loop, so each step is a contiguous row over v (vectorisable). Each
                    // v still adds its k terms in the same order as the column loop it
                    // replaces. `residual` doubles as the accumulator; it is free here.
                    residual.AsSpan().Clear();
                    for (int k = 0; k < keyHeadDim; k++)
                        MulAddRow(residual, state.Slice(k * valueHeadDim, valueHeadDim), convRow[kBase + k]);
                    for (int v = 0; v < valueHeadDim; v++)
                        residual[v] = (convRow[vBase + v] - residual[v]) * beta;

                    for (int k = 0; k < keyHeadDim; k++)
                        AddScaledRow(state.Slice(k * valueHeadDim, valueHeadDim), residual, convRow[kBase + k]);

                    // out[v] = sum_k state[k,v] * query[k], same row-wise reduction.
                    residual.AsSpan().Clear();
                    for (int k = 0; k < keyHeadDim; k++)
                        MulAddRow(residual, state.Slice(k * valueHeadDim, valueHeadDim), convRow[qBase + k]);

                    float sumSq = 0f;
                    int oBase = outRowOffset + hv * valueHeadDim;
                    for (int v = 0; v < valueHeadDim; v++)
                    {
                        float acc = residual[v];
                        float scaled = acc * qScale;
                        outputData[oBase + v] = scaled;
                        sumSq += scaled * scaled;
                    }

                    float inv = 1f / MathF.Sqrt(sumSq / valueHeadDim + eps);
                    for (int v = 0; v < valueHeadDim; v++)
                    {
                        float zg = zData[zRowOffset + hv * valueHeadDim + v];
                        float gate = zg * Sigmoid(zg);
                        outputData[oBase + v] *= inv * ssmNorm[v] * gate;
                    }
                }
            }
        }

        deltaCache?.Advance(seq);
        try
        {
            return _wout.Forward(output, workspace);
        }
        finally
        {
            output.Dispose();
        }
    }

    private static void L2Normalise(Span<float> values, float eps)
    {
        double sumSq = 0;
        for (int i = 0; i < values.Length; i++)
            sumSq += values[i] * values[i];
        float inv = 1f / MathF.Sqrt((float)sumSq + eps);
        for (int i = 0; i < values.Length; i++)
            values[i] *= inv;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static void CopyInto(Tensor<float> target, ReadOnlySpan<float> data, string name)
    {
        if (data.Length != target.ElementCount)
            throw new ArgumentException($"Expected {target.ElementCount} values for {name}, got {data.Length}.");
        data.CopyTo(target.Data);
    }

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in _wqkv.Parameters()) yield return p;
        foreach (var p in _wgate.Parameters()) yield return p;
        foreach (var p in _walpha.Parameters()) yield return p;
        foreach (var p in _wbeta.Parameters()) yield return p;
        foreach (var p in _wout.Parameters()) yield return p;
    }

    public void EnableQuantAwareTraining(QuantDType? target)
    {
        _wqkv.EnableQuantAwareTraining(target);
        _wgate.EnableQuantAwareTraining(target);
        _walpha.EnableQuantAwareTraining(target);
        _wbeta.EnableQuantAwareTraining(target);
        _wout.EnableQuantAwareTraining(target);
    }

    public void FreeFloatWeights()
    {
        _wqkv.FreeFloatWeight();
        _wgate.FreeFloatWeight();
        _walpha.FreeFloatWeight();
        _wbeta.FreeFloatWeight();
        _wout.FreeFloatWeight();
    }

    internal void SuppressWide()
    {
        if (_wqkv is InferenceLinearLayer qkv) qkv.SuppressWide();
        if (_wgate is InferenceLinearLayer gate) gate.SuppressWide();
        if (_walpha is InferenceLinearLayer alpha) alpha.SuppressWide();
        if (_wbeta is InferenceLinearLayer beta) beta.SuppressWide();
        if (_wout is InferenceLinearLayer o) o.SuppressWide();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _wqkv.Dispose();
        _wgate.Dispose();
        _walpha.Dispose();
        _wbeta.Dispose();
        _wout.Dispose();
        if (_ownsConvKernel) _convKernel.Dispose();
        if (_ownsSsmA) _ssmA.Dispose();
        if (_ownsSsmDt) _ssmDt.Dispose();
        if (_ownsSsmNorm) _ssmNorm.Dispose();
        _fallbackConvState.Dispose();
        _fallbackDeltaState.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(DeltaNetLayer));

    // Row helpers for the recurrence. Each element is computed with the same
    // operations, in the same order, as the scalar loop; the vector path only
    // changes how many elements are processed per instruction.

    /// <summary>x[i] *= s.</summary>
    internal static void ScaleInPlace(Span<float> x, float s)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var sv = new Vector<float>(s);
            for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
                (new Vector<float>(x.Slice(i)) * sv).CopyTo(x.Slice(i));
        }
        for (; i < x.Length; i++) x[i] *= s;
    }

    /// <summary>dst[i] += row[i] * s.</summary>
    internal static void MulAddRow(Span<float> dst, ReadOnlySpan<float> row, float s)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var sv = new Vector<float>(s);
            for (; i <= dst.Length - Vector<float>.Count; i += Vector<float>.Count)
                (new Vector<float>(dst.Slice(i)) + new Vector<float>(row.Slice(i)) * sv).CopyTo(dst.Slice(i));
        }
        for (; i < dst.Length; i++) dst[i] += row[i] * s;
    }

    /// <summary>dst[i] += s * x[i].</summary>
    internal static void AddScaledRow(Span<float> dst, ReadOnlySpan<float> x, float s)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var sv = new Vector<float>(s);
            for (; i <= dst.Length - Vector<float>.Count; i += Vector<float>.Count)
                (new Vector<float>(dst.Slice(i)) + sv * new Vector<float>(x.Slice(i))).CopyTo(dst.Slice(i));
        }
        for (; i < dst.Length; i++) dst[i] += s * x[i];
    }
}
