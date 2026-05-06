using JigSawDotNet;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Attention;

public abstract class AttentionLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(Attention)}.{nameof(AttentionKernels)}";

    protected readonly ModelConfig Config;
    protected readonly LinearLayer Wq;
    protected readonly LinearLayer Wk;
    protected readonly LinearLayer Wv;
    protected readonly LinearLayer Wo;
    protected readonly RoPE Rope;
    private bool _disposed;

    protected AttentionLayer(ModelConfig config)
    {
        Config = config;
        int kvDim = config.NumKvHeads * config.HeadDim;
        Wq = new LinearLayer(config.HiddenDim, config.HiddenDim);
        Wk = new LinearLayer(config.HiddenDim, kvDim);
        Wv = new LinearLayer(config.HiddenDim, kvDim);
        Wo = new LinearLayer(config.HiddenDim, config.HiddenDim);
        Rope = new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta);
    }

    [PuzzleCornerPiece(SharpMindConfig.KeyAttention,
        SharpMindConfig.ValMhaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMhaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValGqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValGqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValMqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar))]
    public abstract unsafe void ScaledDotProduct(float* q, float* k, float* v, float* o, int seqLen, int kvLen, int headDim, float scale, bool causal);

    public Tensor<float> Forward(
        Tensor<float> x,
        TensorOps ops,
        int positionOffset = 0,
        bool causal = true,
        KVCache? cache = null)
    {
        ThrowIfDisposed();
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hidden = x.Shape[2];
        int numH = Config.NumHeads;
        int numKv = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int kvDim = numKv * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        using var q = Wq.Forward(x, ops);
        using var k = Wk.Forward(x, ops);
        using var v = Wv.Forward(x, ops);

        using var qr = q.Reshape(batch, seqLen, numH, headDim);
        using var kr = k.Reshape(batch, seqLen, numKv, headDim);
        Rope.ApplyBatched(qr, positionOffset);
        Rope.ApplyBatched(kr, positionOffset);

        if (cache != null)
        {
            cache.Update(k, v, numKv, headDim);
        }

        var output = new Tensor<float>(batch, seqLen, hidden);
        int effectiveKvLen = cache != null ? cache.CurrentPosition : seqLen;

        unsafe
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numH; h++)
                {
                    int kvHead = h / Config.KvGroupSize;
                    float* pQ = qr.DataPtr + (long)(b * seqLen * numH + h) * headDim;
                    
                    float* pK = cache != null 
                        ? cache.Keys.DataPtr + (long)b * (numKv * Config.MaxSeqLen * headDim) 
                                            + (long)kvHead * (Config.MaxSeqLen * headDim)
                        : kr.DataPtr + (long)(b * seqLen * numKv + kvHead) * headDim;
                    
                    float* pV = cache != null 
                        ? cache.Values.DataPtr + (long)b * (numKv * Config.MaxSeqLen * headDim) 
                                              + (long)kvHead * (Config.MaxSeqLen * headDim)
                        : v.DataPtr + (long)(b * seqLen * kvDim + kvHead * headDim);
                        
                    float* pO = output.DataPtr + (long)(b * seqLen * hidden + h * headDim);
                    
                    ScaledDotProduct(pQ, pK, pV, pO, seqLen, effectiveKvLen, headDim, scale, causal);
                }
            }
        }

        var projected = Wo.Forward(output, ops);
        output.Dispose();
        return projected;
    }

    public (Tensor<float> Output, AttentionLayerState State) ForwardWithState(Tensor<float> x, TensorOps ops, int positionOffset = 0)
    {
        var output = Forward(x, ops, positionOffset);
        var state = new AttentionLayerState { Input = x, Output = output };
        return (output, state);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput, AttentionLayerState state, TensorOps ops)
    {
        // Simplified: propagate gradient through output projection
        // Full backward requires full attention kernel gradients - stub for now
        using var wOutT = TensorOps.Transpose(Wo.Weight);
        var dHidden = ops.MatMul(gradOutput, wOutT);
        using var wQT = TensorOps.Transpose(Wq.Weight);
        var gradInput = ops.MatMul(dHidden, wQT);
        dHidden.Dispose();
        wOutT.Dispose();
        wQT.Dispose();
        return gradInput;
    }

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in Wq.Parameters()) yield return p;
        foreach (var p in Wk.Parameters()) yield return p;
        foreach (var p in Wv.Parameters()) yield return p;
        foreach (var p in Wo.Parameters()) yield return p;
    }

    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); }
        _disposed = true;
    }
    ~AttentionLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}

public class AttentionLayerState
{
    public Tensor<float> Input { get; init; } = null!;
    public Tensor<float> Output { get; init; } = null!;
}