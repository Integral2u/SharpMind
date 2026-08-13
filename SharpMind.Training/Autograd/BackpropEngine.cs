using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Training.Autograd;

/// <summary>
/// Full backprop for the training transformer. A recording forward pass
/// (<see cref="ForwardAndRecord"/>) captures every activation the reverse pass
/// needs into a <see cref="ForwardContext"/>; the reverse pass
/// (<see cref="Backward"/>) traverses the graph backwards and accumulates
/// gradients into the shared <see cref="Parameter"/> instances via the
/// JigSaw-assembled <see cref="GradientMapping"/> kernels.
///
/// The engine runs its own forward (reusing the training transformer's float
/// layers) rather than <c>Transformer.Forward</c>, because backprop needs the
/// intermediate activations — Q/K/V, attention probabilities, norm inputs and
/// FFN pre-activations — that the inference path never exposes.
///
/// Supported architectures: decoder transformer with RoPE/ALiBi/NoPE, RMSNorm
/// or LayerNorm, dense or gated (SwiGLU/GeGLU) FFN, MHA/GQA/MQA attention.
/// Weight-tied LM head (the training models built by
/// <see cref="ModelFactory.CreateForTraining"/> are always tied). MoE FFN and
/// Gemma-3 post norms are not supported yet.
/// </summary>
public sealed class BackpropEngine : IDisposable
{
    private readonly Transformer _model;
    private readonly GradientMapping _mapping;
    private readonly Dictionary<Tensor<float>, Parameter> _paramsByTensor;
    private readonly ActivationKind _activation;
    private readonly bool _gemmaScale;
    private bool _disposed;

    /// <param name="model">Training transformer (float layers, weight-tied LM head).</param>
    /// <param name="mapping">Gradient kernels. Create with the same SharpMindConfig the model used.</param>
    /// <param name="parameters">
    /// The parameter instances the optimizer was built from (must be the SAME
    /// instances, captured once — never a fresh <c>model.Parameters()</c> call).
    /// </param>
    /// <param name="config">The SharpMindConfig the training transformer was built with.</param>
    public BackpropEngine(Transformer model, GradientMapping mapping, IReadOnlyList<Parameter> parameters, SharpMindConfig config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(mapping);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(config);

        _model = model;
        _mapping = mapping;
        _activation = config.Activation;
        _gemmaScale = config.Activation == ActivationKind.GELU && config.Gate == GateKind.GeGLU;

        _paramsByTensor = new Dictionary<Tensor<float>, Parameter>();
        foreach (var p in parameters)
            _paramsByTensor[p.Data] = p;
    }

    /// <summary>
    /// Recording forward: embedding → blocks → final norm → logits.
    /// Returns logits as a flat [Batch*SeqLen, VocabSize] tensor. The returned
    /// tensor and the ctx's tensors must be disposed by the caller.
    /// </summary>
    public Tensor<float> ForwardAndRecord(ForwardContext ctx, Tensor<int> tokenIds, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(ctx);
        ArgumentNullException.ThrowIfNull(tokenIds);
        cancellationToken.ThrowIfCancellationRequested();

        var cfg = _model.Config;
        int H = cfg.HiddenDim;
        int M = tokenIds.ElementCount;

        ctx.TokenIds = null; // caller owns the batch; do not double-dispose

        var emb = _model.ForwardEmbedding(tokenIds); // [Batch, SeqLen, HiddenDim]
        ctx.EmbeddingOut = emb;
        if (_gemmaScale)
            ScaleInPlace(emb, MathF.Sqrt(H));

        var x = emb;
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var block = _model.GetBlock(l) ?? throw new InvalidOperationException($"Missing block {l}.");
            var bc = new BlockContext();
            ctx.Blocks.Add(bc);
            x = ForwardBlock(x, block, bc);
        }

        cancellationToken.ThrowIfCancellationRequested();
        var finalNorm = _model.FinalNorm;
        ctx.FinalNormInput = x;
        var (normed, finalState) = finalNorm.ForwardWithState(x);
        ctx.FinalNormOut = normed;
        ctx.FinalNormState = finalState;

        var logits = _model.QuantAwareTrainingTarget is not null and not QuantDType.F32
            ? ProjectHeadQuantAware(normed, _model.EmbeddingWeight, M, cfg.VocabSize)
            : ProjectHead(normed, _model.EmbeddingWeight, M, cfg.VocabSize);
        ctx.Logits = logits;
        return logits;
    }

    /// <summary>
    /// Reverse traversal. <paramref name="dLogits"/> is the loss gradient
    /// w.r.t. logits, shape [Batch*SeqLen, VocabSize] (flat). Accumulates
    /// gradients into the parameters supplied at construction.
    /// </summary>
    public void Backward(ForwardContext ctx, Tensor<float> dLogits, Tensor<int> tokenIds, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(ctx);
        ArgumentNullException.ThrowIfNull(dLogits);
        ArgumentNullException.ThrowIfNull(tokenIds);
        cancellationToken.ThrowIfCancellationRequested();

        var cfg = _model.Config;
        int M = tokenIds.ElementCount;
        int H = cfg.HiddenDim;
        int V = cfg.VocabSize;
        if (dLogits.Rank != 2 || dLogits.Shape.Rows != M || dLogits.Shape.Cols != V)
            throw new ArgumentException($"dLogits must be [{M}, {V}], got {dLogits.Shape}.");

        // 1. LM head (weight-tied): dLogits @ head gives dFinalNormOut and
        //    accumulates the head-weight gradient into the embedding parameter.
        var embParam = Param(_model.EmbeddingWeight);
        using var normedFlat = ctx.FinalNormOut!.Reshape(M, H);
        var dFinalOut = _mapping.Linear(dLogits, normedFlat, embParam);

        // 2. Final norm backward.
        var dX = NormBackward(_model.FinalNorm, ctx.FinalNormState!, dFinalOut);
        dFinalOut.Dispose();

        // 3. Blocks, reversed.
        for (int l = cfg.NumLayers - 1; l >= 0; l--)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var block = _model.GetBlock(l) ?? throw new InvalidOperationException($"Missing block {l}.");
            dX = BackwardBlock(block, ctx.Blocks[l], dX);
        }

        // 4. Embedding backward (scatter-add into selected rows).
        if (_gemmaScale)
            ScaleInPlace(dX, MathF.Sqrt(H));
        using var flatIds = tokenIds.Reshape(M);
        _mapping.Embedding(dX, flatIds, embParam);
        dX.Dispose();
    }

    public void Dispose() { GC.SuppressFinalize(this); _disposed = true; }
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(BackpropEngine));

    // ── Forward ─────────────────────────────────────────────────────────────

    private Tensor<float> ForwardBlock(Tensor<float> x, TransformerBlock block, BlockContext bc)
    {
        if (block.PostAttnNorm is not null || block.PostFfnNorm is not null)
            throw new NotSupportedException("Backprop does not support Gemma-3 post-attention/post-FFN norms yet.");

        var (norm1Out, norm1State) = block.Norm1.ForwardWithState(x);
        bc.Norm1Out = norm1Out;
        bc.Norm1State = norm1State;

        var attnProj = ForwardAttention(norm1Out, block.Attention, bc);
        x.AddInPlace(attnProj);
        attnProj.Dispose();

        var (norm2Out, norm2State) = block.Norm2.ForwardWithState(x);
        bc.Norm2Out = norm2Out;
        bc.Norm2State = norm2State;

        var ffnOut = ForwardFfn(norm2Out, block.Ffn, bc);
        x.AddInPlace(ffnOut);
        ffnOut.Dispose();

        return x;
    }

    private unsafe Tensor<float> ForwardAttention(Tensor<float> norm1Out, AttentionLayer attn, BlockContext bc)
    {
        var cfg = _model.Config;
        int B = norm1Out.Shape[0], S = norm1Out.Shape[1];
        int numH = cfg.NumHeads, numKv = cfg.NumKvHeads, D = cfg.HeadDim;
        int qDim = numH * D, kvDim = numKv * D;
        float scale = 1f / MathF.Sqrt(D);

        var q = attn.Wq.Forward(norm1Out); // [B,S,qDim]
        var k = attn.Wk.Forward(norm1Out); // [B,S,kvDim]
        var v = attn.Wv.Forward(norm1Out); // [B,S,kvDim]

        using var qr = q.Reshape(B, S, numH, D);
        using var kr = k.Reshape(B, S, numKv, D);
        attn.PositionalEncoder.ApplyBatched(qr, 0);
        attn.PositionalEncoder.ApplyBatched(kr, 0);

        var probs = Tensor<float>.Zeros(B, numH, S, S);
        var attnOut = new Tensor<float>(B, S, qDim);

        // DataPtr addresses (all five tensors are freshly allocated, offset 0);
        // captured as long so the Parallel.For body can rebuild pointers.
        long qAddr = (long)q.DataPtr;
        long kAddr = (long)k.DataPtr;
        long vAddr = (long)v.DataPtr;
        long pAddr = (long)probs.DataPtr;
        long oAddr = (long)attnOut.DataPtr;

        // Heads are independent: partition the flattened (batch × head) domain
        // across threads. Each worker writes disjoint probs rows and attnOut
        // columns, so no shared-write racing occurs. The D-dot products are FMA
        // vectorised and the causal softmax uses the SIMD FastExp when AVX2 is
        // available; otherwise the exact scalar path is used.
        bool useSimd = Avx2.IsSupported;
        int totalHeads = B * numH;
        Parallel.For(0, totalHeads, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, headIdx =>
        {
            float* qData = (float*)qAddr;
            float* kData = (float*)kAddr;
            float* vData = (float*)vAddr;
            float* pData = (float*)pAddr;
            float* oData = (float*)oAddr;

            int b = headIdx / numH;
            int h = headIdx % numH;
            int kvHead = h / cfg.KvGroupSize;

            for (int i = 0; i < S; i++)
            {
                // scores = q_i · k_j * scale, causal over j <= i
                float maxScore = float.NegativeInfinity;
                int qBaseRow = ((b * S + i) * numH + h) * D;
                for (int j = 0; j <= i; j++)
                {
                    int qBase = qBaseRow;
                    int kBase = ((b * S + j) * numKv + kvHead) * D;
                    float s = 0f;
                    if (useSimd)
                    {
                        int d = 0;
                        var vSum = Vector256<float>.Zero;
                        for (; d <= D - 8; d += 8)
                        {
                            var qv = Vector256.LoadUnsafe(ref qData[qBase + d]);
                            var kv = Vector256.LoadUnsafe(ref kData[kBase + d]);
                            vSum = Avx.Add(vSum, Avx.Multiply(qv, kv));
                        }
                        s = MathHelpers.HSum256_Avx(vSum);
                        for (; d < D; d++)
                            s += qData[qBase + d] * kData[kBase + d];
                    }
                    else
                    {
                        for (int d = 0; d < D; d++)
                            s += qData[qBase + d] * kData[kBase + d];
                    }
                    s *= scale;
                    pData[((b * numH + h) * S + i) * S + j] = s;
                    if (s > maxScore) maxScore = s;
                }

                // softmax over j <= i
                float sum = 0f;
                int pBase = ((b * numH + h) * S + i) * S;
                if (useSimd)
                {
                    int j = 0;
                    var vMax = Vector256.Create(maxScore);
                    for (; j <= i - 7; j += 8)
                    {
                        var shifted = Avx.Subtract(Vector256.LoadUnsafe(ref pData[pBase + j]), vMax);
                        var p = ActivationKernels.FastExp(shifted);
                        Vector256.StoreUnsafe(p, ref pData[pBase + j]);
                        sum += MathHelpers.HSum256_Avx(p);
                    }
                    for (; j <= i; j++)
                    {
                        float p = FastExpScalar(pData[pBase + j] - maxScore);
                        pData[pBase + j] = p;
                        sum += p;
                    }
                }
                else
                {
                    for (int j = 0; j <= i; j++)
                    {
                        float p = MathF.Exp(pData[pBase + j] - maxScore);
                        pData[pBase + j] = p;
                        sum += p;
                    }
                }
                float inv = 1f / sum;
                if (useSimd)
                {
                    int j0 = 0;
                    var vInv = Vector256.Create(inv);
                    for (; j0 <= i - 7; j0 += 8)
                    {
                        var pow = Avx.Multiply(Vector256.LoadUnsafe(ref pData[pBase + j0]), vInv);
                        Vector256.StoreUnsafe(pow, ref pData[pBase + j0]);
                    }
                    for (; j0 <= i; j0++)
                        pData[pBase + j0] *= inv;
                }
                else
                {
                    for (int j0 = 0; j0 <= i; j0++)
                        pData[pBase + j0] *= inv;
                }

                // attnOut = probs · v
                int oBase = (b * S + i) * qDim + h * D;
                for (int d = 0; d < D; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j <= i; j++)
                        acc += pData[((b * numH + h) * S + i) * S + j] * vData[((b * S + j) * numKv + kvHead) * D + d];
                    oData[oBase + d] = acc;
                }
            }
        });

        var attnProj = attn.Wo.Forward(attnOut);

        bc.Q = q;
        bc.K = k;
        bc.V = v;
        bc.AttnProbs = probs;
        bc.AttnOut = attnOut;
        bc.AttnProjOut = attnProj;
        return attnProj;
    }

    private Tensor<float> ForwardFfn(Tensor<float> norm2Out, FfnLayer ffn, BlockContext bc)
    {
        var cfg = _model.Config;
        int ffnDim = cfg.FfnDim;
        int total = norm2Out.ElementCount / norm2Out.Shape[^1];
        var acts = ffn.Activation;

        switch (ffn)
        {
            case DenseFfnLayer:
            {
                var hidden = ffn.W1Layer!.Forward(norm2Out);
                var acted = acts.Activate(hidden);
                var out_ = ffn.W2Layer!.Forward(acted);
                bc.FfnHidden = hidden;
                bc.FfnActOut = acted;
                bc.FfnOut = out_;
                return out_;
            }
            case GatedFfnLayer:
            {
                using var fused = ffn.WGated!.Forward(norm2Out); // [B,S,2*f]
                using var flat = fused.Reshape(total, 2 * ffnDim);
                var gate = Tensor<float>.Zeros(total, ffnDim);
                var up = Tensor<float>.Zeros(total, ffnDim);
                var acted = Tensor<float>.Zeros(total, ffnDim);
                for (int i = 0; i < total; i++)
                {
                    var row = flat.RowSpan(i);
                    row[..ffnDim].CopyTo(gate.RowSpan(i));
                    row[ffnDim..].CopyTo(up.RowSpan(i));
                    acts.ApplyGate(gate.RowSpan(i), up.RowSpan(i), acted.RowSpan(i));
                }
                var out_ = ffn.WDown!.Forward(acted);
                bc.FfnGate = gate;
                bc.FfnUp = up;
                bc.FfnActOut = acted;
                var out3 = out_.Reshape(norm2Out.Shape.Dims[0], norm2Out.Shape.Dims[1], cfg.HiddenDim);
                out_.Dispose();
                bc.FfnOut = out3;
                return out3;
            }
            default:
                throw new NotSupportedException($"Backprop does not support FFN kind {ffn.GetType().Name} yet.");
        }
    }

    // ── Backward ────────────────────────────────────────────────────────────

    private Tensor<float> BackwardBlock(TransformerBlock block, BlockContext bc, Tensor<float> dX)
    {
        // x_out = x1 + ffnResid ;  x1 = x_in + attnResid
        var dFfnResid = dX;

        var dNorm2Out = FfnBackward(block.Ffn, bc, dFfnResid);
        var dNorm2Input = NormBackward(block.Norm2, bc.Norm2State!, dNorm2Out);
        dNorm2Out.Dispose();

        dX.AddInPlace(dNorm2Input); // + direct path of x_out → x1
        dNorm2Input.Dispose();

        var dNorm1Out = AttentionBackward(block.Attention, bc, dX);
        var dNorm1Input = NormBackward(block.Norm1, bc.Norm1State!, dNorm1Out);
        dNorm1Out.Dispose();

        dX.AddInPlace(dNorm1Input); // + direct path of x1 → x_in
        dNorm1Input.Dispose();

        return dX;
    }

    private Tensor<float> AttentionBackward(AttentionLayer attn, BlockContext bc, Tensor<float> dAttnProj)
    {
        var cfg = _model.Config;
        int B = bc.AttnProbs!.Shape.Dims[0];
        int S = bc.AttnProbs.Shape.Dims[2];
        int numH = cfg.NumHeads, numKv = cfg.NumKvHeads, D = cfg.HeadDim;
        int qDim = numH * D, kvDim = numKv * D, H = cfg.HiddenDim;
        float scale = 1f / MathF.Sqrt(D);
        int M = B * S;

        // Wo backward
        using var dAttnProjFlat = dAttnProj.Reshape(M, H);
        using var attnOutFlat = bc.AttnOut!.Reshape(M, qDim);
        var dAttnOut = LinearBackward(dAttnProjFlat, attnOutFlat, attn.Wo.Weight, BiasParam(attn.Wo));

        // Per-head attention backward → dQ/dK/dV
        // Parallelised over batch rows: each row writes a disjoint block of the
        // dQ/dK/dV row-space, so no shared-write racing occurs even when multiple
        // heads share a KV head (the per-row head loop stays sequential, which is
        // exactly how the sequential version accumulated the shared KV region).
        var dQ = new Tensor<float>(M, qDim);
        var dK = new Tensor<float>(M, kvDim);
        var dV = new Tensor<float>(M, kvDim);

        Parallel.For(0, B, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, b =>
        {
            for (int h = 0; h < numH; h++)
            {
                int kvHead = h / cfg.KvGroupSize;
                using var dOutHead = SliceHead(dAttnOut, b, h, B, S, qDim, D);
                using var qHead = SliceHead(bc.Q!, b, h, B, S, qDim, D);
                using var kHead = SliceHead(bc.K!, b, kvHead, B, S, kvDim, D);
                using var vHead = SliceHead(bc.V!, b, kvHead, B, S, kvDim, D);
                using var probsHead = SliceProbs(bc.AttnProbs, b, h, S);

                var grads = _mapping.Attention(dOutHead, qHead, kHead, vHead, probsHead, scale);
                using (grads.DQ) using (grads.DK) using (grads.DV)
                {
                    AddHead(dQ, grads.DQ, b, h, B, S, qDim, D);
                    AddHead(dK, grads.DK, b, kvHead, B, S, kvDim, D);
                    AddHead(dV, grads.DV, b, kvHead, B, S, kvDim, D);
                }
            }
        });
        dAttnOut.Dispose();

        // RoPE backward (inverse rotation), in place on dQ/dK
        using (var dQr = dQ.Reshape(B, S, numH, D))
            attn.PositionalEncoder.ApplyBatchedBackward(dQr, 0);
        using (var dKr = dK.Reshape(B, S, numKv, D))
            attn.PositionalEncoder.ApplyBatchedBackward(dKr, 0);

        // Wq/Wk/Wv backward
        using var norm1OutFlat = bc.Norm1Out!.Reshape(M, H);
        var dNorm1 = LinearBackward(dQ, norm1OutFlat, attn.Wq.Weight, BiasParam(attn.Wq));
        using var dKpath = LinearBackward(dK, norm1OutFlat, attn.Wk.Weight, BiasParam(attn.Wk));
        using var dVpath = LinearBackward(dV, norm1OutFlat, attn.Wv.Weight, BiasParam(attn.Wv));
        dQ.Dispose();
        dK.Dispose();
        dV.Dispose();
        dNorm1.AddInPlace(dKpath);
        dNorm1.AddInPlace(dVpath);
        return dNorm1;
    }

    private Tensor<float> FfnBackward(FfnLayer ffn, BlockContext bc, Tensor<float> dFfn)
    {
        var cfg = _model.Config;
        int H = cfg.HiddenDim;
        int ffnDim = cfg.FfnDim;
        int rows = dFfn.Shape.Rows;

        switch (ffn)
        {
            case DenseFfnLayer:
            {
                using var actFlat = bc.FfnActOut!.Reshape(rows, ffnDim);
                var dAct = LinearBackward(dFfn, actFlat, ffn.W2Layer!.Weight, BiasParam(ffn.W2Layer));
                using var hiddenFlat = bc.FfnHidden!.Reshape(rows, ffnDim);
                var dHidden = ActivationBackward(dAct, hiddenFlat);
                dAct.Dispose();
                using var norm2Flat = bc.Norm2Out!.Reshape(rows, H);
                return LinearBackward(dHidden, norm2Flat, ffn.W1Layer!.Weight, BiasParam(ffn.W1Layer));
            }
            case GatedFfnLayer:
            {
                using var actFlat = bc.FfnActOut!.Reshape(rows, ffnDim);
                var dAct = LinearBackward(dFfn, actFlat, ffn.WDown!.Weight, BiasParam(ffn.WDown));
                var dGate = Tensor<float>.Zeros(rows, ffnDim);
                var dUp = Tensor<float>.Zeros(rows, ffnDim);
                for (int i = 0; i < rows; i++)
                {
                    var gate = bc.FfnGate!.RowSpan(i);
                    var up = bc.FfnUp!.RowSpan(i);
                    var da = dAct.RowSpan(i);
                    var dg = dGate.RowSpan(i);
                    var du = dUp.RowSpan(i);
                    for (int d = 0; d < ffnDim; d++)
                    {
                        // out = gateValue(g) * u
                        du[d] = da[d] * GateValue(gate[d]);
                        dg[d] = da[d] * GateDerivative(gate[d]) * up[d];
                    }
                }
                dAct.Dispose();

                var dFused = Tensor<float>.Zeros(rows, 2 * ffnDim);
                for (int i = 0; i < rows; i++)
                {
                    dGate.RowSpan(i).CopyTo(dFused.RowSpan(i)[..ffnDim]);
                    dUp.RowSpan(i).CopyTo(dFused.RowSpan(i)[ffnDim..]);
                }
                dGate.Dispose();
                dUp.Dispose();
                using var norm2Flat = bc.Norm2Out!.Reshape(rows, H);
                var dNorm2 = LinearBackward(dFused, norm2Flat, ffn.WGated!.Weight, BiasParam(ffn.WGated));
                dFused.Dispose();
                return dNorm2;
            }
            default:
                throw new NotSupportedException($"Backprop does not support FFN kind {ffn.GetType().Name} yet.");
        }
    }

    private Tensor<float> NormBackward(NormLayer norm, NormLayerState state, Tensor<float> dOut)
    {
        int T = state.Rows;
        int D = state.Dim;

        if (norm is RmsNormLayer)
        {
            var rmsInv = new float[T];
            var xNorm = Tensor<float>.Zeros(T, D);
            for (int i = 0; i < T; i++)
            {
                float ri = state.GetScalarParam(i);
                rmsInv[i] = ri;
                var input = state.GetInput(i);
                var dst = xNorm.RowSpan(i);
                for (int d = 0; d < D; d++)
                    dst[d] = input[d] * ri;
            }
            var dInput = _mapping.RMSNorm(dOut, xNorm, rmsInv, Param(norm.NormWeight));
            xNorm.Dispose();
            return dInput;
        }

        // LayerNorm
        if (norm.NormBias is null)
            throw new InvalidOperationException("LayerNorm backward requires a bias parameter.");
        var bias = Param(norm.NormBias);
        var inputTensor = Tensor<float>.Zeros(T, D);
        for (int i = 0; i < T; i++)
            state.GetInput(i).CopyTo(inputTensor.RowSpan(i));
        var result = _mapping.LayerNorm(dOut, inputTensor, Param(norm.NormWeight), bias, norm.Eps);
        inputTensor.Dispose();
        return result;
    }

    // ── Small helpers ───────────────────────────────────────────────────────

    private Tensor<float> ActivationBackward(Tensor<float> dOutput, Tensor<float> preAct)
        => _activation switch
        {
            ActivationKind.GELU => _mapping.ActivationGELU(dOutput, preAct),
            ActivationKind.SiLU => _mapping.ActivationSiLU(dOutput, preAct),
            ActivationKind.ReLU => ReLUBackward(dOutput, preAct),
            _ => throw new NotSupportedException($"No backward for activation {_activation}."),
        };

    private static Tensor<float> ReLUBackward(Tensor<float> dOutput, Tensor<float> preAct)
    {
        var res = Tensor<float>.Zeros(dOutput.Shape.Dims.ToArray());
        var d = dOutput.Data;
        var x = preAct.Data;
        var r = res.Data;
        for (int i = 0; i < d.Length; i++)
            r[i] = x[i] > 0f ? d[i] : 0f;
        return res;
    }

    private float GateValue(float g)
    {
        float sig = 1f / (1f + MathF.Exp(-g));
        return _activation switch
        {
            ActivationKind.SiLU => g * sig,
            ActivationKind.GELU => 0.5f * g * (1f + MathF.Tanh(Sqrt2PiInv * (g + GeluCoeff * g * g * g))),
            _ => throw new NotSupportedException($"No gated value for activation {_activation}."),
        };
    }

    private float GateDerivative(float g)
    {
        float sig = 1f / (1f + MathF.Exp(-g));
        if (_activation == ActivationKind.SiLU)
            return sig * (1f + g * (1f - sig));
        if (_activation == ActivationKind.GELU)
        {
            float g3 = g * g * g;
            float z = Sqrt2PiInv * (g + GeluCoeff * g3);
            float tanh = MathF.Tanh(z);
            float sech2 = 1f - tanh * tanh;
            return 0.5f * (1f + tanh) + 0.5f * g * sech2 * Sqrt2PiInv * (1f + 3f * GeluCoeff * g * g);
        }
        throw new NotSupportedException($"No gated derivative for activation {_activation}.");
    }

    private const float Sqrt2PiInv = 0.7978845608028654f;
    private const float GeluCoeff = 0.044715f;

    private static Tensor<float> ProjectHead(Tensor<float> normed, Tensor<float> head, int M, int V)
    {
        int H = normed.Shape[^1];
        var logits = new Tensor<float>(M, V);
        var n = normed.Data;
        var w = head.Data;
        var l = logits.Data;
        for (int m = 0; m < M; m++)
        {
            for (int v = 0; v < V; v++)
            {
                float s = 0f;
                for (int h = 0; h < H; h++)
                    s += n[m * H + h] * w[v * H + h];
                l[m * V + v] = s;
            }
        }
        return logits;
    }

    /// <summary>
    /// Fake-quantized head projection used during quantization-aware training:
    /// quantizes the tied head weight to the active target and runs the matching
    /// matmul, so the head can convert to the quantized dtype losslessly at export.
    /// Block formats require both dimensions to be multiples of 32 (validated at
    /// enable time by the encoder path).
    /// </summary>
    private unsafe Tensor<float> ProjectHeadQuantAware(Tensor<float> normed, Tensor<float> head, int M, int V)
    {
        var target = _model.QuantAwareTrainingTarget ?? throw new InvalidOperationException("QAT not active.");
        int H = normed.Shape[^1];
        var logits = new Tensor<float>(M, V);
        var raw = TensorQuantizer.Quantize(head.Data, [V, H], target);
        var fn = QuantizationFactory.Create().QuantizedMatMulOpFor(target);
        fixed (byte* rawPtr = raw)
            fn(normed.DataPtr, rawPtr, logits.DataPtr, M, H, V);
        return logits;
    }

    private static Tensor<float> SliceHead(Tensor<float> t, int b, int h, int B, int S, int cols, int D)
    {
        var res = Tensor<float>.Zeros(S, D);
        var src = t.Data;
        var dst = res.Data;
        for (int i = 0; i < S; i++)
        {
            int srcBase = (b * S + i) * cols + h * D;
            src.Slice(srcBase, D).CopyTo(dst.Slice(i * D, D));
        }
        return res;
    }

    private static Tensor<float> SliceProbs(Tensor<float> probs, int b, int h, int S)
    {
        int numH = probs.Shape.Dims[1];
        var res = Tensor<float>.Zeros(S, S);
        var src = probs.Data;
        var dst = res.Data;
        for (int i = 0; i < S; i++)
        {
            int srcBase = ((b * numH + h) * S + i) * S;
            src.Slice(srcBase, S).CopyTo(dst.Slice(i * S, S));
        }
        return res;
    }

    private static void AddHead(Tensor<float> dst, Tensor<float> src, int b, int h, int B, int S, int cols, int D)
    {
        var d = dst.Data;
        var s = src.Data;
        for (int i = 0; i < S; i++)
        {
            int dstBase = (b * S + i) * cols + h * D;
            int srcBase = i * D;
            for (int j = 0; j < D; j++)
                d[dstBase + j] += s[srcBase + j];
        }
    }

    private static void ScaleInPlace(Tensor<float> t, float scalar)
    {
        var data = t.Data;
        for (int i = 0; i < data.Length; i++)
            data[i] *= scalar;
    }

    /// <summary>
    /// Scalar fast-exp (same degree-6 polynomial as <see cref="ActivationKernels.FastExp"/>),
    /// used for the softmax tail lanes when AVX2 is unavailable.
    /// </summary>
    private static float FastExpScalar(float x)
    {
        x = Math.Clamp(x, -88f, 88f);
        var z = x * 1.4426950408889634f;
        var n = MathF.Round(z);
        var r = z - n;
        var u = r * 0.6931471805599453f;
        var p = 1f + u * (1f + u * (0.5f + u * ((1f / 6f) + u * ((1f / 24f) + u * ((1f / 120f) + u * (1f / 720f))))));
        return p * MathF.Pow(2f, n);
    }

    private Parameter Param(Tensor<float> tensor)
        => _paramsByTensor.TryGetValue(tensor, out var p)
            ? p
            : throw new InvalidOperationException($"No parameter wraps tensor {tensor}.");

    private Parameter? BiasParam(LinearLayer layer) => layer.Bias is null ? null : Param(layer.Bias);

    /// <summary>
    /// GradientMapping.Linear expects its weight as [Out, In] (PyTorch layout), but
    /// LinearLayer stores weights as [In, Out] (transposed at matmul time). Feed the
    /// kernel a transposed weight through a throwaway parameter, then scatter the
    /// produced [Out, In] weight gradient back into the original parameter in place.
    /// The embedding/LM-head weight is already [Vocab, Hidden] = [Out, In], so the
    /// head path calls _mapping.Linear directly with the real parameter.
    /// </summary>
private Tensor<float> LinearBackward(Tensor<float> dOutput, Tensor<float> input, Tensor<float> weight, Parameter? bias)
    {
        using var wT = weight.Transpose(); // [Out, In]
        using var tmpW = new Parameter("__transposed__", wT);
        var dInput = _mapping.Linear(dOutput, input, tmpW, bias);

        var orig = Param(weight);
        int Out = weight.Shape.Cols; // weight is [In, Out]
        int In = weight.Shape.Rows;
        var src = tmpW.Grad.Data;
        var dst = orig.Grad.Data;
        for (int o = 0; o < Out; o++)
        {
            var srcRow = src.Slice(o * In, In);
            for (int i = 0; i < In; i++)
                dst[i * Out + o] += srcRow[i];
        }
        return dInput;
    }
}
