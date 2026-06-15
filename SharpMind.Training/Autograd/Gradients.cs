using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.Autograd;

/// <summary>
/// Static backward (gradient) functions for every operation in a transformer.
/// Each method takes the upstream gradient and the saved activations from the
/// forward pass, and returns the downstream gradient while accumulating
/// parameter gradients into the supplied <see cref="Parameter"/> objects.
///
/// All gradients are *added* into parameter buffers (not overwritten) so that
/// gradient accumulation across multiple batches works correctly.
///
/// Math conventions:
///   dL/dX is written as <c>dX</c> in parameter names.
///   Shapes follow PyTorch convention: [Batch, ..., Features].
/// </summary>
public static class Gradients
{
    
    // Cross-entropy + softmax combined backward
    

    /// <summary>
    /// Computes dL/dLogits for the combined softmax + cross-entropy loss.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> CrossEntropySoftmax(
        Tensor<float> logits,  // [T, VocabSize] flat
        Tensor<int>   labels,  // [T] flat
        int           ignoreId = -100)
    {
        int T       = logits.Shape.Rows;
        int V       = logits.Shape.Cols;
        var dLogits = new Tensor<float>(T, V);

        int realCount = 0;
        for (int t = 0; t < T; t++)
            if (labels[t] != ignoreId) realCount++;

        float scale = realCount > 0 ? 1f / realCount : 0f;

        for (int t = 0; t < T; t++)
        {
            if (labels[t] == ignoreId) continue;

            ReadOnlySpan<float> logRow = logits.RowSpan(t);
            Span<float>         dRow   = dLogits.RowSpan(t);

            // Numerically stable softmax
            float max = logRow[0];
            for (int v = 1; v < V; v++) if (logRow[v] > max) max = logRow[v];

            float sum = 0f;
            for (int v = 0; v < V; v++) { dRow[v] = MathF.Exp(logRow[v] - max); sum += dRow[v]; }
            float inv = 1f / sum;
            for (int v = 0; v < V; v++) dRow[v] *= inv;  // now contains softmax probs

            // Subtract 1 at the target class, scale by 1/N
            dRow[labels[t]] -= 1f;
            for (int v = 0; v < V; v++) dRow[v] *= scale;
        }

        return dLogits;
    }

    
    // Linear backward  y = x @ W^T
    // W: [OutFeatures, InFeatures]   x: [B, InFeatures]   y: [B, OutFeatures]
    // dL/dx = dL/dy @ W              [B, InFeatures]
    // dL/dW = dL/dy^T @ x            [OutFeatures, InFeatures]
    // dL/db = sum(dL/dy, axis=0)     [OutFeatures]
    

    /// <summary>
    /// Returns dInput [B, InFeatures] and accumulates gradients into weight/bias parameters.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Tensor<float> Linear(
        Tensor<float> dOutput,   // [B, OutFeatures]
        Tensor<float> input,     // [B, InFeatures]
        Parameter     weight,    // [OutFeatures, InFeatures]
        Parameter?    bias = null)
    {
        int B   = dOutput.Shape.Rows;
        int Out = dOutput.Shape.Cols;
        int In  = input.Shape.Cols;

        var dInput = new Tensor<float>(B, In);

        // dInput = dOutput @ W   (W is [Out, In])
        fixed (float* pDOut = dOutput.Data, pW = weight.Data.Data, pDIn = dInput.Data)
        {
            for (int b = 0; b < B; b++)
            {
                float* dOutRow = pDOut + (long)b * Out;
                float* dInRow  = pDIn  + (long)b * In;
                for (int i = 0; i < In; i++)
                {
                    float sum = 0f;
                    for (int o = 0; o < Out; o++)
                        sum += dOutRow[o] * pW[(long)o * In + i];
                    dInRow[i] += sum;
                }
            }
        }

        // dW += dOutput^T @ input   accumulated into weight.Grad
        fixed (float* pDOut = dOutput.Data, pIn = input.Data, pDW = weight.Grad.Data)
        {
            for (int o = 0; o < Out; o++)
            {
                float* dWRow = pDW + (long)o * In;
                for (int b = 0; b < B; b++)
                {
                    float dOutOB = pDOut[(long)b * Out + o];
                    float* inRow = pIn + (long)b * In;
                    for (int i = 0; i < In; i++)
                        dWRow[i] += dOutOB * inRow[i];
                }
            }
        }

        // db += sum(dOutput, axis=0)
        if (bias is not null)
        {
            Span<float> dBias = bias.Grad.Data;
            for (int b = 0; b < B; b++)
            {
                ReadOnlySpan<float> dRow = dOutput.RowSpan(b);
                for (int o = 0; o < Out; o++) dBias[o] += dRow[o];
            }
        }

        return dInput;
    }

    
    // RMSNorm backward
    // y = x * rmsInv * w
    // dL/dw = sum_batch(dL/dy * xNorm)
    // dL/dx[i] = rmsInv * (dL/dy[i]*w[i] - xNorm[i] * dot(dL/dy*w, xNorm)/n)
    

    /// <summary>
    /// Returns dInput [T, D] and accumulates into weight gradient.
    /// <paramref name="rmsInv"/> is the per-row 1/rms value saved during forward.
    /// <paramref name="xNorm"/> is the normalised input (x * rmsInv) saved during forward.
    /// </summary>
    public static Tensor<float> RMSNorm(
        Tensor<float> dOutput,  // [T, D]
        Tensor<float> xNorm,    // [T, D]  x * rmsInv (saved from forward)
        float[]       rmsInv,   // [T]
        Parameter     weight)   // [D]
    {
        int T = dOutput.Shape.Rows;
        int D = dOutput.Shape.Cols;
        var dInput = new Tensor<float>(T, D);

        ReadOnlySpan<float> w     = weight.Data.Data;
        Span<float>         dw    = weight.Grad.Data;

        for (int t = 0; t < T; t++)
        {
            ReadOnlySpan<float> dy   = dOutput.RowSpan(t);
            ReadOnlySpan<float> xn   = xNorm.RowSpan(t);
            Span<float>         dxRow = dInput.RowSpan(t);
            float               ri   = rmsInv[t];

            // Accumulate weight gradient: dw += dy * xNorm (summed over batch)
            for (int d = 0; d < D; d++) dw[d] += dy[d] * xn[d];

            // dot(dy * w, xNorm) / D
            float dot = 0f;
            for (int d = 0; d < D; d++) dot += dy[d] * w[d] * xn[d];
            dot /= D;

            // dInput[t,d] = rmsInv * (dy[d]*w[d] - xNorm[d] * dot)
            for (int d = 0; d < D; d++)
                dxRow[d] = ri * (dy[d] * w[d] - xn[d] * dot);
        }

        return dInput;
    }

    
    // LayerNorm backward
    // y = (x - mean) / std * w + b
    

    public static Tensor<float> LayerNorm(
        Tensor<float> dOutput,
        Tensor<float> input,
        Parameter     weight,
        Parameter     bias,
        float         eps = 1e-5f)
    {
        int T = dOutput.Shape.Rows;
        int D = dOutput.Shape.Cols;
        var dInput = new Tensor<float>(T, D);

        ReadOnlySpan<float> w  = weight.Data.Data;
        Span<float>         dw = weight.Grad.Data;
        Span<float>         db = bias.Grad.Data;

        for (int t = 0; t < T; t++)
        {
            ReadOnlySpan<float> x  = input.RowSpan(t);
            ReadOnlySpan<float> dy = dOutput.RowSpan(t);
            Span<float>         dx = dInput.RowSpan(t);

            float mean = 0f;
            foreach (float v in x) mean += v;
            mean /= D;

            float var = 0f;
            foreach (float v in x) { float d = v - mean; var += d * d; }
            var /= D;

            float invStd = 1f / MathF.Sqrt(var + eps);

            // xhat = (x - mean) * invStd
            // dy_dw = dy * xhat  → dw gradient
            // db = dy
            float dyDotXhat = 0f, dySum = 0f;
            for (int d = 0; d < D; d++)
            {
                float xhat = (x[d] - mean) * invStd;
                dw[d]      += dy[d] * xhat;
                db[d]      += dy[d];
                dyDotXhat  += dy[d] * w[d] * xhat;
                dySum      += dy[d] * w[d];
            }

            // dL/dx = invStd * (dy*w - (1/D)*(dySum + xhat*dyDotXhat))
            for (int d = 0; d < D; d++)
            {
                float xhat = (x[d] - mean) * invStd;
                dx[d] = invStd * (dy[d] * w[d] - (dySum + xhat * dyDotXhat) / D);
            }
        }

        return dInput;
    }

    
    // Scaled dot-product attention backward
    // scores = Q @ K^T / sqrt(d)    probs = softmax(scores, causal)
    // out = probs @ V
    

    /// <summary>
    /// Returns (dQ, dK, dV) for one head.
    /// Q,K,V: [SeqLen, HeadDim]   probs: [SeqLen, SeqLen]   dOut: [SeqLen, HeadDim]
    /// </summary>
    public static unsafe (Tensor<float> dQ, Tensor<float> dK, Tensor<float> dV)
        Attention(
            Tensor<float> dOut,   // [S, HeadDim]
            Tensor<float> q,      // [S, HeadDim]
            Tensor<float> k,      // [S, HeadDim]
            Tensor<float> v,      // [S, HeadDim]
            Tensor<float> probs,  // [S, S]
            float         scale)
    {
        int S = q.Shape.Rows;
        int D = q.Shape.Cols;

        var dQ    = new Tensor<float>(S, D);
        var dK    = new Tensor<float>(S, D);
        var dV    = new Tensor<float>(S, D);
        var dProbs = new Tensor<float>(S, S);

        // dV = probs^T @ dOut   [S, D]
        fixed (float* pP = probs.Data, pDOut = dOut.Data, pDV = dV.Data)
        {
            for (int j = 0; j < S; j++)
                for (int d = 0; d < D; d++)
                {
                    float sum = 0f;
                    for (int i = 0; i < S; i++) sum += pP[(long)i * S + j] * pDOut[(long)i * D + d];
                    pDV[(long)j * D + d] += sum;
                }
        }

        // dProbs = dOut @ V^T   [S, S]
        fixed (float* pDOut = dOut.Data, pV = v.Data, pDP = dProbs.Data)
        {
            for (int i = 0; i < S; i++)
                for (int j = 0; j < S; j++)
                {
                    float sum = 0f;
                    for (int d = 0; d < D; d++) sum += pDOut[(long)i * D + d] * pV[(long)j * D + d];
                    pDP[(long)i * S + j] = sum;
                }
        }

        var dScores = new Tensor<float>(S, S);
        for (int i = 0; i < S; i++)
        {
            ReadOnlySpan<float> pRow  = probs.RowSpan(i);
            ReadOnlySpan<float> dpRow = dProbs.RowSpan(i);
            Span<float>         dsRow = dScores.RowSpan(i);
            float dot = 0f;
            for (int j = 0; j < S; j++) dot += pRow[j] * dpRow[j];
            for (int j = 0; j < S; j++) dsRow[j] = pRow[j] * (dpRow[j] - dot);
        }
        dProbs.Dispose();

        // dQ = dScores @ K * scale   [S, D]
        // dK = dScores^T @ Q * scale [S, D]
        fixed (float* pDS = dScores.Data, pQ = q.Data, pK = k.Data, pDQ = dQ.Data, pDK = dK.Data)
        {
            for (int i = 0; i < S; i++)
                for (int d = 0; d < D; d++)
                {
                    float sumQ = 0f, sumK = 0f;
                    for (int j = 0; j < S; j++)
                    {
                        sumQ += pDS[(long)i * S + j] * pK[(long)j * D + d];
                        sumK += pDS[(long)j * S + i] * pQ[(long)j * D + d];
                    }
                    pDQ[(long)i * D + d] += sumQ * scale;
                    pDK[(long)i * D + d] += sumK * scale;
                }
        }
        dScores.Dispose();

        return (dQ, dK, dV);
    }

    
    // Embedding backward — scatter-add gradient into the embedding table
    

    /// <summary>
    /// Accumulates dOutput into the rows of the embedding weight gradient
    /// selected by tokenIds. There is no "dInput" for an embedding lookup —
    /// the integer indices have no gradient.
    /// </summary>
    public static void Embedding(
        Tensor<float> dOutput,   // [T, EmbedDim] flat
        Tensor<int>   tokenIds,  // [T] flat
        Parameter     weight)    // [VocabSize, EmbedDim]
    {
        int T   = dOutput.Shape.Rows;
        int D   = dOutput.Shape.Cols;
        Span<float> dW = weight.Grad.Data;

        for (int t = 0; t < T; t++)
        {
            int id = tokenIds[t];
            ReadOnlySpan<float> dRow  = dOutput.RowSpan(t);
            Span<float>         dWRow = dW.Slice(id * D, D);
            for (int d = 0; d < D; d++) dWRow[d] += dRow[d];
        }
    }

    
    // Fast transcendental helpers — polynomial approximations
    

    /// <summary>exp(x) via range-reduced degree-6 polynomial, ≈5 ULP over [-88, 88].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> FastExp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-88.0f)), Vector256.Create(88.0f));
        var z = Avx.Multiply(x, Vector256.Create(1.4426950408889634f));
        var magic = Vector256.Create(12582912.0f);
        var nF = Avx.Subtract(Avx.Add(z, magic), magic);
        var nI = Avx2.ConvertToVector256Int32(nF);
        var r = Avx.Subtract(z, nF);
        var u = Avx.Multiply(r, Vector256.Create(0.6931471805599453f));

        // Horner degree-6
        var p = Avx.Add(Vector256.Create(1.0f),
            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f),
                Avx.Multiply(u, Avx.Add(Vector256.Create(0.5f),
                    Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 6.0f),
                        Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 24.0f),
                            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 120.0f),
                                Avx.Multiply(u, Vector256.Create(1.0f / 720.0f))
                            ))
                        ))
                    ))
                ))
            ))
        );

        var expAdj = Avx2.Add(nI, Vector256.Create(127));
        expAdj = Avx2.Min(Avx2.Max(expAdj, Vector256.Create(0)), Vector256.Create(254));
        var pow2nBits = Avx2.ShiftLeftLogical(expAdj, 23);
        return Avx.Multiply(p, Vector256.AsSingle(pow2nBits));
    }

    /// <summary>tanh(z) = (exp(2z) - 1) / (exp(2z) + 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> FastTanh(Vector256<float> z)
    {
        z = Avx.Min(Avx.Max(z, Vector256.Create(-9.0f)), Vector256.Create(9.0f));
        var twoZ = Avx.Multiply(z, Vector256.Create(2.0f));
        var e2z = FastExp(twoZ);
        var one = Vector256.Create(1.0f);
        return Avx.Divide(Avx.Subtract(e2z, one), Avx.Add(e2z, one));
    }

    
    // Activation function backward — SiLU and GELU derivatives
    

    /// <summary>SiLU backward: d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)).</summary>
    public static unsafe Tensor<float> SiLUBackward(Tensor<float> dOutput, Tensor<float> preAct)
    {
        var dInput = new Tensor<float>(dOutput.Shape);
        var src = preAct.Data;
        var dst = dInput.Data;
        var dy = dOutput.Data;

        if (Avx2.IsSupported)
        {
            fixed (float* pX = src, pDy = dy, pDst = dst)
            {
                int i = 0, n = dst.Length;
                var one = Vector256.Create(1.0f);
                for (; i <= n - 8; i += 8)
                {
                    var x = Vector256.LoadUnsafe(ref pX[i]);
                    var d = Vector256.LoadUnsafe(ref pDy[i]);
                    var sig = Avx.Divide(one, Avx.Add(one, FastExp(Avx.Subtract(Vector256<float>.Zero, x))));
                    Vector256.StoreUnsafe(
                        Avx.Multiply(d, Avx.Multiply(sig, Avx.Subtract(Avx.Add(one, x), Avx.Multiply(x, sig)))),
                        ref pDst[i]);
                }
                for (; i < n; i++)
                {
                    float x = pX[i];
                    float sig = 1f / (1f + MathF.Exp(-x));
                    pDst[i] = pDy[i] * (sig + x * sig * (1f - sig));
                }
            }
        }
        else
        {
            for (int i = 0; i < src.Length; i++)
            {
                float x = src[i];
                float sig = 1f / (1f + MathF.Exp(-x));
                dst[i] = dy[i] * (sig + x * sig * (1f - sig));
            }
        }
        return dInput;
    }

    private const float SqrtTwoPiInv = 0.7978845608f;
    private const float GeluCoeff = 0.044715f;

    /// <summary>GELU backward (tanh approximation derivative).</summary>
    public static unsafe Tensor<float> GELUBackward(Tensor<float> dOutput, Tensor<float> preAct)
    {
        var dInput = new Tensor<float>(dOutput.Shape);
        var src = preAct.Data;
        var dst = dInput.Data;
        var dy = dOutput.Data;

        if (Avx2.IsSupported)
        {
            fixed (float* pX = src, pDy = dy, pDst = dst)
            {
                int i = 0, n = dst.Length;
                var half = Vector256.Create(0.5f);
                var one = Vector256.Create(1.0f);
                var vSqrt2PiInv = Vector256.Create(0.7978845608f);
                var vCoeff = Vector256.Create(0.044715f);
                var v3Coeff = Vector256.Create(3f * 0.044715f);

                for (; i <= n - 8; i += 8)
                {
                    var x = Vector256.LoadUnsafe(ref pX[i]);
                    var d = Vector256.LoadUnsafe(ref pDy[i]);
                    var x3 = Avx.Multiply(Avx.Multiply(x, x), x);
                    var inner = Avx.Multiply(vSqrt2PiInv, Avx.Add(x, Avx.Multiply(vCoeff, x3)));
                    var t = FastTanh(inner);
                    var dtanh = Avx.Subtract(one, Avx.Multiply(t, t));
                    var dInner = Avx.Multiply(vSqrt2PiInv, Avx.Add(one, Avx.Multiply(v3Coeff, Avx.Multiply(x, x))));
                    var geluGrad = Avx.Add(
                        Avx.Multiply(half, Avx.Add(one, t)),
                        Avx.Multiply(half, Avx.Multiply(x, Avx.Multiply(dtanh, dInner))));
                    Vector256.StoreUnsafe(Avx.Multiply(d, geluGrad), ref pDst[i]);
                }
                for (; i < n; i++)
                {
                    float x = pX[i];
                    float x3 = x * x * x;
                    float inner = SqrtTwoPiInv * (x + GeluCoeff * x3);
                    float tanh = MathF.Tanh(inner);
                    float dtanh = 1f - tanh * tanh;
                    float dInner = SqrtTwoPiInv * (1f + 3f * GeluCoeff * x * x);
                    pDst[i] = pDy[i] * (0.5f * (1f + tanh) + 0.5f * x * dtanh * dInner);
                }
            }
        }
        else
        {
            for (int i = 0; i < src.Length; i++)
            {
                float x = src[i];
                float x3 = x * x * x;
                float inner = SqrtTwoPiInv * (x + GeluCoeff * x3);
                float tanh = MathF.Tanh(inner);
                float dtanh = 1f - tanh * tanh;
                float dInner = SqrtTwoPiInv * (1f + 3f * GeluCoeff * x * x);
                dst[i] = dy[i] * (0.5f * (1f + tanh) + 0.5f * x * dtanh * dInner);
            }
        }
        return dInput;
    }

    
    // Gradient clipping — L2 norm clipping across all parameters
    

    /// <summary>
    /// Clips the global gradient norm to <paramref name="maxNorm"/> in-place.
    /// Returns the pre-clip norm for logging.
    /// </summary>
    public static float ClipGlobalNorm(
        IEnumerable<Parameter> parameters,
        float maxNorm)
    {
        float totalNormSq = 0f;
        foreach (var p in parameters)
        {
            var grad = p.Grad.Data;
            foreach (float g in grad) totalNormSq += g * g;
        }

        float totalNorm = MathF.Sqrt(totalNormSq);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / totalNorm;
            foreach (var p in parameters)
            {
                var grad = p.Grad.Data;
                for (int i = 0; i < grad.Length; i++) grad[i] *= scale;
            }
        }

        return totalNorm;
    }

    /// <summary>
    /// Clips the global gradient norm to <paramref name="maxNorm"/> in-place.
    /// Uses the JigSaw-assembled <paramref name="ops"/> for the L2 norm accumulation
    /// so the AVX2 or scalar path is selected at factory time, not here.
    /// Returns the pre-clip norm for logging.
    /// </summary>
    public static float ClipGlobalNorm(
        IEnumerable<Parameter> parameters,
        float maxNorm,
        TrainingOps ops)
    {
        float totalNormSq = 0f;
        foreach (var p in parameters)
            totalNormSq += ops.L2NormSq(p.Grad.Data);

        float totalNorm = MathF.Sqrt(totalNormSq);

        if (totalNorm > maxNorm)
        {
            float scale = maxNorm / totalNorm;
            foreach (var p in parameters)
            {
                var grad = p.Grad.Data;
                for (int i = 0; i < grad.Length; i++) grad[i] *= scale;
            }
        }

        return totalNorm;
    }
}
