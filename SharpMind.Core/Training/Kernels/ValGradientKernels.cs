using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using JigSawDotNet;
using SharpMind.Training.Kernels;

namespace SharpMind.Core.Training.Kernels;

public sealed class ValLinearBackward : ILinearBackward
{
    public unsafe Tensor<float> Compute(
        Tensor<float> dOutput,
        Tensor<float> input,
        Parameter     weight,
        Parameter?    bias = null)
    {
        int B   = dOutput.Shape.Rows;
        int Out = dOutput.Shape.Cols;
        int In  = input.Shape.Cols;

        var dInput = new Tensor<float>(B, In);

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
}

public sealed class ValRMSNormBackward : IRMSNormBackward
{
    public Tensor<float> Compute(
        Tensor<float> dOutput,
        Tensor<float> xNorm,
        float[]       rmsInv,
        Parameter     weight)
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

            for (int d = 0; d < D; d++) dw[d] += dy[d] * xn[d];

            float dot = 0f;
            for (int d = 0; d < D; d++) dot += dy[d] * w[d] * xn[d];
            dot /= D;

            for (int d = 0; d < D; d++)
                dxRow[d] = ri * (dy[d] * w[d] - xn[d] * dot);
        }

        return dInput;
    }
}

public sealed class ValLayerNormBackward : ILayerNormBackward
{
    public Tensor<float> Compute(
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

            float dyDotXhat = 0f, dySum = 0f;
            for (int d = 0; d < D; d++)
            {
                float xhat = (x[d] - mean) * invStd;
                dw[d]      += dy[d] * xhat;
                db[d]      += dy[d];
                dyDotXhat  += dy[d] * w[d] * xhat;
                dySum      += dy[d] * w[d];
            }

            for (int d = 0; d < D; d++)
            {
                float xhat = (x[d] - mean) * invStd;
                dx[d] = invStd * (dy[d] * w[d] - (dySum + xhat * dyDotXhat) / D);
            }
        }

        return dInput;
    }
}

public sealed class ValAttentionBackward : IAttentionBackward
{
    public unsafe (Tensor<float> dQ, Tensor<float> dK, Tensor<float> dV) Compute(
        Tensor<float> dOut,
        Tensor<float> q,
        Tensor<float> k,
        Tensor<float> v,
        Tensor<float> probs,
        float         scale)
    {
        int S = q.Shape.Rows;
        int D = q.Shape.Cols;

        var dQ    = new Tensor<float>(S, D);
        var dK    = new Tensor<float>(S, D);
        var dV    = new Tensor<float>(S, D);
        var dProbs = new Tensor<float>(S, S);

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
}

public sealed class ValEmbeddingBackward : IEmbeddingBackward
{
    public void Compute(
        Tensor<float> dOutput,
        Tensor<int>   tokenIds,
        Parameter     weight)
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
}

public sealed class ValActivationBackward : IActivationBackward
{
    public Tensor<float> Compute(Tensor<float> dOutput, Tensor<float> preAct, ActivationType type)
    {
        var dInput = new Tensor<float>(dOutput.Shape);
        var src    = preAct.Data;
        var dst    = dInput.Data;
        var dy     = dOutput.Data;

        if (type == ActivationType.SiLU)
        {
            for (int i = 0; i < src.Length; i++)
            {
                float x   = src[i];
                float sig = 1f / (1f + MathF.Exp(-x));
                dst[i] = dy[i] * (sig + x * sig * (1f - sig));
            }
        }
        else // GELU
        {
            const float SqrtTwoPiInv = 0.7978845608f;
            const float GeluCoeff    = 0.044715f;
            for (int i = 0; i < src.Length; i++)
            {
                float x    = src[i];
                float x3   = x * x * x;
                float inner = SqrtTwoPiInv * (x + GeluCoeff * x3);
                float tanh  = MathF.Tanh(inner);
                float dtanh = 1f - tanh * tanh;
                float dInner = SqrtTwoPiInv * (1f + 3f * GeluCoeff * x * x);
                dst[i] = dy[i] * (0.5f * (1f + tanh) + 0.5f * x * dtanh * dInner);
            }
        }
        return dInput;
    }
}
