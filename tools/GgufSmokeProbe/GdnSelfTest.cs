using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Layers.DeltaNet;

namespace GgufSmokeProbe;

internal static class GdnSelfTest
{
    public static int RunReal(string path)
    {
        var meta = SharpMind.Model.Format.GgufLoader.LoadMeta(path);
        var cfg = SharpMind.Model.Format.GgufLoader.LoadConfig(meta)!;
        var sharp = cfg.ForModel();
        using var weights = SharpMind.Model.ModelFactory.CreateWeights(
            cfg, sharp, SharpMind.Core.Quantization.QuantizationFactory.Create(sharp.ToJigSawMapping()), path,
            SharpMind.Model.Config.LoadMode.Full);
        weights.InitializeWeights();
        var bw = weights.Blocks[0];
        int seq = 5;
        var x = new Tensor<float>(1, seq, cfg.HiddenDim);
        var rng = new Random(99);
        for (int i = 0; i < x.Data.Length; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        using var layer = new DeltaNetLayer(cfg, null!, bw, mapping: null);
        using var cache = new DeltaNetCache(cfg.SsmConvKernel, cfg.SsmConvDim, cfg.SsmTimeStepRank,
            cfg.SsmStateSize, cfg.SsmValueHeadDim, 1, 64);
        using var got = layer.Forward(x, cache);
        float[] want = Reference(cfg, bw, x.Data, seq);
        float maxDiff = 0;
        int maxAt = -1;
        for (int i = 0; i < want.Length; i++)
        {
            float d = MathF.Abs(want[i] - got.Data[i]);
            if (d > maxDiff) { maxDiff = d; maxAt = i; }
        }
        Console.WriteLine($"gdn real self-test: max|diff|={maxDiff:G6} at {maxAt} (ours={got.Data[maxAt]:G6} ref={want[maxAt]:G6})");
        return maxDiff < 1e-3f ? 0 : 1;
    }

    public static int RunAttentionReal(string path)
    {
        var meta = SharpMind.Model.Format.GgufLoader.LoadMeta(path);
        var cfg = SharpMind.Model.Format.GgufLoader.LoadConfig(meta)!;
        var sharp = cfg.ForModel();
        using var weights = SharpMind.Model.ModelFactory.CreateWeights(
            cfg, sharp, SharpMind.Core.Quantization.QuantizationFactory.Create(sharp.ToJigSawMapping()), path,
            SharpMind.Model.Config.LoadMode.Full);
        weights.InitializeWeights();
        using var model = SharpMind.Model.ModelFactory.CreateTransformer(weights, sharp);
        var attn = model.GetBlock(3)!.Attention!;
        var bw = weights.Blocks[3];
        int seq = 5;
        var x = new Tensor<float>(1, seq, cfg.HiddenDim);
        var rng = new Random(99);
        for (int i = 0; i < x.Data.Length; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        using var got = attn.Forward(x, 0, true, null, null, 0);
        float[] want = AttentionRef(cfg, bw, x.Data, seq);
        float maxDiff = 0;
        int maxAt = -1;
        for (int i = 0; i < want.Length; i++)
        {
            float d = MathF.Abs(want[i] - got.Data[i]);
            if (d > maxDiff) { maxDiff = d; maxAt = i; }
        }
        Console.WriteLine($"attn real self-test: max|diff|={maxDiff:G6} at {maxAt} (ours={got.Data[maxAt]:G6} ref={want[maxAt]:G6})");
        return maxDiff < 1e-3f ? 0 : 1;
    }

    private static float[] AttentionRef(ModelConfig cfg, TransformerWeights.BlockWeights bw, ReadOnlySpan<float> x, int seq)
    {
        int hidden = cfg.HiddenDim, numH = cfg.NumHeads, numKv = cfg.NumKvHeads;
        int hd = cfg.HeadDim, ropeDim = cfg.RopeDim ?? hd, ropePairs = ropeDim / 2;
        float eps = cfg.NormEps, theta = cfg.RopeTheta;
        int qDim = numH * hd, kvDim = numKv * hd;
        int group = numH / numKv;

        float[] qFull = MatMul(bw.Wq!, hidden, 2 * qDim, x, seq);
        float[] kRaw = MatMul(bw.Wk!, hidden, kvDim, x, seq);
        float[] vRaw = MatMul(bw.Wv!, hidden, kvDim, x, seq);

        var q = new float[seq * numH * hd];
        var gate = new float[seq * numH * hd];
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < numH; h++)
            {
                int src = (t * numH + h) * 2 * hd;
                int dst = (t * numH + h) * hd;
                for (int d = 0; d < hd; d++)
                {
                    q[dst + d] = qFull[src + d];
                    gate[dst + d] = qFull[src + hd + d];
                }
            }

        NormRows(q, seq * numH, hd, bw.QNormW!.Data, eps);
        NormRows(kRaw, seq * numKv, hd, bw.KNormW!.Data, eps);
        Rope(q, seq, numH, hd, ropeDim, theta);
        Rope(kRaw, seq, numKv, hd, ropeDim, theta);

        var outv = new float[seq * qDim];
        var scores = new float[seq];
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < numH; h++)
            {
                int kvh = h / group;
                float max = float.NegativeInfinity;
                for (int j = 0; j <= t; j++)
                {
                    float s = 0;
                    for (int d = 0; d < hd; d++)
                        s += q[(t * numH + h) * hd + d] * kRaw[(j * numKv + kvh) * hd + d];
                    s /= MathF.Sqrt(hd);
                    scores[j] = s;
                    if (s > max) max = s;
                }
                float sum = 0;
                for (int j = 0; j <= t; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
                for (int d = 0; d < hd; d++)
                {
                    float acc = 0;
                    for (int j = 0; j <= t; j++)
                        acc += scores[j] / sum * vRaw[(j * numKv + kvh) * hd + d];
                    float g = 1f / (1f + MathF.Exp(-gate[(t * numH + h) * hd + d]));
                    outv[(t * numH + h) * hd + d] = acc * g;
                }
            }

        return MatMul(bw.Wo!, qDim, hidden, outv, seq);
    }

    private static void NormRows(float[] values, int rows, int dim, ReadOnlySpan<float> weight, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            float ss = 0;
            var row = values.AsSpan(r * dim, dim);
            foreach (float f in row) ss += f * f;
            float inv = 1f / MathF.Sqrt(ss / dim + eps);
            for (int i = 0; i < dim; i++) row[i] = row[i] * inv * weight[i];
        }
    }

    private static void Rope(float[] values, int seq, int heads, int hd, int ropeDim, float theta)
    {
        int pairs = ropeDim / 2;
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < heads; h++)
            {
                int off = (t * heads + h) * hd;
                for (int d = 0; d < pairs; d++)
                {
                    float angle = t / MathF.Pow(theta, 2f * d / ropeDim);
                    float c = MathF.Cos(angle), s = MathF.Sin(angle);
                    float a = values[off + d], b = values[off + d + pairs];
                    values[off + d] = a * c - b * s;
                    values[off + d + pairs] = b * c + a * s;
                }
            }
    }

    public static int RunAttentionPosition(string path)
    {
        var meta = SharpMind.Model.Format.GgufLoader.LoadMeta(path);
        var cfg = SharpMind.Model.Format.GgufLoader.LoadConfig(meta)!;
        var sharp = cfg.ForModel();
        using var weights = SharpMind.Model.ModelFactory.CreateWeights(
            cfg, sharp, SharpMind.Core.Quantization.QuantizationFactory.Create(sharp.ToJigSawMapping()), path,
            SharpMind.Model.Config.LoadMode.Full);
        weights.InitializeWeights();
        using var model = SharpMind.Model.ModelFactory.CreateTransformer(weights, sharp);
        var attn = model.GetBlock(3)!.Attention!;
        var x = new Tensor<float>(1, 5, cfg.HiddenDim);
        var rng = new Random(3);
        for (int i = 0; i < x.Data.Length; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        using var a = attn.Forward(x, 0, true, null, null, 0);
        using var b = attn.Forward(x, 1, true, null, null, 0);
        float max = 0;
        for (int i = 0; i < a.Data.Length; i++) max = MathF.Max(max, MathF.Abs(a.Data[i] - b.Data[i]));
        Console.WriteLine($"attn position sensitivity: max|pos0-pos1|={max:G6}");
        return max > 1e-4f ? 0 : 1;
    }

    public static int Run()
    {
        const int hidden = 8;
        const int nK = 2;
        const int nV = 2;
        const int kd = 4;
        const int vd = 4;
        const int keyDim = nK * kd;
        const int valueDim = nV * vd;
        const int convDim = 2 * keyDim + valueDim;
        const int taps = 4;
        const int seq = 5;

        var cfg = new ModelConfig
        {
            Architecture = "qwen35",
            VocabSize = 32,
            HiddenDim = hidden,
            NumLayers = 1,
            NumHeads = 1,
            NumKvHeads = 1,
            FfnDim = 16,
            MaxSeqLen = 32,
            SsmConvKernel = taps,
            SsmInnerSize = valueDim,
            SsmStateSize = kd,
            SsmTimeStepRank = nV,
            SsmGroupCount = nK,
            FullAttentionInterval = 4,
            GatedAttentionOutput = true,
            NormEps = 1e-6f,
        };

        var rng = new Random(1234);
        var bw = new TransformerWeights.BlockWeights
        {
            LayerIndex = 0,
            Wqkv = Rand(rng, hidden, convDim),
            WGate = Rand(rng, hidden, valueDim),
            WAlpha = Rand(rng, hidden, nV),
            WBeta = Rand(rng, hidden, nV),
            WOut = Rand(rng, valueDim, hidden),
            WConv = Rand(rng, taps, convDim),
            SsmA = Rand(rng, nV),
            SsmDt = Rand(rng, nV),
            SsmNorm = Rand(rng, vd),
        };
        // Keep A_log-style decay negative, as in real checkpoints.
        for (int i = 0; i < nV; i++) bw.SsmA!.Data[i] = -MathF.Abs(bw.SsmA!.Data[i]);

        var x = new Tensor<float>(1, seq, hidden);
        for (int i = 0; i < x.Data.Length; i++) x.Data[i] = (float)(rng.NextDouble() * 2 - 1);

        using var layer = new DeltaNetLayer(cfg, null!, bw, mapping: null);
        using var cache = new DeltaNetCache(taps, convDim, nV, kd, vd, 1, 64);
        using var got = layer.Forward(x, cache);

        float[] want = Reference(cfg, bw, x.Data, seq);

        float maxDiff = 0;
        int maxAt = -1;
        for (int i = 0; i < want.Length; i++)
        {
            float d = MathF.Abs(want[i] - got.Data[i]);
            if (d > maxDiff) { maxDiff = d; maxAt = i; }
        }
        Console.WriteLine($"gdn self-test: max|diff|={maxDiff:G6} at {maxAt} (ours={got.Data[maxAt]:G6} ref={want[maxAt]:G6})");
        return maxDiff < 1e-4f ? 0 : 1;
    }

    private static Tensor<float> Rand(Random rng, int a, int b)
    {
        var t = new Tensor<float>(a, b);
        for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        return t;
    }

    private static Tensor<float> Rand(Random rng, int n)
    {
        var t = new Tensor<float>(n);
        for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        return t;
    }

    private static float[] Reference(ModelConfig cfg, TransformerWeights.BlockWeights bw, ReadOnlySpan<float> x, int seq)
    {
        int hidden = cfg.HiddenDim, nK = cfg.SsmGroupCount, nV = cfg.SsmTimeStepRank;
        int kd = cfg.SsmStateSize, vd = cfg.SsmValueHeadDim;
        int keyDim = cfg.SsmKeyDim, valueDim = cfg.SsmValueDim, convDim = cfg.SsmConvDim;
        int taps = cfg.SsmConvKernel, stateRows = taps - 1;
        int group = nV / nK;

        float[] qkv = MatMul(bw.Wqkv!, hidden, convDim, x, seq);
        float[] z = MatMul(bw.WGate!, hidden, valueDim, x, seq);
        float[] alpha = MatMul(bw.WAlpha!, hidden, nV, x, seq);
        float[] beta = MatMul(bw.WBeta!, hidden, nV, x, seq);

        var conv = new float[convDim];
        var delta = new float[nV * kd * vd];
        var sk = new float[vd];
        var outv = new float[valueDim];
        float[] y = new float[seq * hidden];

        for (int t = 0; t < seq; t++)
        {
            int qkvOff = t * convDim;
            for (int c = 0; c < convDim; c++)
            {
                float acc = 0;
                for (int k = 0; k < taps; k++)
                {
                    int srcT = t - (taps - 1) + k;
                    if (srcT >= 0) acc += qkv[srcT * convDim + c] * bw.WConv!.Data[k * convDim + c];
                }
                conv[c] = acc / (1f + MathF.Exp(-acc));
            }

            for (int h = 0; h < nK; h++)
            {
                L2(conv.AsSpan(h * kd, kd), cfg.NormEps);
                L2(conv.AsSpan(keyDim + h * kd, kd), cfg.NormEps);
            }

            for (int h = 0; h < nV; h++)
            {
                int kh = h / group;
                int qb = kh * kd, kb = keyDim + kh * kd, vb = 2 * keyDim + h * vd;
                float a = alpha[t * nV + h] + bw.SsmDt!.Data[h];
                float sp = a > 20 ? a : MathF.Log(1 + MathF.Exp(a));
                float decay = MathF.Exp(sp * bw.SsmA!.Data[h]);
                float b = 1f / (1f + MathF.Exp(-beta[t * nV + h]));
                int sb = h * kd * vd;

                for (int i = 0; i < kd * vd; i++) delta[sb + i] *= decay;
                for (int v = 0; v < vd; v++)
                {
                    float mem = 0;
                    for (int k = 0; k < kd; k++) mem += delta[sb + k * vd + v] * conv[kb + k];
                    sk[v] = (conv[vb + v] - mem) * b;
                }
                for (int k = 0; k < kd; k++)
                    for (int v = 0; v < vd; v++)
                        delta[sb + k * vd + v] += conv[kb + k] * sk[v];

                float ss = 0;
                for (int v = 0; v < vd; v++)
                {
                    float acc = 0;
                    for (int k = 0; k < kd; k++) acc += delta[sb + k * vd + v] * conv[qb + k];
                    acc /= MathF.Sqrt(kd);
                    outv[h * vd + v] = acc;
                    ss += acc * acc;
                }
                float inv = 1f / MathF.Sqrt(ss / vd + cfg.NormEps);
                for (int v = 0; v < vd; v++)
                {
                    float gz = z[t * valueDim + h * vd + v];
                    outv[h * vd + v] *= inv * bw.SsmNorm!.Data[v] * (gz / (1f + MathF.Exp(-gz)));
                }
            }

            for (int o = 0; o < hidden; o++)
            {
                float acc = 0;
                for (int i = 0; i < valueDim; i++)
                    acc += outv[i] * bw.WOut!.Data[i * hidden + o];
                y[t * hidden + o] = acc;
            }
        }
        return y;
    }

    private static float[] MatMul(Tensor<float> w, int inf, int outf, ReadOnlySpan<float> x, int rows)
    {
        var y = new float[rows * outf];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outf; o++)
            {
                float acc = 0;
                for (int i = 0; i < inf; i++) acc += x[r * inf + i] * w.Data[i * outf + o];
                y[r * outf + o] = acc;
            }
        return y;
    }

    private static void L2(Span<float> v, float eps)
    {
        double ss = 0;
        foreach (float f in v) ss += f * f;
        float inv = 1f / MathF.Sqrt((float)ss + eps);
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }
}
