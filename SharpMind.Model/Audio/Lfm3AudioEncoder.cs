using SharpMind.Model.Format;

namespace SharpMind.Model.Audio;

/// <summary>
/// Native LFM3-Audio speech encoder. Reads the encoder and projector tensors from the
/// Audio mmproj and turns a log-mel spectrogram into one 2048-wide embedding per eight
/// frames. The graph follows llama.cpp's <c>clip_graph_lfm3a</c>: a stride-2 conv
/// downsampler, a causal 17-layer LFM2 stack (10 short-conv, 7 attention), a final RMS
/// norm and a LayerNorm/GELU projector. Attention uses NEOX RoPE with theta 1e6 and a
/// 32-token sliding window.
/// </summary>
public sealed class Lfm3AudioEncoder : IDisposable
{
    public const int MelBins = 128;
    public const int OutputDim = 2048;

    private const int Hidden = 768;
    private const int Heads = 8;
    private const int KvHeads = 4;
    private const int HeadDim = 96;
    private const int FfnDim = 1792;
    private const int LayerCount = 17;
    private const int Window = 32;
    private const int ConvChannels = 256;
    private const int DownsampledBins = 16;
    private const float Eps = 1e-5f;
    private const float RopeTheta = 1_000_000f;

    private readonly Downsampler _downsampler;
    private readonly Block[] _blocks;
    private readonly float[] _postNorm;
    private readonly Projector _projector;
    private bool _disposed;

    public Lfm3AudioEncoder(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _downsampler = new Downsampler(reader);
        _blocks = new Block[LayerCount];
        for (int layer = 0; layer < LayerCount; layer++)
            _blocks[layer] = Block.Load(reader, layer);
        _postNorm = reader.ReadFloat32("a.post_ln.weight");
        _projector = new Projector(reader);
    }

    /// <summary>Number of encoder tokens for a spectrogram with <paramref name="frames"/> frames.</summary>
    public static int TokenCount(int frames) => (frames + 7) / 8;

    /// <summary>Encodes a time-major mel spectrogram, <c>[frames, 128]</c>, to <c>[tokens, 2048]</c>.</summary>
    public float[,] Encode(float[,] mel)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(mel);
        if (mel.GetLength(1) != MelBins)
            throw new ArgumentException($"Expected {MelBins} mel bins, got {mel.GetLength(1)}.", nameof(mel));

        int frames = mel.GetLength(0);
        var flat = new float[frames * MelBins];
        Buffer.BlockCopy(mel, 0, flat, 0, flat.Length * sizeof(float));
        return Encode(flat, frames);
    }

    /// <summary>Encodes a flat time-major mel spectrogram.</summary>
    public float[,] Encode(ReadOnlySpan<float> mel, int frames)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (frames <= 0) return new float[0, OutputDim];
        if (mel.Length < frames * MelBins)
            throw new ArgumentException("Mel buffer is shorter than frames * 128.", nameof(mel));

        float[] hidden = _downsampler.Forward(mel, frames);
        int tokens = hidden.Length / Hidden;

        for (int layer = 0; layer < LayerCount; layer++)
            hidden = _blocks[layer].Forward(hidden, tokens);

        RmsNormInPlace(hidden, _postNorm, tokens, Hidden, Eps);
        var projected = new float[tokens * OutputDim];
        _projector.Forward(hidden, tokens, projected);

        var result = new float[tokens, OutputDim];
        Buffer.BlockCopy(projected, 0, result, 0, projected.Length * sizeof(float));
        return result;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Block block in _blocks) block.Dispose();
    }

    private sealed class Downsampler
    {
        private readonly float[] _conv0W, _conv0B, _conv2W, _conv2B;
        private readonly float[] _lin3W, _lin3B, _conv5W, _conv5B;
        private readonly float[] _lin6W, _lin6B, _outW, _outB;

        public Downsampler(GgufTensorReader reader)
        {
            _conv0W = reader.ReadFloat32("a.conv1d.0.weight");
            _conv0B = reader.ReadFloat32("a.conv1d.0.bias");
            _conv2W = reader.ReadFloat32("a.conv1d.2.weight");
            _conv2B = reader.ReadFloat32("a.conv1d.2.bias");
            _lin3W = reader.ReadFloat32("a.conv1d.3.weight");
            _lin3B = reader.ReadFloat32("a.conv1d.3.bias");
            _conv5W = reader.ReadFloat32("a.conv1d.5.weight");
            _conv5B = reader.ReadFloat32("a.conv1d.5.bias");
            _lin6W = reader.ReadFloat32("a.conv1d.6.weight");
            _lin6B = reader.ReadFloat32("a.conv1d.6.bias");
            _outW = reader.ReadFloat32("a.pre_encode.out.weight");
            _outB = reader.ReadFloat32("a.pre_encode.out.bias");
        }

        /// <summary>Input [frames, 128] -> [ceil(frames / 8), 768].</summary>
        public float[] Forward(ReadOnlySpan<float> mel, int frames)
        {
            int t0 = frames, f0 = MelBins, c0 = 1;
            float[] cur = mel[..(frames * MelBins)].ToArray();

            int t1 = (t0 + 1) / 2, f1 = 64;
            var next = new float[t1 * f1 * ConvChannels];
            Conv2d(cur, t0, f0, c0, ConvChannels, _conv0W, _conv0B, next, t1, f1, depthwise: false);
            SiLU(next);
            cur = next; t0 = t1; f0 = f1; c0 = ConvChannels;

            t1 = (t0 + 1) / 2; f1 = 32;
            next = new float[t1 * f1 * c0];
            Conv2d(cur, t0, f0, c0, c0, _conv2W, _conv2B, next, t1, f1, depthwise: true);
            Pointwise(next, t1 * f1, c0, _lin3W, _lin3B);
            SiLU(next);
            cur = next; t0 = t1; f0 = f1;

            t1 = (t0 + 1) / 2; f1 = DownsampledBins;
            next = new float[t1 * f1 * c0];
            Conv2d(cur, t0, f0, c0, c0, _conv5W, _conv5B, next, t1, f1, depthwise: true);
            Pointwise(next, t1 * f1, c0, _lin6W, _lin6B);
            SiLU(next);

            // [t, f, c] -> [t, f + f0*c] -> linear 4096 -> 768
            int flatDim = f1 * c0;
            var tokens = new float[t1 * Hidden];
            var row = new float[flatDim];
            for (int t = 0; t < t1; t++)
            {
                for (int f = 0; f < f1; f++)
                {
                    int source = (t * f1 + f) * c0;
                    for (int c = 0; c < c0; c++) row[f + f1 * c] = next[source + c];
                }
                MatMul(row, _outW, flatDim, Hidden, tokens.AsSpan(t * Hidden, Hidden));
                for (int h = 0; h < Hidden; h++) tokens[t * Hidden + h] += _outB[h];
            }

            return tokens;
        }
    }

    private sealed class Block : IDisposable
    {
        private readonly bool _isConv;
        private readonly float[] _ln1, _ln2;
        private readonly float[] _pw1, _pw2, _convDw;
        private readonly float[] _qW, _kW, _vW, _oW, _qNorm, _kNorm;
        private readonly float[] _gateW, _upW, _downW;
        private float[]? _scratch;

        private Block(GgufTensorReader reader, int layer)
        {
            string prefix = $"a.blk.{layer}.";
            _ln1 = reader.ReadFloat32(prefix + "ln1.weight");
            _ln2 = reader.ReadFloat32(prefix + "ln2.weight");
            _isConv = reader.TryGetInfo(prefix + "conv_dw.weight", out _);
            if (_isConv)
            {
                _convDw = reader.ReadFloat32(prefix + "conv_dw.weight");
                _pw1 = reader.ReadFloat32(prefix + "conv_pw1.weight");
                _pw2 = reader.ReadFloat32(prefix + "conv_pw2.weight");
                _qW = _kW = _vW = _oW = _qNorm = _kNorm = [];
            }
            else
            {
                _convDw = _pw1 = _pw2 = [];
                _qW = reader.ReadFloat32(prefix + "attn_q.weight");
                _kW = reader.ReadFloat32(prefix + "attn_k.weight");
                _vW = reader.ReadFloat32(prefix + "attn_v.weight");
                _oW = reader.ReadFloat32(prefix + "attn_out.weight");
                _qNorm = reader.ReadFloat32(prefix + "attn_q_norm.weight");
                _kNorm = reader.ReadFloat32(prefix + "attn_k_norm.weight");
            }
            _gateW = reader.ReadFloat32(prefix + "ffn_gate.weight");
            _upW = reader.ReadFloat32(prefix + "ffn_up.weight");
            _downW = reader.ReadFloat32(prefix + "ffn_down.weight");
        }

        public static Block Load(GgufTensorReader reader, int layer) => new(reader, layer);

        public float[] Forward(float[] input, int tokens)
        {
            float[] norm = Scratch(tokens * Hidden);
            RmsNorm(input, _ln1, norm, tokens, Hidden, Eps);

            float[] mixed = _isConv ? ConvMix(norm, tokens) : AttentionMix(norm, tokens);
            var residual = new float[tokens * Hidden];
            for (int i = 0; i < residual.Length; i++) residual[i] = input[i] + mixed[i];

            RmsNorm(residual, _ln2, norm, tokens, Hidden, Eps);
            float[] ffn = Ffn(norm, tokens);
            for (int i = 0; i < residual.Length; i++) residual[i] += ffn[i];
            return residual;
        }

        private float[] ConvMix(float[] x, int tokens)
        {
            int bcxDim = 3 * Hidden;
            var bcx = new float[tokens * bcxDim];
            MatMulBatch(x, _pw1, Hidden, bcxDim, bcx, tokens);

            var bx = new float[tokens * Hidden];
            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < Hidden; h++)
                    bx[t * Hidden + h] = bcx[t * bcxDim + h] * bcx[t * bcxDim + 2 * Hidden + h];

            var conv = new float[tokens * Hidden];
            for (int t = 0; t < tokens; t++)
            {
                for (int h = 0; h < Hidden; h++)
                {
                    float sum = 0;
                    for (int k = 0; k < 3; k++)
                    {
                        int source = t - (2 - k);
                        if (source < 0) continue;
                        sum += bx[source * Hidden + h] * _convDw[k + 3 * h];
                    }
                    conv[t * Hidden + h] = sum;
                }
            }

            for (int t = 0; t < tokens; t++)
                for (int h = 0; h < Hidden; h++)
                    conv[t * Hidden + h] *= bcx[t * bcxDim + Hidden + h];

            var output = new float[tokens * Hidden];
            MatMulBatch(conv, _pw2, Hidden, Hidden, output, tokens);
            return output;
        }

        private float[] AttentionMix(float[] x, int tokens)
        {
            var q = new float[tokens * Hidden];
            var k = new float[tokens * KvHeads * HeadDim];
            var v = new float[tokens * KvHeads * HeadDim];
            MatMulBatch(x, _qW, Hidden, Hidden, q, tokens);
            MatMulBatch(x, _kW, Hidden, KvHeads * HeadDim, k, tokens);
            MatMulBatch(x, _vW, Hidden, KvHeads * HeadDim, v, tokens);

            int group = Heads / KvHeads;
            var output = new float[tokens * Hidden];
            var scores = new float[Window];

            // Normalize and rotate each query/key once; rotating inside the attention loop would
            // apply RoPE again for every later query.
            for (int t = 0; t < tokens; t++)
            {
                for (int head = 0; head < Heads; head++)
                {
                    var row = q.AsSpan(t * Hidden + head * HeadDim, HeadDim);
                    RmsNormVec(row, _qNorm, HeadDim);
                    ApplyRope(row, t, HeadDim);
                }
                for (int head = 0; head < KvHeads; head++)
                {
                    var row = k.AsSpan(t * KvHeads * HeadDim + head * HeadDim, HeadDim);
                    RmsNormVec(row, _kNorm, HeadDim);
                    ApplyRope(row, t, HeadDim);
                }
            }

            for (int t = 0; t < tokens; t++)
            {
                int first = Math.Max(0, t - Window + 1);
                for (int head = 0; head < Heads; head++)
                {
                    int kvHead = head / group;
                    int qOffset = t * Hidden + head * HeadDim;

                    int count = t - first + 1;
                    for (int s = first; s <= t; s++)
                    {
                        int keyOffset = s * KvHeads * HeadDim + kvHead * HeadDim;
                        float dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[qOffset + d] * k[keyOffset + d];
                        scores[s - first] = dot / MathF.Sqrt(HeadDim);
                    }

                    Softmax(scores.AsSpan(0, count));
                    var headOut = output.AsSpan(t * Hidden + head * HeadDim, HeadDim);
                    for (int s = first; s <= t; s++)
                    {
                        float weight = scores[s - first];
                        int valueOffset = s * KvHeads * HeadDim + kvHead * HeadDim;
                        for (int d = 0; d < HeadDim; d++) headOut[d] += weight * v[valueOffset + d];
                    }
                }
            }

            var result = new float[tokens * Hidden];
            MatMulBatch(output, _oW, Hidden, Hidden, result, tokens);
            return result;
        }

        private float[] Ffn(float[] x, int tokens)
        {
            var gate = new float[tokens * FfnDim];
            var up = new float[tokens * FfnDim];
            MatMulBatch(x, _gateW, Hidden, FfnDim, gate, tokens);
            MatMulBatch(x, _upW, Hidden, FfnDim, up, tokens);
            for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];
            var output = new float[tokens * Hidden];
            MatMulBatch(gate, _downW, FfnDim, Hidden, output, tokens);
            return output;
        }

        private float[] Scratch(int size)
        {
            if (_scratch is null || _scratch.Length < size) _scratch = new float[size];
            return _scratch;
        }

        public void Dispose() { }
    }

    private sealed class Projector
    {
        private readonly float[] _lnW, _lnB, _w1, _b1, _w2, _b2;

        public Projector(GgufTensorReader reader)
        {
            _lnW = reader.ReadFloat32("mm.a.mlp.0.weight");
            _lnB = reader.ReadFloat32("mm.a.mlp.0.bias");
            _w1 = reader.ReadFloat32("mm.a.mlp.1.weight");
            _b1 = reader.ReadFloat32("mm.a.mlp.1.bias");
            _w2 = reader.ReadFloat32("mm.a.mlp.3.weight");
            _b2 = reader.ReadFloat32("mm.a.mlp.3.bias");
        }

        public void Forward(float[] input, int tokens, float[] output)
        {
            var norm = new float[Hidden];
            var first = new float[OutputDim];
            for (int t = 0; t < tokens; t++)
            {
                LayerNorm(input.AsSpan(t * Hidden, Hidden), _lnW, _lnB, norm, Hidden, Eps);
                MatMul(norm, _w1, Hidden, OutputDim, first);
                for (int i = 0; i < OutputDim; i++) first[i] = Gelu(first[i] + _b1[i]);
                MatMul(first, _w2, OutputDim, OutputDim, output.AsSpan(t * OutputDim, OutputDim));
                for (int i = 0; i < OutputDim; i++) output[t * OutputDim + i] += _b2[i];
            }
        }
    }

    private static void Conv2d(
        float[] input, int inT, int inF, int inC, int outC,
        float[] weight, float[] bias, float[] output, int outT, int outF, bool depthwise)
    {
        for (int t = 0; t < outT; t++)
        {
            for (int f = 0; f < outF; f++)
            {
                for (int o = 0; o < outC; o++)
                {
                    float sum = bias[o];
                    if (depthwise)
                    {
                        for (int kt = 0; kt < 3; kt++)
                        {
                            int sourceT = t * 2 + kt - 1;
                            if ((uint)sourceT >= (uint)inT) continue;
                            for (int kf = 0; kf < 3; kf++)
                            {
                                int sourceF = f * 2 + kf - 1;
                                if ((uint)sourceF >= (uint)inF) continue;
                                // GGUF stores conv2d kernels as [kW, kH, in, out]; kW is time.
                                sum += input[(sourceT * inF + sourceF) * inC + o] * weight[kt + 3 * kf + 9 * o];
                            }
                        }
                    }
                    else
                    {
                        for (int kt = 0; kt < 3; kt++)
                        {
                            int sourceT = t * 2 + kt - 1;
                            if ((uint)sourceT >= (uint)inT) continue;
                            for (int kf = 0; kf < 3; kf++)
                            {
                                int sourceF = f * 2 + kf - 1;
                                if ((uint)sourceF >= (uint)inF) continue;
                                sum += input[(sourceT * inF + sourceF) * inC] * weight[kt + 3 * kf + 9 * o];
                            }
                        }
                    }
                    output[(t * outF + f) * outC + o] = sum;
                }
            }
        }
    }

    private static void Pointwise(float[] data, int positions, int channels, float[] weight, float[] bias)
    {
        var copy = (float[])data.Clone();
        for (int p = 0; p < positions; p++)
        {
            for (int o = 0; o < channels; o++)
            {
                float sum = bias[o];
                for (int i = 0; i < channels; i++)
                    sum += copy[p * channels + i] * weight[i + channels * o];
                data[p * channels + o] = sum;
            }
        }
    }

    private static void MatMul(ReadOnlySpan<float> x, float[] weight, int inDim, int outDim, Span<float> y)
    {
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0;
            int row = inDim * o;
            for (int i = 0; i < inDim; i++) sum += x[i] * weight[row + i];
            y[o] = sum;
        }
    }

    private static void MatMulBatch(float[] x, float[] weight, int inDim, int outDim, float[] y, int count)
    {
        for (int t = 0; t < count; t++)
            MatMul(x.AsSpan(t * inDim, inDim), weight, inDim, outDim, y.AsSpan(t * outDim, outDim));
    }

    private static void RmsNorm(float[] x, float[] weight, float[] y, int count, int dim, float eps)
    {
        for (int t = 0; t < count; t++) RmsNormVec(x.AsSpan(t * dim, dim), weight, y.AsSpan(t * dim, dim), dim, eps);
    }

    private static void RmsNormVec(ReadOnlySpan<float> x, float[] weight, Span<float> y, int dim, float eps)
    {
        double sum = 0;
        for (int i = 0; i < dim; i++) sum += x[i] * x[i];
        float scale = 1f / MathF.Sqrt((float)(sum / dim) + eps);
        for (int i = 0; i < dim; i++) y[i] = x[i] * scale * weight[i];
    }

    private static void RmsNormVec(Span<float> x, float[] weight, int dim) => RmsNormVec(x, weight, x, dim, Eps);

    private static void RmsNormInPlace(float[] x, float[] weight, int count, int dim, float eps)
    {
        for (int t = 0; t < count; t++)
        {
            var row = x.AsSpan(t * dim, dim);
            RmsNormVec(row, weight, row, dim, eps);
        }
    }

    private static void LayerNorm(ReadOnlySpan<float> x, float[] weight, float[] bias, Span<float> y, int dim, float eps)
    {
        double mean = 0;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        double variance = 0;
        for (int i = 0; i < dim; i++)
        {
            double delta = x[i] - mean;
            variance += delta * delta;
        }
        float scale = 1f / MathF.Sqrt((float)(variance / dim) + eps);
        for (int i = 0; i < dim; i++) y[i] = (float)((x[i] - mean) * scale) * weight[i] + bias[i];
    }

    private static void ApplyRope(Span<float> values, int position, int headDim)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            float frequency = MathF.Exp(-2f * i / headDim * MathF.Log(RopeTheta));
            float angle = position * frequency;
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            float a = values[i];
            float b = values[i + half];
            values[i] = a * cos - b * sin;
            values[i + half] = b * cos + a * sin;
        }
    }

    private static void Softmax(Span<float> values)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < values.Length; i++) if (values[i] > max) max = values[i];
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            sum += values[i];
        }
        float scale = (float)(1.0 / sum);
        for (int i = 0; i < values.Length; i++) values[i] *= scale;
    }

    private static void SiLU(float[] values)
    {
        for (int i = 0; i < values.Length; i++) values[i] = Silu(values[i]);
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x * 0.70710678f));

    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * x);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t
                        + 0.254829592f) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}
