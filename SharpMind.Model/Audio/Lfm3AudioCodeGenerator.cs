using System.Numerics;
using SharpMind.Model.Format;

namespace SharpMind.Model.Audio;

/// <summary>
/// LFM3-Audio speech-code generator. For one language-model hidden state it runs the
/// three-layer bidirectional depthformer over twelve codebook slots and fills the
/// codebooks with MaskGIT: three passes of four codebooks, each pass scoring the
/// still-masked slots and sampling with the model card's speech settings.
/// </summary>
public sealed class Lfm3AudioCodeGenerator : IDisposable
{
    public const int Codebooks = 12;
    public const int CodebookSize = 2017;
    public const int EosCode = 2016;
    public const int Hidden = 1024;
    public const int TextHidden = 2048;

    private const int Heads = 32;
    private const int KvHeads = 8;
    private const int HeadDim = 32;
    private const int FfnDim = 2816;
    private const int Layers = 3;
    private const float Eps = 1e-5f;
    private const float RopeTheta = 1_000_000f;

    /// <summary>Sampling temperature of the depth decoder (the checkpoint uses 1.0).</summary>
    public float Temperature { get; set; } = 1.0f;

    /// <summary>Top-k of the depth decoder (the checkpoint uses 128).</summary>
    public int TopK { get; set; } = 128;

    /// <summary>Per-codebook repetition penalty (the checkpoint uses 1.1).</summary>
    public float RepetitionPenalty { get; set; } = 1.1f;

    /// <summary>Frames of code history covered by the penalty (4096 in the checkpoint).</summary>
    public int RepetitionWindow { get; set; } = 4096;

    private readonly float[] _projW;
    private readonly float[] _projB;
    private readonly float[] _embd;
    private readonly float[] _outEmbd;
    private readonly float[] _maskEmbedding;
    private readonly float[] _headNorm;
    private readonly float[] _outputNorm;
    private readonly DepthBlock[] _blocks;
    private bool _disposed;

    public Lfm3AudioCodeGenerator(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _projW = reader.ReadFloat32("a.gen.code.proj_in.weight");
        _projB = reader.ReadFloat32("a.gen.code.proj_in.bias");
        _embd = reader.ReadFloat32("a.gen.code.embd.weight");
        _outEmbd = reader.ReadFloat32("a.gen.code.out_embd.weight");
        _maskEmbedding = reader.ReadFloat32("a.gen.code.mask_embd");
        _headNorm = reader.ReadFloat32("a.gen.code.head_norm.weight");
        _outputNorm = reader.ReadFloat32("a.gen.code.output_norm.weight");
        _blocks = new DepthBlock[Layers];
        for (int layer = 0; layer < Layers; layer++)
            _blocks[layer] = new DepthBlock(reader, layer);
    }

    /// <summary>
    /// Generates the twelve codes of one speech frame. <paramref name="history"/> holds
    /// the previously generated codes for each codebook (used by the repetition penalty).
    /// Returns <see cref="EosCode"/> in every codebook for the end-of-speech frame.
    /// </summary>
    public int[] GenerateFrame(ReadOnlySpan<float> hidden, IReadOnlyList<int>[] history, Random random)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var projected = new float[Codebooks * Hidden];
        MatMul(hidden, _projW, TextHidden, Codebooks * Hidden, projected);
        for (int i = 0; i < projected.Length; i++) projected[i] += _projB[i];

        var codes = new int[Codebooks];
        Array.Fill(codes, -1);
        var revealed = new bool[Codebooks];
        var order = new int[Codebooks];
        for (int i = 0; i < Codebooks; i++) order[i] = i;
        Shuffle(order, random);

        for (int pass = 0; pass < 3; pass++)
        {
            float[] logits = DepthLogits(projected, codes, revealed);
            for (int p = 0; p < 4; p++)
            {
                int cb = order[pass * 4 + p];
                codes[cb] = Sample(logits, cb, history[cb], random);
                revealed[cb] = true;
            }
        }

        return codes;
    }

    /// <summary>
    /// Computes the per-codebook logits for one depth-decoder pass. Exposed for diagnostics; the
    /// generation loop uses it through <see cref="GenerateFrame"/>.
    /// </summary>
    public float[] ComputeLogits(ReadOnlySpan<float> hidden, int[] codes, bool[] revealed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var projected = new float[Codebooks * Hidden];
        MatMul(hidden, _projW, TextHidden, Codebooks * Hidden, projected);
        for (int i = 0; i < projected.Length; i++) projected[i] += _projB[i];
        return DepthLogits(projected, codes, revealed);
    }

    /// <summary>Sums the codebook feedback embeddings for the next text-model input.</summary>
    public float[] Feedback(ReadOnlySpan<int> codes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var feedback = new float[TextHidden];
        for (int cb = 0; cb < Codebooks; cb++)
        {
            int code = codes[cb];
            if (code < 0 || code >= CodebookSize) continue;
            int row = TextHidden * (code + CodebookSize * cb);
            for (int d = 0; d < TextHidden; d++) feedback[d] += _outEmbd[row + d];
        }
        return feedback;
    }

    private float[] DepthLogits(float[] projected, int[] codes, bool[] revealed)
    {
        var slots = new float[Codebooks * Hidden];
        for (int cb = 0; cb < Codebooks; cb++)
        {
            int baseOffset = cb * Hidden;
            for (int d = 0; d < Hidden; d++) slots[baseOffset + d] = projected[baseOffset + d];
            if (revealed[cb])
            {
                int row = Hidden * (codes[cb] + CodebookSize * cb);
                for (int d = 0; d < Hidden; d++) slots[baseOffset + d] += _embd[row + d];
            }
            else
            {
                for (int d = 0; d < Hidden; d++) slots[baseOffset + d] += _maskEmbedding[d];
            }
        }

        for (int layer = 0; layer < Layers; layer++)
            slots = _blocks[layer].Forward(slots);

        RmsNorm(slots, _outputNorm, Codebooks, Hidden, Eps);

        var logits = new float[Codebooks * CodebookSize];
        var normalized = new float[Hidden];
        for (int cb = 0; cb < Codebooks; cb++)
        {
            var headWeight = _headNorm.AsSpan(cb * Hidden, Hidden);
            RmsNorm(slots.AsSpan(cb * Hidden, Hidden), headWeight, normalized, Eps);
            for (int code = 0; code < CodebookSize; code++)
            {
                int row = Hidden * (code + CodebookSize * cb);
                double sum = 0;
                for (int d = 0; d < Hidden; d++) sum += _embd[row + d] * normalized[d];
                logits[cb * CodebookSize + code] = (float)sum;
            }
        }
        return logits;
    }

    private int Sample(float[] logits, int codebook, IReadOnlyList<int> history, Random random)
    {
        int offset = codebook * CodebookSize;
        var scores = new float[CodebookSize];
        Array.Copy(logits, offset, scores, 0, CodebookSize);

        // transformers' RepetitionPenaltyLogitsProcessor gathers the original score for each
        // history token and scatters once, so a repeated code is penalised once, not per occurrence.
        if (RepetitionPenalty != 1.0f && history.Count > 0)
        {
            int start = Math.Max(0, history.Count - RepetitionWindow);
            var seen = new bool[CodebookSize];
            for (int i = start; i < history.Count; i++)
            {
                int code = history[i];
                if (code < 0 || code >= CodebookSize || seen[code]) continue;
                seen[code] = true;
                scores[code] = scores[code] > 0 ? scores[code] / RepetitionPenalty : scores[code] * RepetitionPenalty;
            }
        }

        if (Temperature > 0)
            for (int i = 0; i < scores.Length; i++) scores[i] /= Temperature;

        if (TopK > 0 && TopK < scores.Length)
        {
            var sorted = (float[])scores.Clone();
            Array.Sort(sorted);
            float threshold = sorted[^TopK];
            for (int i = 0; i < scores.Length; i++)
                if (scores[i] < threshold) scores[i] = float.NegativeInfinity;
        }

        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        double total = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = MathF.Exp(scores[i] - max);
            total += scores[i];
        }

        double pick = random.NextDouble() * total;
        double cumulative = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            cumulative += scores[i];
            if (pick <= cumulative) return i;
        }
        return scores.Length - 1;
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    internal static void MatMul(ReadOnlySpan<float> x, float[] weight, int inDim, int outDim, Span<float> y)
    {
        // The depth decoder and the codec are the only float-matmul users; a wide
        // projection is worth fanning out over the two cores the audio loop runs on.
        if (outDim >= 512 && inDim >= 256)
        {
            float[] xa = x.ToArray();
            float[] ya = new float[outDim];
            Parallel.For(0, outDim, o => ya[o] = Dot(xa, weight.AsSpan(inDim * o, inDim)));
            ya.CopyTo(y);
            return;
        }

        ReadOnlySpan<float> w = weight;
        for (int o = 0; o < outDim; o++)
            y[o] = Dot(x, w.Slice(inDim * o, inDim));
    }

    /// <summary>
    /// Batched float matmul: <paramref name="x"/> is <c>[rows, inDim]</c> and
    /// <paramref name="y"/> is <c>[rows, outDim]</c>. Rows are independent, so the work
    /// fans out across the cores without one parallel region per token.
    /// </summary>
    internal static void MatMulBatch(float[] x, float[] weight, int inDim, int outDim, float[] y, int rows)
    {
        if (rows >= 4)
        {
            Parallel.For(0, rows, r =>
            {
                int xOffset = r * inDim;
                int yOffset = r * outDim;
                for (int o = 0; o < outDim; o++)
                    y[yOffset + o] = Dot(x.AsSpan(xOffset, inDim), weight.AsSpan(o * inDim, inDim));
            });
            return;
        }

        for (int r = 0; r < rows; r++)
            MatMul(x.AsSpan(r * inDim, inDim), weight, inDim, outDim, y.AsSpan(r * outDim, outDim));
    }

    /// <summary>Hardware-accelerated dot product; the audio components are small enough to run in float.</summary>
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int width = Vector<float>.Count;
        int i = 0;
        Vector<float> acc = Vector<float>.Zero;
        for (; i <= a.Length - width; i += width)
            acc += new Vector<float>(a[i..]) * new Vector<float>(b[i..]);
        float sum = Vector.Dot(acc, Vector<float>.One);
        for (; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> y, float eps)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++) sum += x[i] * x[i];
        float scale = 1f / MathF.Sqrt((float)(sum / x.Length) + eps);
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * scale * weight[i];
    }

    internal static void RmsNorm(float[] x, float[] weight, int count, int dim, float eps)
    {
        for (int i = 0; i < count; i++)
            RmsNorm(x.AsSpan(i * dim, dim), weight, x.AsSpan(i * dim, dim), eps);
    }

    internal static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, float[] y, int count, int dim, float eps)
    {
        for (int i = 0; i < count; i++)
            RmsNorm(x.Slice(i * dim, dim), weight, y.AsSpan(i * dim, dim), eps);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (DepthBlock block in _blocks) block.Dispose();
    }

    private sealed class DepthBlock : IDisposable
    {
        // The depth transformer has use_sandwich_norm=False, so the checkpoint ships only
        // the two pre-norms; unlike the codec decoder there are no post-attention/post-FFN norms.
        private readonly float[] _ln1, _ln2;
        private readonly float[] _qW, _kW, _vW, _oW, _qNorm, _kNorm, _gate;
        private readonly float[] _gateW, _upW, _downW;

        public DepthBlock(GgufTensorReader reader, int layer)
        {
            string prefix = $"a.gen.code.blk.{layer}.";
            _ln1 = reader.ReadFloat32(prefix + "ln1.weight");
            _ln2 = reader.ReadFloat32(prefix + "ln2.weight");
            _qW = reader.ReadFloat32(prefix + "attn_q.weight");
            _kW = reader.ReadFloat32(prefix + "attn_k.weight");
            _vW = reader.ReadFloat32(prefix + "attn_v.weight");
            _oW = reader.ReadFloat32(prefix + "attn_out.weight");
            _qNorm = reader.ReadFloat32(prefix + "attn_q_norm.weight");
            _kNorm = reader.ReadFloat32(prefix + "attn_k_norm.weight");
            _gate = reader.ReadFloat32(prefix + "attn_gate.weight");
            _gateW = reader.ReadFloat32(prefix + "ffn_gate.weight");
            _upW = reader.ReadFloat32(prefix + "ffn_up.weight");
            _downW = reader.ReadFloat32(prefix + "ffn_down.weight");
        }

        public float[] Forward(float[] input)
        {
            var norm = new float[input.Length];
            RmsNorm(input, _ln1, norm, Codebooks, Hidden, Eps);

            var mixed = Attention(norm);
            var residual = new float[input.Length];
            for (int i = 0; i < residual.Length; i++) residual[i] = input[i] + mixed[i];

            var norm2 = new float[residual.Length];
            RmsNorm(residual, _ln2, norm2, Codebooks, Hidden, Eps);
            float[] ffn = Ffn(norm2);
            for (int i = 0; i < residual.Length; i++) residual[i] += ffn[i];
            return residual;
        }

        private float[] Attention(float[] x)
        {
            var q = new float[Codebooks * Hidden];
            var k = new float[Codebooks * KvHeads * HeadDim];
            var v = new float[Codebooks * KvHeads * HeadDim];
            var gate = new float[Codebooks * Heads];
            for (int slot = 0; slot < Codebooks; slot++)
            {
                int offset = slot * Hidden;
                MatMul(x.AsSpan(offset, Hidden), _qW, Hidden, Hidden, q.AsSpan(offset, Hidden));
                MatMul(x.AsSpan(offset, Hidden), _kW, Hidden, KvHeads * HeadDim,
                    k.AsSpan(slot * KvHeads * HeadDim, KvHeads * HeadDim));
                MatMul(x.AsSpan(offset, Hidden), _vW, Hidden, KvHeads * HeadDim,
                    v.AsSpan(slot * KvHeads * HeadDim, KvHeads * HeadDim));
                MatMul(x.AsSpan(offset, Hidden), _gate, Hidden, Heads, gate.AsSpan(slot * Heads, Heads));
                for (int h = 0; h < Heads; h++)
                {
                    var row = q.AsSpan(offset + h * HeadDim, HeadDim);
                    RmsNorm(row, _qNorm, row, Eps);
                    ApplyRope(row, slot);
                }
                for (int h = 0; h < KvHeads; h++)
                {
                    var row = k.AsSpan(slot * KvHeads * HeadDim + h * HeadDim, HeadDim);
                    RmsNorm(row, _kNorm, row, Eps);
                    ApplyRope(row, slot);
                }
            }

            var output = new float[Codebooks * Hidden];
            var scores = new float[Codebooks];
            int group = Heads / KvHeads;
            for (int slot = 0; slot < Codebooks; slot++)
            {
                for (int head = 0; head < Heads; head++)
                {
                    int kvHead = head / group;
                    int qOffset = slot * Hidden + head * HeadDim;
                    for (int key = 0; key < Codebooks; key++)
                    {
                        int kOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[qOffset + d] * k[kOffset + d];
                        scores[key] = dot / MathF.Sqrt(HeadDim);
                    }
                    Softmax(scores);
                    float headGate = Sigmoid(gate[slot * Heads + head]);
                    var row = output.AsSpan(qOffset, HeadDim);
                    for (int key = 0; key < Codebooks; key++)
                    {
                        int vOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float weight = scores[key] * headGate;
                        for (int d = 0; d < HeadDim; d++) row[d] += weight * v[vOffset + d];
                    }
                }
            }

            var result = new float[Codebooks * Hidden];
            for (int slot = 0; slot < Codebooks; slot++)
                MatMul(output.AsSpan(slot * Hidden, Hidden), _oW, Hidden, Hidden, result.AsSpan(slot * Hidden, Hidden));
            return result;
        }

        private float[] Ffn(float[] x)
        {
            var gate = new float[Codebooks * FfnDim];
            var up = new float[Codebooks * FfnDim];
            for (int slot = 0; slot < Codebooks; slot++)
            {
                int offset = slot * Hidden;
                MatMul(x.AsSpan(offset, Hidden), _gateW, Hidden, FfnDim, gate.AsSpan(slot * FfnDim, FfnDim));
                MatMul(x.AsSpan(offset, Hidden), _upW, Hidden, FfnDim, up.AsSpan(slot * FfnDim, FfnDim));
            }
            for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];
            var output = new float[Codebooks * Hidden];
            for (int slot = 0; slot < Codebooks; slot++)
                MatMul(gate.AsSpan(slot * FfnDim, FfnDim), _downW, FfnDim, Hidden, output.AsSpan(slot * Hidden, Hidden));
            return output;
        }

        private static void ApplyRope(Span<float> values, int position)
        {
            int half = HeadDim / 2;
            for (int i = 0; i < half; i++)
            {
                float frequency = MathF.Exp(-2f * i / HeadDim * MathF.Log(RopeTheta));
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

        private static float Silu(float x) => x / (1f + MathF.Exp(-x));

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        public void Dispose() { }
    }
}
