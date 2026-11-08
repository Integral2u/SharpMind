using SharpMind.Model.Audio;
using SharpMind.Model.Format;

namespace SharpMind.Model.Vision;

/// <summary>
/// Native LFM3-VL vision tower: a linear patchifier, fourteen pre-norm blocks alternating
/// gated depthwise 2D convolution and bidirectional grouped-query attention with axial 2D
/// RoPE, and the 2x2 merge projector that maps the patch grid into the text hidden space.
/// It reads the vision tensors from the <c>mmproj</c>, so the text backbone stays the
/// ordinary LFM3 decoder.
/// </summary>
public sealed class Lfm3VlVisionEncoder : IDisposable
{
    public const int PatchSize = 16;
    public const int Merge = 2;
    public const int Hidden = 1024;
    public const int Heads = 16;
    public const int KvHeads = 8;
    public const int HeadDim = 64;
    public const int FfnDim = 2560;
    public const int Layers = 14;
    public const int TextHidden = 2048;
    public const int ProjectorHidden = 4096;
    private const float Eps = 1e-5f;
    private const float ProjectorEps = 1e-6f;
    private const float RopeTheta = 1_000_000f;

    private readonly float[] _patch0;      // [16,16,3,1024]
    private readonly float[] _patch1;      // [16,16,3,1024]
    private readonly float[] _embeddingNorm;
    private readonly Block[] _blocks;
    private readonly float[] _inputNormW;
    private readonly float[] _inputNormB;
    private readonly float[] _projW1;      // [4096,4096]
    private readonly float[] _projB1;
    private readonly float[] _projW2;      // [4096,2048]
    private readonly float[] _projB2;
    private bool _disposed;

    public Lfm3VlVisionEncoder(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _patch0 = reader.ReadFloat32("v.patch_embd.weight");
        _patch1 = reader.ReadFloat32("v.patch_embd.weight.1");
        _embeddingNorm = reader.ReadFloat32("v.embedding_norm.weight");
        _blocks = new Block[Layers];
        for (int layer = 0; layer < Layers; layer++) _blocks[layer] = new Block(reader, layer);
        _inputNormW = reader.ReadFloat32("mm.input_norm.weight");
        _inputNormB = reader.ReadFloat32("mm.input_norm.bias");
        _projW1 = reader.ReadFloat32("mm.0.weight");
        _projB1 = reader.ReadFloat32("mm.0.bias");
        _projW2 = reader.ReadFloat32("mm.2.weight");
        _projB2 = reader.ReadFloat32("mm.2.bias");
    }

    /// <summary>True when the projector carries the LFM3-VL vision tower.</summary>
    public static bool IsVisionProjector(string path)
    {
        try
        {
            using var reader = new GgufTensorReader(path);
            return reader.TryGetInfo("v.patch_embd.weight", out _) &&
                   reader.TryGetInfo("mm.input_norm.weight", out _);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Encodes one RGB image in CHW float form (values 0..1) into merged text-space features.
    /// The output is <c>[tokens * TextHidden]</c>, row-major, with <c>tokens = patches / 4</c>.
    /// </summary>
    public float[] Encode(ReadOnlySpan<float> chw, int height, int width)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (chw.Length != 3 * height * width)
            throw new ArgumentException("Expected a 3-channel CHW image.", nameof(chw));
        if (height % (PatchSize * Merge) != 0 || width % (PatchSize * Merge) != 0)
            throw new ArgumentException($"Image size {width}x{height} must be a multiple of {PatchSize * Merge}.");

        int gridH = height / PatchSize;
        int gridW = width / PatchSize;
        int patches = gridH * gridW;
        int blockH = gridH / Merge;
        int blockW = gridW / Merge;

        // Patchify in merge-block order: [block_h, block_w, sub_h, sub_w, channel, patch_h, patch_w].
        var patches768 = new float[patches * 3 * PatchSize * PatchSize];
        for (int bh = 0; bh < blockH; bh++)
        {
            for (int bw = 0; bw < blockW; bw++)
            {
                for (int sh = 0; sh < Merge; sh++)
                {
                    for (int sw = 0; sw < Merge; sw++)
                    {
                        int token = ((bh * blockW + bw) * Merge + sh) * Merge + sw;
                        int y0 = (bh * Merge + sh) * PatchSize;
                        int x0 = (bw * Merge + sw) * PatchSize;
                        int offset = token * 3 * PatchSize * PatchSize;
                        for (int c = 0; c < 3; c++)
                        {
                            int channelBase = c * height * width;
                            for (int py = 0; py < PatchSize; py++)
                            {
                                int src = channelBase + (y0 + py) * width + x0;
                                int dst = offset + c * PatchSize * PatchSize + py * PatchSize;
                                for (int px = 0; px < PatchSize; px++)
                                    patches768[dst + px] = chw[src + px];
                            }
                        }
                    }
                }
            }
        }

        // The patchifier weight is split into the two temporal slots; the image is
        // duplicated across them, so both projections read the same patch.
        var projected0 = new float[patches * Hidden];
        var projected1 = new float[patches * Hidden];
        int patchDim = 3 * PatchSize * PatchSize;
        Lfm3AudioCodeGenerator.MatMulBatch(patches768, _patch0, patchDim, Hidden, projected0, patches);
        Lfm3AudioCodeGenerator.MatMulBatch(patches768, _patch1, patchDim, Hidden, projected1, patches);
        var hidden = new float[patches * Hidden];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = projected0[i] + projected1[i];

        Lfm3AudioCodeGenerator.RmsNorm(hidden, _embeddingNorm, patches, Hidden, Eps);

        // Per-patch (row, column) grid position, used by the axial 2D RoPE.
        var posH = new int[patches];
        var posW = new int[patches];
        for (int bh = 0; bh < blockH; bh++)
            for (int bw = 0; bw < blockW; bw++)
                for (int sh = 0; sh < Merge; sh++)
                    for (int sw = 0; sw < Merge; sw++)
                    {
                        int token = ((bh * blockW + bw) * Merge + sh) * Merge + sw;
                        posH[token] = bh * Merge + sh;
                        posW[token] = bw * Merge + sw;
                    }

        for (int layer = 0; layer < Layers; layer++)
            hidden = _blocks[layer].Forward(hidden, posH, posW, blockH, blockW);

        // Projector: LayerNorm -> 2x2 merge (4 * 1024) -> 4096 -> GELU -> 2048.
        int tokens = patches / (Merge * Merge);
        var features = new float[tokens * TextHidden];
        var normed = new float[Hidden];
        var merged = new float[tokens * ProjectorHidden];
        for (int t = 0; t < tokens; t++)
        {
            int baseToken = t * Merge * Merge;
            for (int sub = 0; sub < Merge * Merge; sub++)
            {
                int src = (baseToken + sub) * Hidden;
                LayerNorm(hidden.AsSpan(src, Hidden), _inputNormW, _inputNormB, normed, ProjectorEps);
                normed.CopyTo(merged.AsSpan(t * ProjectorHidden + sub * Hidden, Hidden));
            }
        }

        var projectedProj = new float[tokens * ProjectorHidden];
        Lfm3AudioCodeGenerator.MatMulBatch(merged, _projW1, ProjectorHidden, ProjectorHidden, projectedProj, tokens);
        for (int i = 0; i < projectedProj.Length; i++)
            projectedProj[i] = Gelu(projectedProj[i] + _projB1[i % ProjectorHidden]);

        Lfm3AudioCodeGenerator.MatMulBatch(projectedProj, _projW2, ProjectorHidden, TextHidden, features, tokens);
        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < TextHidden; i++)
                features[t * TextHidden + i] += _projB2[i];

        return features;
    }

    /// <summary>Exact GELU through the Abramowitz-Stegun 7.1.26 error-function approximation.</summary>
    private static float Gelu(float value)
    {
        float x = value / MathF.Sqrt(2f);
        float t = 1f / (1f + 0.3275911f * MathF.Abs(x));
        float erf = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                           - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
        if (x < 0) erf = -erf;
        return 0.5f * value * (1f + erf);
    }

    private static void LayerNorm(
        ReadOnlySpan<float> x, float[] weight, float[] bias, Span<float> y, float eps)
    {
        double sum = 0;
        for (int i = 0; i < x.Length; i++) sum += x[i];
        float mean = (float)(sum / x.Length);
        double variance = 0;
        for (int i = 0; i < x.Length; i++)
        {
            float delta = x[i] - mean;
            variance += (double)delta * delta;
        }
        float scale = 1f / MathF.Sqrt((float)(variance / x.Length) + eps);
        for (int i = 0; i < x.Length; i++) y[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }

    private static void ApplyVisionRope(Span<float> values, int positionH, int positionW)
    {
        const int half = HeadDim / 2;
        for (int i = 0; i < half; i++)
        {
            int axisPosition = i < half / 2 ? positionH : positionW;
            int frequencyIndex = i % (half / 2);
            float frequency = MathF.Exp(-2f * frequencyIndex / half * MathF.Log(RopeTheta));
            float angle = axisPosition * frequency;
            float cos = MathF.Cos(angle);
            float sin = MathF.Sin(angle);
            float a = values[i];
            float b = values[i + half];
            values[i] = a * cos - b * sin;
            values[i + half] = b * cos + a * sin;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Block block in _blocks) block.Dispose();
    }

    private sealed class Block : IDisposable
    {
        private readonly bool _isConv;
        private readonly float[] _ln1, _ln2;
        private readonly float[] _qW, _kW, _vW, _oW, _qNorm, _kNorm;
        private readonly float[] _inProj, _spatial, _outProj;
        private readonly float[] _gateW, _upW, _downW;

        public Block(GgufTensorReader reader, int layer)
        {
            string prefix = $"v.blk.{layer}.";
            _ln1 = reader.ReadFloat32(prefix + "ln1.weight");
            _ln2 = reader.ReadFloat32(prefix + "ln2.weight");
            _gateW = reader.ReadFloat32(prefix + "ffn_gate.weight");
            _upW = reader.ReadFloat32(prefix + "ffn_up.weight");
            _downW = reader.ReadFloat32(prefix + "ffn_down.weight");
            _isConv = reader.TryGetInfo(prefix + "conv2d_in_proj.weight", out _);
            if (_isConv)
            {
                _inProj = reader.ReadFloat32(prefix + "conv2d_in_proj.weight");
                _spatial = reader.ReadFloat32(prefix + "conv2d_spatial.weight");
                _outProj = reader.ReadFloat32(prefix + "conv2d_out_proj.weight");
                _qW = _kW = _vW = _oW = _qNorm = _kNorm = [];
            }
            else
            {
                _inProj = _spatial = _outProj = [];
                _qW = reader.ReadFloat32(prefix + "attn_q.weight");
                _kW = reader.ReadFloat32(prefix + "attn_k.weight");
                _vW = reader.ReadFloat32(prefix + "attn_v.weight");
                _oW = reader.ReadFloat32(prefix + "attn_out.weight");
                _qNorm = reader.ReadFloat32(prefix + "attn_q_norm.weight");
                _kNorm = reader.ReadFloat32(prefix + "attn_k_norm.weight");
            }
        }

        public float[] Forward(float[] input, int[] posH, int[] posW, int blockH, int blockW)
        {
            int patches = posH.Length;
            var norm = new float[input.Length];
            Lfm3AudioCodeGenerator.RmsNorm(input, _ln1, norm, patches, Hidden, Eps);
            float[] mixed = _isConv ? Conv(norm, blockH, blockW) : Attention(norm, posH, posW);
            var residual = new float[input.Length];
            for (int i = 0; i < residual.Length; i++) residual[i] = input[i] + mixed[i];

            var norm2 = new float[residual.Length];
            Lfm3AudioCodeGenerator.RmsNorm(residual, _ln2, norm2, patches, Hidden, Eps);
            float[] ffn = Ffn(norm2, patches);
            for (int i = 0; i < residual.Length; i++) residual[i] += ffn[i];
            return residual;
        }

        private float[] Attention(float[] x, int[] posH, int[] posW)
        {
            int patches = posH.Length;
            var q = new float[patches * Hidden];
            var k = new float[patches * KvHeads * HeadDim];
            var v = new float[patches * KvHeads * HeadDim];
            Lfm3AudioCodeGenerator.MatMulBatch(x, _qW, Hidden, Hidden, q, patches);
            Lfm3AudioCodeGenerator.MatMulBatch(x, _kW, Hidden, KvHeads * HeadDim, k, patches);
            Lfm3AudioCodeGenerator.MatMulBatch(x, _vW, Hidden, KvHeads * HeadDim, v, patches);
            for (int t = 0; t < patches; t++)
            {
                int offset = t * Hidden;
                for (int h = 0; h < Heads; h++)
                {
                    var row = q.AsSpan(offset + h * HeadDim, HeadDim);
                    Lfm3AudioCodeGenerator.RmsNorm(row, _qNorm, row, Eps);
                    ApplyVisionRope(row, posH[t], posW[t]);
                }
                for (int h = 0; h < KvHeads; h++)
                {
                    var row = k.AsSpan(t * KvHeads * HeadDim + h * HeadDim, HeadDim);
                    Lfm3AudioCodeGenerator.RmsNorm(row, _kNorm, row, Eps);
                    ApplyVisionRope(row, posH[t], posW[t]);
                }
            }

            var output = new float[patches * Hidden];
            int group = Heads / KvHeads;
            Parallel.For(0, patches, t =>
            {
                var scores = new float[patches];
                for (int head = 0; head < Heads; head++)
                {
                    int kvHead = head / group;
                    int qOffset = t * Hidden + head * HeadDim;
                    for (int key = 0; key < patches; key++)
                    {
                        int kOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[qOffset + d] * k[kOffset + d];
                        scores[key] = dot / MathF.Sqrt(HeadDim);
                    }
                    Softmax(scores);
                    var row = output.AsSpan(qOffset, HeadDim);
                    for (int key = 0; key < patches; key++)
                    {
                        int vOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float weight = scores[key];
                        for (int d = 0; d < HeadDim; d++) row[d] += weight * v[vOffset + d];
                    }
                }
            });

            var result = new float[patches * Hidden];
            Lfm3AudioCodeGenerator.MatMulBatch(output, _oW, Hidden, Hidden, result, patches);
            return result;
        }

        private float[] Conv(float[] x, int blockH, int blockW)
        {
            int patches = blockH * blockW * Merge * Merge;
            var bcx = new float[patches * 3 * Hidden];
            Lfm3AudioCodeGenerator.MatMulBatch(x, _inProj, Hidden, 3 * Hidden, bcx, patches);

            // gated = B * x, laid out in merge-block order.
            var gated = new float[patches * Hidden];
            for (int t = 0; t < patches; t++)
                for (int c = 0; c < Hidden; c++)
                    gated[t * Hidden + c] = bcx[t * 3 * Hidden + c] * bcx[t * 3 * Hidden + 2 * Hidden + c];

            // Unmerge to the raster patch grid: [channel, height, width].
            int height = blockH * Merge;
            int width = blockW * Merge;
            var raster = new float[Hidden * height * width];
            for (int bh = 0; bh < blockH; bh++)
                for (int bw = 0; bw < blockW; bw++)
                    for (int sh = 0; sh < Merge; sh++)
                        for (int sw = 0; sw < Merge; sw++)
                        {
                            int token = ((bh * blockW + bw) * Merge + sh) * Merge + sw;
                            int y = bh * Merge + sh;
                            int px = bw * Merge + sw;
                            for (int c = 0; c < Hidden; c++)
                                raster[(c * height + y) * width + px] = gated[token * Hidden + c];
                        }

            // Depthwise 3x3 convolution, padding 1.
            var convolved = new float[raster.Length];
            Parallel.For(0, Hidden, c =>
            {
                int channelBase = c * height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int px = 0; px < width; px++)
                    {
                        float sum = 0;
                        for (int ky = 0; ky < 3; ky++)
                        {
                            int sy = y + ky - 1;
                            if (sy < 0 || sy >= height) continue;
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int sx = px + kx - 1;
                                if (sx < 0 || sx >= width) continue;
                                // conv2d_spatial is [3, 3, 1, hidden]: kx + 3*ky + 9*c.
                                sum += raster[channelBase + sy * width + sx] * _spatial[kx + 3 * ky + 9 * c];
                            }
                        }
                        convolved[channelBase + y * width + px] = sum;
                    }
                }
            });

            // Re-merge to block order and apply the C gate.
            var merged = new float[patches * Hidden];
            for (int bh = 0; bh < blockH; bh++)
                for (int bw = 0; bw < blockW; bw++)
                    for (int sh = 0; sh < Merge; sh++)
                        for (int sw = 0; sw < Merge; sw++)
                        {
                            int token = ((bh * blockW + bw) * Merge + sh) * Merge + sw;
                            int y = bh * Merge + sh;
                            int px = bw * Merge + sw;
                            for (int c = 0; c < Hidden; c++)
                                merged[token * Hidden + c] =
                                    convolved[(c * height + y) * width + px] * bcx[token * 3 * Hidden + Hidden + c];
                        }

            var output = new float[patches * Hidden];
            Lfm3AudioCodeGenerator.MatMulBatch(merged, _outProj, Hidden, Hidden, output, patches);
            return output;
        }

        private float[] Ffn(float[] x, int patches)
        {
            var gate = new float[patches * FfnDim];
            var up = new float[patches * FfnDim];
            Lfm3AudioCodeGenerator.MatMulBatch(x, _gateW, Hidden, FfnDim, gate, patches);
            Lfm3AudioCodeGenerator.MatMulBatch(x, _upW, Hidden, FfnDim, up, patches);
            for (int i = 0; i < gate.Length; i++)
            {
                float value = gate[i];
                gate[i] = value / (1f + MathF.Exp(-value)) * up[i];
            }

            var output = new float[patches * Hidden];
            Lfm3AudioCodeGenerator.MatMulBatch(gate, _downW, FfnDim, Hidden, output, patches);
            return output;
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

        public void Dispose() { }
    }
}
