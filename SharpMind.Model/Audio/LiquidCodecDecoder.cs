using System.Numerics;
using SharpMind.Model.Format;

namespace SharpMind.Model.Audio;

/// <summary>
/// LiquidCodec speech decoder. Turns 12 codebooks per frame into 24 kHz audio:
/// FSQ code lookup, a 256-wide latent, nearest upsampling by six STFT frames per
/// codec frame, an eight-layer causal LFM decoder (window 30, RoPE 1e6) and an
/// inverse STFT with a 1280-sample window and 320-sample hop.
/// </summary>
public sealed class LiquidCodecDecoder : IDisposable
{
    public const int Codebooks = 12;
    public const int CodebookSize = 2016;
    public const int SampleRate = 24_000;
    public const int NFft = 1280;
    public const int Hop = 320;
    public const int FrameStack = 6;
    public const int Bins = NFft / 2 + 1; // 641

    private const int Hidden = 512;
    private const int FfnDim = 2304;
    private const int Heads = 8;
    private const int KvHeads = 2;
    private const int HeadDim = 64;
    private const int Window = 30;
    private const float Eps = 1e-5f;
    private const float RopeTheta = 1_000_000f;

    private readonly float[] _codeTable;   // [4, 2016]
    private readonly float[] _outProjW;    // [48, 256]
    private readonly float[] _outProjB;
    private readonly float[] _inProjW;     // [256, 512]
    private readonly float[] _inProjB;
    private readonly float[] _outputNorm;
    private readonly float[] _finalW;      // [512, 1282]
    private readonly float[] _finalB;
    private readonly CodecBlock[] _blocks;
    private static readonly double[] HannWindow = BuildHannWindow();
    private bool _disposed;

    public LiquidCodecDecoder(GgufTensorReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _codeTable = reader.ReadFloat32("a.gen.wav.quant.first.codebook.weight");
        _outProjW = reader.ReadFloat32("a.gen.wav.quant.first.out_proj.weight");
        _outProjB = reader.ReadFloat32("a.gen.wav.quant.first.out_proj.bias");
        _inProjW = reader.ReadFloat32("a.gen.wav.tfm.in_proj.weight");
        _inProjB = reader.ReadFloat32("a.gen.wav.tfm.in_proj.bias");
        _outputNorm = reader.ReadFloat32("a.gen.wav.tfm.output_norm.weight");
        _finalW = reader.ReadFloat32("a.gen.wav.tfm.out_proj.weight");
        _finalB = reader.ReadFloat32("a.gen.wav.tfm.out_proj.bias");
        _blocks = new CodecBlock[8];
        for (int layer = 0; layer < 8; layer++)
            _blocks[layer] = new CodecBlock(reader, layer);
    }

    private static double[] BuildHannWindow()
    {
        var window = new double[NFft];
        for (int i = 0; i < NFft; i++)
            window[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / NFft);
        return window;
    }

    /// <summary>Decodes codec frames (each twelve codes) into 24 kHz float samples.</summary>
    public float[] Decode(IReadOnlyList<int[]> frames)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (frames.Count == 0) return [];

        int frameCount = frames.Count;
        var latents = new float[frameCount * 48];
        for (int f = 0; f < frameCount; f++)
        {
            int[] codes = frames[f];
            for (int cb = 0; cb < Codebooks; cb++)
            {
                int code = codes[cb];
                if (code < 0 || code >= CodebookSize) continue;
                for (int level = 0; level < 4; level++)
                    latents[f * 48 + cb * 4 + level] = _codeTable[level + 4 * code];
            }
        }

        var projected = new float[frameCount * 256];
        for (int f = 0; f < frameCount; f++)
        {
            Lfm3AudioCodeGenerator.MatMul(latents.AsSpan(f * 48, 48), _outProjW, 48, 256, projected.AsSpan(f * 256, 256));
            for (int i = 0; i < 256; i++) projected[f * 256 + i] += _outProjB[i];
        }

        var embeddings = new float[frameCount * Hidden];
        for (int f = 0; f < frameCount; f++)
        {
            Lfm3AudioCodeGenerator.MatMul(projected.AsSpan(f * 256, 256), _inProjW, 256, Hidden,
                embeddings.AsSpan(f * Hidden, Hidden));
            for (int i = 0; i < Hidden; i++) embeddings[f * Hidden + i] += _inProjB[i];
        }

        int positions = frameCount * FrameStack;
        var hidden = new float[positions * Hidden];
        for (int p = 0; p < positions; p++)
            embeddings.AsSpan((p / FrameStack) * Hidden, Hidden)
                .CopyTo(hidden.AsSpan(p * Hidden, Hidden));

        for (int layer = 0; layer < _blocks.Length; layer++)
            hidden = _blocks[layer].Forward(hidden, positions);

        Lfm3AudioCodeGenerator.RmsNorm(hidden, _outputNorm, positions, Hidden, Eps);

        var spectrum = new float[positions * 2 * Bins];
        for (int p = 0; p < positions; p++)
        {
            Lfm3AudioCodeGenerator.MatMul(hidden.AsSpan(p * Hidden, Hidden), _finalW, Hidden, 2 * Bins,
                spectrum.AsSpan(p * 2 * Bins, 2 * Bins));
            for (int i = 0; i < 2 * Bins; i++) spectrum[p * 2 * Bins + i] += _finalB[i];
        }

        return InverseStft(spectrum, positions);
    }

    /// <summary>
    /// Overlap-adds the decoder's polar STFT frames back to 24 kHz mono. Exposed so tests can
    /// exercise the window, hop and trim without the neural decoder.
    /// </summary>
    public static float[] InverseStft(ReadOnlySpan<float> spectrum, int positions)
    {
        if (positions <= 0) return [];
        if (spectrum.Length < positions * 2 * Bins)
            throw new ArgumentException("Spectrum is shorter than the frame count.", nameof(spectrum));

        double[] window = HannWindow;
        int length = (positions - 1) * Hop + NFft;
        var audio = new double[length];
        var envelope = new double[length];
        var full = new Complex[NFft];
        var frame = new double[NFft];

        for (int p = 0; p < positions; p++)
        {
            int offset = p * 2 * Bins;
            for (int k = 0; k < Bins; k++)
            {
                float logMagnitude = spectrum[offset + k];
                float angle = spectrum[offset + Bins + k];
                double magnitude = Math.Exp(logMagnitude);
                full[k] = new Complex(magnitude * Math.Cos(angle), magnitude * Math.Sin(angle));
            }
            // DC and Nyquist are real for a real-valued signal; irfft discards their angle.
            full[0] = new Complex(full[0].Real, 0);
            full[Bins - 1] = new Complex(full[Bins - 1].Real, 0);
            for (int k = 1; k < Bins - 1; k++)
                full[NFft - k] = Complex.Conjugate(full[k]);

            Fft.Inverse(full, frame);
            int start = p * Hop;
            for (int i = 0; i < NFft; i++)
            {
                audio[start + i] += frame[i] * window[i];
                envelope[start + i] += window[i] * window[i];
            }
        }

        int trim = (NFft - Hop) / 2;
        int outputLength = length - 2 * trim;
        if (outputLength <= 0) return [];
        var output = new float[outputLength];
        for (int i = 0; i < outputLength; i++)
        {
            double env = envelope[trim + i];
            output[i] = env > 1e-12 ? (float)(audio[trim + i] / env) : 0f;
        }
        return output;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (CodecBlock block in _blocks) block.Dispose();
    }

    private sealed class CodecBlock : IDisposable
    {
        private readonly bool _isConv;
        private readonly float[] _ln1, _ln2, _attnPost, _ffnPost;
        private readonly float[] _pw1, _pw2, _conv;
        private readonly float[] _qW, _kW, _vW, _oW, _qNorm, _kNorm, _gate;
        private readonly float[] _gateW, _upW, _downW;

        public CodecBlock(GgufTensorReader reader, int layer)
        {
            string prefix = $"a.gen.wav.tfm.blk.{layer}.";
            _ln1 = reader.ReadFloat32(prefix + "ln1.weight");
            _ln2 = reader.ReadFloat32(prefix + "ln2.weight");
            _attnPost = reader.ReadFloat32(prefix + "attn_post_norm.weight");
            _ffnPost = reader.ReadFloat32(prefix + "ffn_post_norm.weight");
            _isConv = reader.TryGetInfo(prefix + "conv_dw.weight", out _);
            if (_isConv)
            {
                _conv = reader.ReadFloat32(prefix + "conv_dw.weight");
                _pw1 = reader.ReadFloat32(prefix + "conv_pw1.weight");
                _pw2 = reader.ReadFloat32(prefix + "conv_pw2.weight");
                _qW = _kW = _vW = _oW = _qNorm = _kNorm = _gate = [];
            }
            else
            {
                _conv = _pw1 = _pw2 = [];
                _qW = reader.ReadFloat32(prefix + "attn_q.weight");
                _kW = reader.ReadFloat32(prefix + "attn_k.weight");
                _vW = reader.ReadFloat32(prefix + "attn_v.weight");
                _oW = reader.ReadFloat32(prefix + "attn_out.weight");
                _qNorm = reader.ReadFloat32(prefix + "attn_q_norm.weight");
                _kNorm = reader.ReadFloat32(prefix + "attn_k_norm.weight");
                _gate = reader.ReadFloat32(prefix + "attn_gate.weight");
            }
            _gateW = reader.ReadFloat32(prefix + "ffn_gate.weight");
            _upW = reader.ReadFloat32(prefix + "ffn_up.weight");
            _downW = reader.ReadFloat32(prefix + "ffn_down.weight");
        }

        public float[] Forward(float[] input, int positions)
        {
            var norm = new float[input.Length];
            Lfm3AudioCodeGenerator.RmsNorm(input, _ln1, norm, positions, Hidden, Eps);

            float[] mixed = _isConv ? Conv(norm, positions) : Attention(norm, positions);
            Lfm3AudioCodeGenerator.RmsNorm(mixed, _attnPost, positions, Hidden, Eps);
            var residual = new float[input.Length];
            for (int i = 0; i < residual.Length; i++) residual[i] = input[i] + mixed[i];

            var norm2 = new float[residual.Length];
            Lfm3AudioCodeGenerator.RmsNorm(residual, _ln2, norm2, positions, Hidden, Eps);
            float[] ffn = Ffn(norm2, positions);
            Lfm3AudioCodeGenerator.RmsNorm(ffn, _ffnPost, positions, Hidden, Eps);
            for (int i = 0; i < residual.Length; i++) residual[i] += ffn[i];
            return residual;
        }

        private float[] Conv(float[] x, int positions)
        {
            const int taps = 7;
            var bcx = new float[positions * 3 * Hidden];
            for (int p = 0; p < positions; p++)
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(p * Hidden, Hidden), _pw1, Hidden, 3 * Hidden,
                    bcx.AsSpan(p * 3 * Hidden, 3 * Hidden));

            var bx = new float[positions * Hidden];
            for (int p = 0; p < positions; p++)
                for (int h = 0; h < Hidden; h++)
                    bx[p * Hidden + h] = bcx[p * 3 * Hidden + h] * bcx[p * 3 * Hidden + 2 * Hidden + h];

            var conv = new float[positions * Hidden];
            for (int p = 0; p < positions; p++)
            {
                for (int h = 0; h < Hidden; h++)
                {
                    float sum = 0;
                    for (int k = 0; k < taps; k++)
                    {
                        int source = p - (taps - 1 - k);
                        if (source < 0) continue;
                        sum += bx[source * Hidden + h] * _conv[k + taps * h];
                    }
                    conv[p * Hidden + h] = sum * bcx[p * 3 * Hidden + Hidden + h];
                }
            }

            var output = new float[positions * Hidden];
            for (int p = 0; p < positions; p++)
                Lfm3AudioCodeGenerator.MatMul(conv.AsSpan(p * Hidden, Hidden), _pw2, Hidden, Hidden,
                    output.AsSpan(p * Hidden, Hidden));
            return output;
        }

        private float[] Attention(float[] x, int positions)
        {
            var q = new float[positions * Hidden];
            var k = new float[positions * KvHeads * HeadDim];
            var v = new float[positions * KvHeads * HeadDim];
            var gate = new float[positions * Heads];
            for (int p = 0; p < positions; p++)
            {
                int offset = p * Hidden;
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _qW, Hidden, Hidden, q.AsSpan(offset, Hidden));
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _kW, Hidden, KvHeads * HeadDim,
                    k.AsSpan(p * KvHeads * HeadDim, KvHeads * HeadDim));
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _vW, Hidden, KvHeads * HeadDim,
                    v.AsSpan(p * KvHeads * HeadDim, KvHeads * HeadDim));
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _gate, Hidden, Heads,
                    gate.AsSpan(p * Heads, Heads));

                for (int h = 0; h < Heads; h++)
                {
                    var row = q.AsSpan(offset + h * HeadDim, HeadDim);
                    Lfm3AudioCodeGenerator.RmsNorm(row, _qNorm, row, Eps);
                    ApplyRope(row, p);
                }
                for (int h = 0; h < KvHeads; h++)
                {
                    var row = k.AsSpan(p * KvHeads * HeadDim + h * HeadDim, HeadDim);
                    Lfm3AudioCodeGenerator.RmsNorm(row, _kNorm, row, Eps);
                    ApplyRope(row, p);
                }
            }

            var output = new float[positions * Hidden];
            var scores = new float[Window];
            int group = Heads / KvHeads;
            for (int p = 0; p < positions; p++)
            {
                int first = Math.Max(0, p - Window + 1);
                for (int head = 0; head < Heads; head++)
                {
                    int kvHead = head / group;
                    int qOffset = p * Hidden + head * HeadDim;
                    int count = p - first + 1;
                    for (int key = first; key <= p; key++)
                    {
                        int kOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[qOffset + d] * k[kOffset + d];
                        scores[key - first] = dot / MathF.Sqrt(HeadDim);
                    }
                    Softmax(scores.AsSpan(0, count));
                    float headGate = Sigmoid(gate[p * Heads + head]);
                    var row = output.AsSpan(qOffset, HeadDim);
                    for (int key = first; key <= p; key++)
                    {
                        int vOffset = key * KvHeads * HeadDim + kvHead * HeadDim;
                        float weight = scores[key - first] * headGate;
                        for (int d = 0; d < HeadDim; d++) row[d] += weight * v[vOffset + d];
                    }
                }
            }

            var result = new float[positions * Hidden];
            for (int p = 0; p < positions; p++)
                Lfm3AudioCodeGenerator.MatMul(output.AsSpan(p * Hidden, Hidden), _oW, Hidden, Hidden,
                    result.AsSpan(p * Hidden, Hidden));
            return result;
        }

        private float[] Ffn(float[] x, int positions)
        {
            var gate = new float[positions * FfnDim];
            var up = new float[positions * FfnDim];
            for (int p = 0; p < positions; p++)
            {
                int offset = p * Hidden;
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _gateW, Hidden, FfnDim,
                    gate.AsSpan(p * FfnDim, FfnDim));
                Lfm3AudioCodeGenerator.MatMul(x.AsSpan(offset, Hidden), _upW, Hidden, FfnDim,
                    up.AsSpan(p * FfnDim, FfnDim));
            }
            for (int i = 0; i < gate.Length; i++) gate[i] = Silu(gate[i]) * up[i];
            var output = new float[positions * Hidden];
            for (int p = 0; p < positions; p++)
                Lfm3AudioCodeGenerator.MatMul(gate.AsSpan(p * FfnDim, FfnDim), _downW, FfnDim, Hidden,
                    output.AsSpan(p * Hidden, Hidden));
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
