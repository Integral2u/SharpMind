using System.Text;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace GgufSmokeProbe;

/// <summary>
/// An intentionally naive, self-contained Qwen2-MoE forward pass used as an
/// oracle for SharpMind. It shares no math with SharpMind.Model: it parses the
/// GGUF container itself, dequantizes only F32 and Q8_0 (both trivial and
/// unambiguous), and runs every matmul as a plain row-major loop.
///
/// Requantize the model to Q8_0 before use so K-quant bit layouts never come
/// into play. Tokenization is borrowed from SharpMind on purpose -- that is
/// shared, well-tested code and not what this oracle is here to check.
/// </summary>
internal static class Qwen2MoEReference
{
    private const int TypeF32 = 0;
    private const int TypeQ8_0 = 8;

    private sealed class Tensor
    {
        public string Name = "";
        public long[] Dims = [];
        public int Type;
        public long Offset;                    // relative to the data section
        public long NumelOverride = -1;
        public long Numel => NumelOverride >= 0 ? NumelOverride : Dims.Aggregate(1L, (a, b) => a * b);
        public long NumBytes => Type == TypeF32 ? Numel * 4 : (Numel / 32) * 34;
    }

    private sealed class Reader : IDisposable
    {
        private readonly FileStream _fs;
        private readonly BinaryReader _br;
        public readonly Dictionary<string, (int Type, object Val)> Kvs = new(StringComparer.Ordinal);
        public readonly List<Tensor> Tensors = [];
        public long DataOffset;

        public Reader(string path)
        {
            _fs = File.OpenRead(path);
            _br = new BinaryReader(_fs, Encoding.UTF8);
            if (_br.ReadUInt32() != 0x46554747u) throw new InvalidDataException("not a GGUF file");
            _ = _br.ReadUInt32();                       // version
            ulong nTensors = _br.ReadUInt64();
            ulong nKv = _br.ReadUInt64();

            bool trace = Environment.GetEnvironmentVariable("REF_TRACE") == "1";
            for (ulong i = 0; i < nKv; i++)
            {
                string k = Str();
                int t = (int)_br.ReadUInt32();
                Kvs[k] = (t, Value(t));
                if (trace) Console.Error.WriteLine($"  KV {i,2} {k} type={t}");
            }

            for (ulong i = 0; i < nTensors; i++)
            {
                var ts = new Tensor { Name = Str() };
                int nd = (int)_br.ReadUInt32();          // n_dims is uint32, not uint64
                ts.Dims = new long[nd];
                for (int j = 0; j < nd; j++) ts.Dims[j] = (long)_br.ReadUInt64();
                ts.Type = (int)_br.ReadUInt32();
                ts.Offset = (long)_br.ReadUInt64();
                if (trace)
                    Console.Error.WriteLine($"  TS {i,3} {ts.Name} nd={nd} type={ts.Type} off={ts.Offset}");
                Tensors.Add(ts);
            }

            long align = Int("general.alignment", 32);
            DataOffset = ((_fs.Position + align - 1) / align) * align;
            if (trace) Console.Error.WriteLine($"  dataOffset={DataOffset} align={align}");
        }

        private string Str()
        {
            ulong n = _br.ReadUInt64();
            return Encoding.UTF8.GetString(_br.ReadBytes((int)n));
        }

        private object Value(int t)
        {
            switch (t)
            {
                case 0: return _br.ReadByte();
                case 1: return _br.ReadSByte();
                case 2: return _br.ReadUInt16();
                case 3: return _br.ReadInt16();
                case 4: return _br.ReadUInt32();
                case 5: return _br.ReadInt32();
                case 6: return _br.ReadSingle();
                case 7: return _br.ReadByte();
                case 8: return Str();
                case 9: return ArrayValue((int)_br.ReadUInt32());
                case 10: return _br.ReadUInt64();
                case 11: return _br.ReadInt64();
                case 12: return _br.ReadDouble();
                default: throw new InvalidDataException($"kv type {t}");
            }
        }

        private object ArrayValue(int elemType)
        {
            ulong n = _br.ReadUInt64();
            object last = null!;
            for (ulong i = 0; i < n; i++) last = Value(elemType);
            return last;
        }

        public int Int(string key, int dflt = 0) =>
            Kvs.TryGetValue(key, out var v) ? (int)ToInt64(v.Val) : dflt;

        public float Float(string key, float dflt = 0) =>
            Kvs.TryGetValue(key, out var v) ? ToFloat(v.Val) : dflt;

        public static long ToInt64(object o) => o switch
        {
            byte b => b,
            sbyte sb => sb,
            ushort us => us,
            short s => s,
            uint u => u,
            int i => i,
            ulong ul => (long)ul,
            long l => l,
            bool bo => bo ? 1 : 0,
            _ => 0
        };

        private static float ToFloat(object o) => o switch
        {
            float f => f,
            double d => (float)d,
            _ => ToInt64(o)
        };

        public byte[] Bytes(long abs, int len)
        {
            var buf = new byte[len];
            _fs.Position = abs;
            int done = 0;
            while (done < len)
            {
                int r = _fs.Read(buf, done, len - done);
                if (r <= 0) throw new EndOfStreamException("short read");
                done += r;
            }
            return buf;
        }

        public void Dispose() { _br.Dispose(); _fs.Dispose(); }
    }

    private static float[] Dequant(Reader r, Tensor t)
    {
        byte[] raw = r.Bytes(r.DataOffset + t.Offset, (int)t.NumBytes);
        long n = t.Numel;
        var dst = new float[n];
        if (t.Type == TypeF32)
        {
            Buffer.BlockCopy(raw, 0, dst, 0, (int)(n * 4));
            return dst;
        }
        if (t.Type != TypeQ8_0) throw new NotSupportedException($"dtype {t.Type} on {t.Name}; requantize to Q8_0");
        for (long b = 0; b < n / 32; b++)
        {
            int bo = (int)(b * 34);
            float d = (float)BitConverter.ToHalf(new ReadOnlySpan<byte>(raw, bo, 2));
            long wo = b * 32;
            for (int i = 0; i < 32; i++) dst[wo + i] = d * (sbyte)raw[bo + 2 + i];
        }
        return dst;
    }

    /// <summary>Views one expert plane of a [in, out, nExpert] tensor as [in, out].</summary>
    private static Tensor Plane(Tensor t, int expert, int nExpert)
    {
        long plane = t.Numel / nExpert;
        long planeBytes = t.Type == TypeF32 ? plane * 4 : (plane / 32) * 34;
        return new Tensor
        {
            Name = $"{t.Name}#{expert}",
            Dims = [t.Dims[0], t.Dims[1]],
            Type = t.Type,
            Offset = t.Offset + expert * planeBytes,
            NumelOverride = plane,
        };
    }

    /// <summary>Copies src[srcOff .. srcOff+len) into dst[dstOff .. dstOff+len).</summary>
    private static void CopyRow(float[] dst, int dstOff, int len, float[] src, int srcOff)
    {
        Array.Copy(src, srcOff, dst, dstOff, len);
    }

    /// <summary>
    /// Y[t, o] = dot(X[t, 0..inF), w[o, 0..inF]). GGUF/ggml tensors declare ne0 = inF as the
    /// contiguous dimension, so output row o starts at w[o * inF] and is inF elements long.
    /// </summary>
    private static void MatVec(float[] X, int nTok, float[] w, int inF, int outF, float[] Y)
    {
        for (int t = 0; t < nTok; t++)
        {
            int xOff = t * inF;
            int yOff = t * outF;
            for (int o = 0; o < outF; o++)
            {
                int ro = o * inF;
                float sum = 0f;
                for (int i = 0; i < inF; i++) sum += X[xOff + i] * w[ro + i];
                Y[yOff + o] = sum;
            }
        }
    }

    private static void AddRowBias(float[] Y, int nTok, int outF, float[] bias)
    {
        for (int t = 0; t < nTok; t++)
            for (int o = 0; o < outF; o++) Y[t * outF + o] += bias[o];
    }

    private static void RmsNormRows(float[] X, int nTok, int hidden, float[] w, float eps, float[] Y)
    {
        for (int t = 0; t < nTok; t++)
        {
            double s = 0;
            int off = t * hidden;
            for (int i = 0; i < hidden; i++) s += (double)X[off + i] * X[off + i];
            float inv = 1f / MathF.Sqrt((float)(s / hidden) + eps);
            for (int i = 0; i < hidden; i++) Y[off + i] = X[off + i] * inv * w[i];
        }
    }

    private static void SoftmaxRange(float[] y, int from, int count)
    {
        float m = float.NegativeInfinity;
        for (int i = 0; i < count; i++) if (y[from + i] > m) m = y[from + i];
        double s = 0;
        for (int i = 0; i < count; i++) { y[from + i] = MathF.Exp(y[from + i] - m); s += y[from + i]; }
        float inv = (float)(1.0 / s);
        for (int i = 0; i < count; i++) y[from + i] *= inv;
    }

    /// <summary>NeoX rotate-half: pairs (i, i + dim/2) for i &lt; dim/2.</summary>
    private static void Rope(float[] V, int nTok, int heads, int headDim, float thetaBase)
    {
        int half = headDim / 2;
        var freqs = new float[half];
        for (int i = 0; i < half; i++) freqs[i] = 1f / MathF.Pow(thetaBase, 2f * i / headDim);
        for (int t = 0; t < nTok; t++)
        for (int h = 0; h < heads; h++)
        {
            int off = t * heads * headDim + h * headDim;
            var p = new float[headDim];
            CopyRow(p, 0, headDim, V, off);
            for (int i = 0; i < half; i++)
            {
                float a = t * freqs[i];
                float c = MathF.Cos(a), s = MathF.Sin(a);
                float x1 = p[i], x2 = p[i + half];
                p[i] = x1 * c - x2 * s;
                p[i + half] = x1 * s + x2 * c;
            }
            CopyRow(V, off, headDim, p, 0);
        }
    }

    private static double AvgAbs(float[] a, int from, int count)
    {
        double s = 0;
        for (int i = 0; i < count; i++) s += Math.Abs(a[from + i]);
        return count == 0 ? 0 : s / count;
    }

    private static double MaxOf(float[] a)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i]));
        return m;
    }

    private static void SwiGlu(float[] x, float[] wg, float[] wu, float[] wd,
                               int hidden, int ff, float[] y, float[] tmp)
    {
        MatVec(x, 1, wg, hidden, ff, tmp);
        float[] up = new float[ff];
        MatVec(x, 1, wu, hidden, ff, up);
        for (int i = 0; i < ff; i++) tmp[i] *= 1f / (1f + MathF.Exp(-tmp[i])) * up[i];
        MatVec(tmp, 1, wd, ff, hidden, y);
    }

    public static int Run(string path, string[] args)
    {
        string promptArg = args.Length > 2 ? args[2] : "";
        if (promptArg.StartsWith("@")) promptArg = File.ReadAllText(promptArg[1..]);

        using var r = new Reader(path);
        var T = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var t in r.Tensors) T[t.Name] = t;

        string arch = r.Kvs.TryGetValue("general.architecture", out var gkv) && gkv.Val is string gstr ? gstr : "";
        int hidden = r.Int($"{arch}.embedding_length");
        int nHead = r.Int($"{arch}.attention.head_count");
        int nKv = r.Int($"{arch}.attention.head_count_kv", nHead);
        int headDim = hidden / nHead;
        int nLayer = r.Int($"{arch}.block_count");
        int nExpert = r.Int($"{arch}.expert_count");
        int topK = r.Int($"{arch}.expert_used_count", 1);
        float eps = r.Float($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        float theta = r.Float($"{arch}.rope.freq_base", 10000f);
        bool normProb = r.Int($"{arch}.expert_weights_norm") != 0;
        int kvGroup = Math.Max(1, nHead / nKv);
        int qDim = nHead * headDim, kvDim = nKv * headDim;

        Console.WriteLine($"REF arch={arch} hidden={hidden} headDim={headDim} heads={nHead} kv={nKv} " +
                          $"kvGroup={kvGroup} layers={nLayer} experts={nExpert} topK={topK} " +
                          $"eps={eps:G3} theta={theta:G6} normProb={normProb}");

        // Tokenization comes from SharpMind (shared, trusted code).
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out _, out _, out Tokenizer? tok);
        if (tok is null) { Console.WriteLine("REF ERROR: no tokenizer in file"); return 1; }

        int[] ids = [785];
        if (promptArg.Length > 0) ids = tok.Encode(promptArg, addBos: false, addEos: false);
        int nTok = ids.Length;
        Console.WriteLine($"REF prompt tokens={nTok}: {string.Join(",", ids)}");

        Tensor emb = T["token_embd.weight"];
        int nVocab = (int)emb.Dims[1];
        var X = new float[nTok * hidden];
        long embRowBytes = emb.Type == TypeF32 ? hidden * 4L : ((hidden / 32) * 34L);
        for (int t = 0; t < nTok; t++)
        {
            var row = new Tensor
            {
                Name = $"emb[{ids[t]}]",
                Dims = [hidden],
                Type = emb.Type,
                Offset = emb.Offset + ids[t] * embRowBytes,
                NumelOverride = hidden,
            };
            CopyRow(X, t * hidden, hidden, Dequant(r, row), 0);
        }

        var H1 = new float[nTok * hidden];
        var H2 = new float[nTok * hidden];
        var Q = new float[nTok * qDim];
        var K = new float[nTok * kvDim];
        var V = new float[nTok * kvDim];
        var Proj = new float[nTok * qDim];
        var Scratch = new float[nTok * hidden];
        var scores = new float[nTok];

        int maxBlk = int.TryParse(Environment.GetEnvironmentVariable("REF_BLOCKS"), out int mb) ? mb : nLayer;
        bool isMoE = T.ContainsKey("blk.0.ffn_gate_exps.weight");
        Console.WriteLine($"REF isMoE={isMoE} (expert_count kv={nExpert})");
        for (int L = 0; L < Math.Min(nLayer, maxBlk); L++)
        {
            string B = $"blk.{L}.";
            bool hasBias = T.ContainsKey(B + "attn_q.bias");

            // ---- attention ----
            RmsNormRows(X, nTok, hidden, Dequant(r, T[B + "attn_norm.weight"]), eps, H1);
            MatVec(H1, nTok, Dequant(r, T[B + "attn_q.weight"]), hidden, qDim, Q);
            if (hasBias) AddRowBias(Q, nTok, qDim, Dequant(r, T[B + "attn_q.bias"]));
            MatVec(H1, nTok, Dequant(r, T[B + "attn_k.weight"]), hidden, kvDim, K);
            if (hasBias) AddRowBias(K, nTok, kvDim, Dequant(r, T[B + "attn_k.bias"]));
            MatVec(H1, nTok, Dequant(r, T[B + "attn_v.weight"]), hidden, kvDim, V);
            if (hasBias) AddRowBias(V, nTok, kvDim, Dequant(r, T[B + "attn_v.bias"]));
            Rope(Q, nTok, nHead, headDim, theta);
            Rope(K, nTok, nKv, headDim, theta);

            for (int h = 0; h < nHead; h++)
            {
                int kvh = h / kvGroup;
                for (int t = 0; t < nTok; t++)
                {
                    int n = t + 1;                        // causal
                    for (int i = 0; i < n; i++)
                    {
                        float d = 0;
                        for (int j = 0; j < headDim; j++)
                            d += Q[t * qDim + h * headDim + j] * K[i * kvDim + kvh * headDim + j];
                        scores[i] = d / MathF.Sqrt(headDim);
                    }
                    SoftmaxRange(scores, 0, n);
                    for (int j = 0; j < headDim; j++) Proj[t * qDim + h * headDim + j] = 0f;
                    for (int i = 0; i < n; i++)
                    {
                        float p = scores[i];
                        if (p == 0f) continue;
                        for (int j = 0; j < headDim; j++)
                            Proj[t * qDim + h * headDim + j] += p * V[i * kvDim + kvh * headDim + j];
                    }
                }
            }
            MatVec(Proj, nTok, Dequant(r, T[B + "attn_output.weight"]), qDim, hidden, Scratch);
            for (int i = 0; i < nTok * hidden; i++) X[i] += Scratch[i];
            double attnMag = AvgAbs(Scratch, 0, Scratch.Length);
            double xMagBefore = AvgAbs(X, 0, X.Length);

            // ---- FFN ----
            RmsNormRows(X, nTok, hidden, Dequant(r, T[B + "ffn_norm.weight"]), eps, H2);

            if (!isMoE)
            {
                int ffnDim = (int)T[B + "ffn_gate.weight"].Dims[1];
                float[] denseOut = new float[hidden];
                for (int t = 0; t < nTok; t++)
                {
                    var h2t = new float[hidden];
                    CopyRow(h2t, 0, hidden, H2, t * hidden);
                    SwiGlu(h2t, Dequant(r, T[B + "ffn_gate.weight"]),
                               Dequant(r, T[B + "ffn_up.weight"]),
                               Dequant(r, T[B + "ffn_down.weight"]),
                               hidden, ffnDim, denseOut, new float[ffnDim]);
                    for (int i = 0; i < hidden; i++) X[t * hidden + i] += denseOut[i];
                }
                Console.WriteLine($"  REF blk{L} |x|avg={xMagBefore:G6} |attn|avg={attnMag:G6} " +
                                  $"ffnDelta={Math.Abs(AvgAbs(X, (nTok - 1) * hidden, hidden) - xMagBefore):G6} " +
                                  $"|x|avgNow={AvgAbs(X, (nTok - 1) * hidden, hidden):G6}");
                GC.Collect();
                continue;
            }

            var router = new float[nTok * nExpert];
            MatVec(H2, nTok, Dequant(r, T[B + "ffn_gate_inp.weight"]), hidden, nExpert, router);
            var probs = (float[])router.Clone();
            for (int t = 0; t < nTok; t++) SoftmaxRange(probs, t * nExpert, nExpert);

            if (L == 0)
            {
                var wT = Dequant(r, T[B + "ffn_gate_inp.weight"]);
                string Fmt(float[] p)
                {
                    var ord = Enumerable.Range(0, nExpert).OrderByDescending(i => p[i]).Take(topK);
                    return string.Join(" ", ord.Select(i => $"{i}:{p[i]:F5}"));
                }
                float[] RouteFrom(float[] xin, bool transposed)
                {
                    var p = new float[nExpert];
                    for (int j = 0; j < nExpert; j++)
                    {
                        float s = 0f;
                        for (int i = 0; i < hidden; i++)
                            s += xin[i] * (transposed ? wT[i * nExpert + j] : wT[j * hidden + i]);
                        p[j] = s;
                    }
                    SoftmaxRange(p, 0, nExpert);
                    return p;
                }
                var residual = new float[hidden];
                CopyRow(residual, 0, hidden, X, (nTok - 1) * hidden);
                var normed = new float[hidden];
                CopyRow(normed, 0, hidden, H2, (nTok - 1) * hidden);
                double lmax = 0;
                for (int j = 0; j < nExpert; j++) lmax = Math.Max(lmax, Math.Abs(router[j]));
                Console.Error.WriteLine($"[router] |X|avg={AvgAbs(residual, 0, hidden):F5} " +
                    $"|H2|avg={AvgAbs(normed, 0, hidden):F5} logitMaxAbs={lmax:F4}");
                Console.Error.WriteLine($"[router] file w n={wT.Length} |w|avg={AvgAbs(wT, 0, wT.Length):G6} " +
                    $"max={MaxOf(wT):G6}");
                Console.Error.WriteLine($"[router] ggml+H2    {Fmt(RouteFrom(normed, false))}");
            }

            Tensor gEx = T[B + "ffn_gate_exps.weight"];
            Tensor uEx = T[B + "ffn_up_exps.weight"];
            Tensor dEx = T[B + "ffn_down_exps.weight"];
            int expFfn = (int)gEx.Dims[1];
            bool hasShared = T.ContainsKey(B + "ffn_gate_shexp.weight");

            var tmp = new float[expFfn];
            var upv = new float[expFfn];
            var expOut = new float[hidden];
            var routed = new float[hidden];

            for (int t = 0; t < nTok; t++)
            {
                var pr = new float[nExpert];
                CopyRow(pr, 0, nExpert, probs, t * nExpert);
                var top = Enumerable.Range(0, nExpert).OrderByDescending(i => pr[i]).Take(topK).ToArray();
                float wsum = top.Sum(e => pr[e]);
                if (normProb && wsum > 0f) for (int i = 0; i < top.Length; i++) pr[top[i]] /= wsum;

                var h2t = new float[hidden];
                CopyRow(h2t, 0, hidden, H2, t * hidden);
                Array.Clear(routed);

                for (int ei = 0; ei < top.Length; ei++)
                {
                    int e = top[ei];
                    float[] wg = Dequant(r, Plane(gEx, e, nExpert));
                    float[] wu = Dequant(r, Plane(uEx, e, nExpert));
                    float[] wd = Dequant(r, Plane(dEx, e, nExpert));
                    MatVec(h2t, 1, wg, hidden, expFfn, tmp);
                    MatVec(h2t, 1, wu, hidden, expFfn, upv);
                    for (int i = 0; i < expFfn; i++) tmp[i] *= 1f / (1f + MathF.Exp(-tmp[i])) * upv[i];
                    MatVec(tmp, 1, wd, expFfn, hidden, expOut);
                    float w = pr[e];
                    for (int i = 0; i < hidden; i++) routed[i] += expOut[i] * w;
                }

                float gateScalar = 1f;
                if (hasShared)
                {
                    int shFfn = (int)T[B + "ffn_gate_shexp.weight"].Dims[1];
                    var shared = new float[hidden];
                    SwiGlu(h2t, Dequant(r, T[B + "ffn_gate_shexp.weight"]),
                               Dequant(r, T[B + "ffn_up_shexp.weight"]),
                               Dequant(r, T[B + "ffn_down_shexp.weight"]),
                               hidden, shFfn, shared, new float[shFfn]);
                    var gv = new float[1];
                    MatVec(h2t, 1, Dequant(r, T[B + "ffn_gate_inp_shexp.weight"]), hidden, 1, gv);
                    gateScalar = 1f / (1f + MathF.Exp(-gv[0]));
                    for (int i = 0; i < hidden; i++) routed[i] += shared[i] * gateScalar;
                }

                for (int i = 0; i < hidden; i++) X[t * hidden + i] += routed[i];

                if (L == 0)
                    Console.WriteLine($"  REF blk0 tok{t} |routed|avg={AvgAbs(routed, 0, hidden):G6} " +
                                      $"gate={gateScalar:G6} top={string.Join("/", top)} " +
                                      $"w=[{string.Join(",", top.Select(e => pr[e].ToString("F4")))}]");
            }

            double xMagAfter = AvgAbs(X, (nTok - 1) * hidden, hidden);
            Console.WriteLine($"  REF blk{L} |x|avg={xMagBefore:G6} |attn|avg={attnMag:G6} " +
                              $"ffnDelta={Math.Abs(xMagAfter - xMagBefore):G6} |x|avgNow={xMagAfter:G6}");
            GC.Collect();
        }

        var lastRow = new float[hidden];
        CopyRow(lastRow, 0, hidden, X, (nTok - 1) * hidden);
        var fin = new float[hidden];
        RmsNormRows(lastRow, 1, hidden, Dequant(r, T["output_norm.weight"]), eps, fin);

        var logits = new float[nVocab];
        Tensor outW = T.TryGetValue("output.weight", out var ow) ? ow : emb;   // tied embeddings
        Console.WriteLine($"REF lm_head={outW.Name} tied={ReferenceEquals(outW, emb)}");

        if (Environment.GetEnvironmentVariable("REF_DUMP") is string dumpPath && dumpPath.Length > 0)
        {
            File.WriteAllLines(dumpPath + ".fin", fin.Select(v => v.ToString("R")));
            Console.WriteLine($"REF wrote {dumpPath}.fin ({fin.Length} floats)");
        }
        MatVec(fin, 1, Dequant(r, outW), hidden, nVocab, logits);

        double mean = 0; foreach (float v in logits) mean += v; mean /= nVocab;
        double var2 = 0; foreach (float v in logits) var2 += (v - mean) * (v - mean);
        var ord = Enumerable.Range(0, nVocab).OrderByDescending(i => logits[i]).ToArray();
        Console.WriteLine($"REF logits mean={mean:F4} std={Math.Sqrt(var2 / nVocab):F4} top10:");
        for (int i = 0; i < 10; i++)
        {
            int id = ord[i];
            Console.WriteLine($"    {i}: {id} '{tok.IdToToken(id)}' {logits[id]:F4}");
        }
        Console.WriteLine($"REF margin1={logits[ord[0]] - logits[ord[1]]:F4}");
        return 0;
    }
}
