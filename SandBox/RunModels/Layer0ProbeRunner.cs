using LLama;
using LLama.Abstractions;
using LLama.Common;
using LLama.Native;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SandBox.RunModels
{
    /// <summary>
    /// Decisive layer-0 probe on the 3-token prefix:
    ///  (A) RoPE spec-conformance: SharpMind's post-RoPE Q/K vs an INDEPENDENT
    ///      reference RoPE (theta=500000, half-pairing) computed from the spec,
    ///      at positions 0,1,2.
    ///  (B) llama.cpp block_count metadata override: empirical test whether it
    ///      actually truncates the model (embeddings must differ from full model).
    ///  (C) Full naive layer-0 reference (spec RoPE + scalar attention + FFN from
    ///      SharpMind's own weights) vs SharpMind's block-0 forward, per position.
    /// </summary>
    public static class Layer0ProbeRunner
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

        public static async Task RunAsync(string modelName)
        {
            string modelFile = Path.Combine(ModelPath, $"{modelName}.gguf");
            if (!File.Exists(modelFile)) { Console.Error.WriteLine($"Model '{modelName}' not found"); return; }

            Console.WriteLine($"########## {modelName} — layer-0 probe (RoPE spec + override + naive reference) ##########");

            int[] prompt = { 128000, 128006, 9125 };
            Console.WriteLine($"Prompt: {string.Join(" ", prompt)}");

            var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
            metaHelper.Load(modelFile, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);

            var sharpConfig = modelConfig.ForModel() with { UseHooks = false };
            var mapping = new MappingBuilder(sharpConfig.ResolvedHardware)
                .ApplyPreset(sharpConfig)
                .ApplyQuantPreset(sharpConfig)
                .Build();
            GC.Collect(); GC.WaitForPendingFinalizers();
            var sw = Stopwatch.StartNew();
            var qOps = QuantizationFactory.Create(mapping);
            using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelFile);
            weights.InitializeWeights();
            Console.WriteLine($"SharpMind load: {sw.Elapsed.TotalSeconds:F1}s");

            // optimizeMemory:false keeps the dequantized float weights (otherwise
            // FreeFloatWeight replaces them with (InFeatures,1) placeholders).
            using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping, optimizeMemory: false);
            var config = weights.Config;
            int numH = config.NumHeads, numKv = config.NumKvHeads, headDim = config.HeadDim;
            int qDim = numH * headDim, kvDim = numKv * headDim;
            int hiddenDim = config.HiddenDim;
            float theta = config.RopeTheta;
            int ropeDim = (config.RopeDim ?? 0) > 0 ? config.RopeDim!.Value : headDim;
            Console.WriteLine($"Config: heads={numH} kv={numKv} headDim={headDim} hidden={hiddenDim} theta={theta} ropeDim={ropeDim}");

            var caches = new IKVCache[config.NumLayers];
            for (int i = 0; i < caches.Length; i++)
                caches[i] = new KVCacherBuilder().CreateKVCache(1, numKv, config.MaxSeqLen, headDim);

            using var input = new Tensor<int>(1, prompt.Length);
            prompt.CopyTo(input.Data);
            using var emb = model.ForwardEmbedding(input);

            var blk = model.GetBlock(0)!;
            using var normed = blk.Norm1.Forward(emb, null);
            using var q = blk.Attention.Wq.Forward(normed, null);
            using var k = blk.Attention.Wk.Forward(normed, null);
            using var v = blk.Attention.Wv.Forward(normed, null);

            // ---- (A) RoPE spec-conformance ----
            Console.WriteLine($"\n=== (A) RoPE spec-conformance (SharpMind vs independent reference) ===");
            RopeSpecCheck(blk, q, k, numH, numKv, headDim, ropeDim, theta, prompt.Length);
            Console.WriteLine("RoPE check done");

            // ---- (B) llama.block_count override empirical test ----
            // Proven dead by OverrideTestRunner (all K produce identical embeddings).
            Console.WriteLine($"\n=== (B) llama.cpp block_count override empirical test ===");
            Console.WriteLine("SKIPPED: OverrideTestRunner proved block_count override is ignored (all K identical).");

            // ---- (C) full naive layer-0 reference vs SharpMind block-0 ----
            Console.WriteLine($"\n=== (C) naive layer-0 reference vs SharpMind block-0 forward ===");
            NaiveLayer0Check(model, blk, emb, caches[0], config, numH, numKv, headDim, ropeDim, theta, prompt.Length);

            foreach (var c in caches) c.Dispose();
        }

        // ---------------- (A) RoPE spec-conformance ----------------

        private static void RopeSpecCheck(TransformerBlock blk, Tensor<float> q, Tensor<float> k,
            int numH, int numKv, int headDim, int ropeDim, float theta, int seqLen)
        {
            int half = ropeDim / 2;
            float[] freqs = new float[half];
            for (int i = 0; i < half; i++)
                freqs[i] = 1f / MathF.Pow(theta, 2f * i / ropeDim);

            // Reference-rotate every (pos, head) vector using spec formulas.
            using var refQ = new Tensor<float>(1, seqLen, numH * headDim);
            using var refK = new Tensor<float>(1, seqLen, numKv * headDim);
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numH; h++)
                {
                    var dst = refQ.Data.Slice((s * numH + h) * headDim, headDim);
                    var src = q.Data.Slice((s * numH + h) * headDim, headDim);
                    ReferenceRope(src, dst, headDim, half, freqs, s);
                }
                for (int h = 0; h < numKv; h++)
                {
                    var dst = refK.Data.Slice((s * numKv + h) * headDim, headDim);
                    var src = k.Data.Slice((s * numKv + h) * headDim, headDim);
                    ReferenceRope(src, dst, headDim, half, freqs, s);
                }
            }

            // SharpMind's own RoPE (in-place on views of q,k).
            using var qr = q.Reshape(1, seqLen, numH, headDim);
            using var kr = k.Reshape(1, seqLen, numKv, headDim);
            blk.Attention.PositionalEncoder.ApplyBatched(qr, 0);
            blk.Attention.PositionalEncoder.ApplyBatched(kr, 0);

            Console.WriteLine("pos  Q maxAbsDiff  Q cos   K maxAbsDiff  K cos");
            for (int s = 0; s < seqLen; s++)
            {
                double qMax = 0, kMax = 0, qCosMin = 1, kCosMin = 1;
                for (int h = 0; h < numH; h++)
                {
                    var a = q.Data.Slice((s * numH + h) * headDim, headDim);
                    var b = refQ.Data.Slice((s * numH + h) * headDim, headDim);
                    qMax = Math.Max(qMax, MaxAbs(a, b));
                    qCosMin = Math.Min(qCosMin, Cos(a, b));
                }
                for (int h = 0; h < numKv; h++)
                {
                    var a = k.Data.Slice((s * numKv + h) * headDim, headDim);
                    var b = refK.Data.Slice((s * numKv + h) * headDim, headDim);
                    kMax = Math.Max(kMax, MaxAbs(a, b));
                    kCosMin = Math.Min(kCosMin, Cos(a, b));
                }
                Console.WriteLine($"{s,3}  {qMax,10:E2}  {qCosMin,6:F4}  {kMax,10:E2}  {kCosMin,6:F4}");
            }
        }

        private static void ReferenceRope(ReadOnlySpan<float> src, Span<float> dst, int headDim, int half, float[] freqs, int pos)
        {
            src.CopyTo(dst);
            for (int i = 0; i < half; i++)
            {
                float ang = pos * freqs[i];
                float cos = MathF.Cos(ang), sin = MathF.Sin(ang);
                float x0 = src[i], x1 = src[i + half];
                dst[i] = x0 * cos - x1 * sin;
                dst[i + half] = x1 * cos + x0 * sin;
            }
        }

        // ---------------- (B) override empirical test ----------------

        private static void OverrideCheck(string modelFile, int[] promptIds)
        {
            var embFull = LlamaEmbeddings(modelFile, promptIds, null);
            var embOne = LlamaEmbeddings(modelFile, promptIds, new List<MetadataOverride> { new("llama.block_count", 1) });
            int n = embFull.Length;
            double cos01 = Cos(embFull, embOne);
            double rmsDiff = 0;
            for (int d = 0; d < n; d++)
                rmsDiff += (embFull[d] - embOne[d]) * (embFull[d] - embOne[d]);
            rmsDiff = Math.Sqrt(rmsDiff / n);
            Console.WriteLine($"pos-2 embedding: full vs block_count=1  cos={cos01:F6}  rmsDiff={rmsDiff:E3}");
            Console.WriteLine(cos01 > 0.9999
                ? "OVERRIDE FAILS: block_count=1 produced identical output to full model -> override ignored."
                : "OVERRIDE WORKS: block_count=1 truncated the model -> per-layer oracle is valid.");
        }

        private static float[] LlamaEmbeddings(string modelFile, int[] promptIds, List<MetadataOverride>? overrides)
        {
            var modelParams = new ModelParams(modelFile)
            {
                ContextSize = 4096,
                GpuLayerCount = 0,
                Embeddings = true,
                PoolingType = LLamaPoolingType.None,
                MetadataOverrides = overrides
            };
            using var weights = LLamaWeights.LoadFromFile(modelParams);
            using var ctx = weights.CreateContext(modelParams);
            var batch = new LLamaBatch();
            for (int i = 0; i < promptIds.Length; i++)
                batch.Add(promptIds[i], i, (LLamaSeqId)0, true);
            var result = ctx.NativeHandle.Decode(batch);
            if (result != DecodeResult.Ok) throw new InvalidOperationException($"llama_decode failed: {result}");
            var emb = ctx.NativeHandle.GetEmbeddingsIth((LLamaPos)2);
            return emb.ToArray();
        }

        // ---------------- (C) naive layer-0 reference ----------------

        private static void NaiveLayer0Check(Transformer model, TransformerBlock blk, Tensor<float> emb,
            IKVCache cache, ModelConfig config, int numH, int numKv, int headDim, int ropeDim, float theta, int seqLen)
        {
            int hiddenDim = config.HiddenDim;
            int qDim = numH * headDim, kvDim = numKv * headDim;
            int ffnDim = config.FfnDim;
            int half = ropeDim / 2;
            float scale = 1f / MathF.Sqrt(headDim);
            float[] freqs = new float[half];
            for (int i = 0; i < half; i++)
                freqs[i] = 1f / MathF.Pow(theta, 2f * i / ropeDim);

            // SharpMind's block-0 output (blk.Forward mutates its input in place -> use a copy).
            var smEmb = new Tensor<float>(emb.Shape);
            emb.Data.CopyTo(smEmb.Data);
            using var smOut = blk.Forward(smEmb, cache, 0, true, null);

            // ---- naive reference ----
            using var normed = blk.Norm1.Forward(emb, null);                 // reuse validated norm
            ReadOnlySpan<float> Wq = blk.Attention.Wq.Weight.Data; float[]? bq = BiasOrNull(blk.Attention.Wq);
            ReadOnlySpan<float> Wk = blk.Attention.Wk.Weight.Data; float[]? bk = BiasOrNull(blk.Attention.Wk);
            ReadOnlySpan<float> Wv = blk.Attention.Wv.Weight.Data; float[]? bv = BiasOrNull(blk.Attention.Wv);
            ReadOnlySpan<float> Wo = blk.Attention.Wo.Weight.Data; float[]? bo = BiasOrNull(blk.Attention.Wo);

            var gatedLayer = blk.Ffn.WGated;
            ReadOnlySpan<float> gateUp = gatedLayer is null ? ReadOnlySpan<float>.Empty : gatedLayer.Weight.Data;
            float[]? bg = gatedLayer is null || gatedLayer.Bias is null ? null : gatedLayer.Bias.Data.ToArray();
            var downLayer = blk.Ffn.WDown;
            ReadOnlySpan<float> wDown = downLayer is null ? ReadOnlySpan<float>.Empty : downLayer.Weight.Data;
            float[]? bDown = downLayer is null || downLayer.Bias is null ? null : downLayer.Bias.Data.ToArray();

            Console.WriteLine($"dims: hiddenDim={hiddenDim} qDim={qDim} kvDim={kvDim} ffnDim={ffnDim} ropeDim={ropeDim}");
            Console.WriteLine($"Wq.Data={blk.Attention.Wq.Weight.Data.Length} ({blk.Attention.Wq.Weight.Shape})  Wk.Data={blk.Attention.Wk.Weight.Data.Length} ({blk.Attention.Wk.Weight.Shape})  Wv.Data={blk.Attention.Wv.Weight.Data.Length} ({blk.Attention.Wv.Weight.Shape})  Wo.Data={blk.Attention.Wo.Weight.Data.Length} ({blk.Attention.Wo.Weight.Shape})");
            Console.WriteLine($"gateUp.Data={gateUp.Length} ({gatedLayer?.Weight.Shape})  wDown.Data={wDown.Length} ({downLayer?.Weight.Shape})");
            Console.WriteLine($"normed.Data={normed.Data.Length}  emb.Data={emb.Data.Length}  smOut.Data={smOut.Data.Length}");

            float[] q = new float[seqLen * qDim], k = new float[seqLen * kvDim], v = new float[seqLen * kvDim];
            for (int s = 0; s < seqLen; s++)
            {
                var row = normed.Data.Slice(s * hiddenDim, hiddenDim);
                MatMulAddBias(row, Wq, bq, hiddenDim, qDim, q, s * qDim);
                MatMulAddBias(row, Wk, bk, hiddenDim, kvDim, k, s * kvDim);
                MatMulAddBias(row, Wv, bv, hiddenDim, kvDim, v, s * kvDim);
            }
            // Reference RoPE on q and k.
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numH; h++)
                    ReferenceRope(q.AsSpan(s * qDim + h * headDim, headDim), q.AsSpan(s * qDim + h * headDim, headDim), headDim, half, freqs, s);
                for (int h = 0; h < numKv; h++)
                    ReferenceRope(k.AsSpan(s * kvDim + h * headDim, headDim), k.AsSpan(s * kvDim + h * headDim, headDim), headDim, half, freqs, s);
            }

            // Scalar attention (GQA).
            float[] attn = new float[seqLen * qDim];
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numH; h++)
                {
                    int kvHead = h / (numH / numKv);
                    var qRow = q.AsSpan(s * qDim + h * headDim, headDim);
                    int effLen = s + 1;
                    float[] scores = new float[effLen];
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < effLen; j++)
                    {
                        var kRow = k.AsSpan(j * kvDim + kvHead * headDim, headDim);
                        float dot = 0;
                        for (int d = 0; d < headDim; d++) dot += qRow[d] * kRow[d];
                        scores[j] = dot * scale;
                        if (scores[j] > max) max = scores[j];
                    }
                    float lSum = 0;
                    for (int j = 0; j < effLen; j++) { scores[j] = MathF.Exp(scores[j] - max); lSum += scores[j]; }
                    var dst = attn.AsSpan(s * qDim + h * headDim, headDim);
                    dst.Clear();
                    for (int j = 0; j < effLen; j++)
                    {
                        float sm = scores[j] / lSum;
                        var vRow = v.AsSpan(j * kvDim + kvHead * headDim, headDim);
                        for (int d = 0; d < headDim; d++) dst[d] += sm * vRow[d];
                    }
                }
            }

            // Wo + residual + norm2 + FFN + residual.
            using var h1 = new Tensor<float>(1, seqLen, hiddenDim);
            for (int s = 0; s < seqLen; s++)
            {
                float[] proj = new float[hiddenDim];
                MatMulAddBias(attn.AsSpan(s * qDim, qDim), Wo, bo, qDim, hiddenDim, proj, 0);
                var src = emb.Data.Slice(s * hiddenDim, hiddenDim);
                var dst = h1.Data.Slice(s * hiddenDim, hiddenDim);
                for (int d = 0; d < hiddenDim; d++) dst[d] = src[d] + proj[d];
            }
            using var normed2 = blk.Norm2.Forward(h1, null);

            using var h2 = new Tensor<float>(1, seqLen, hiddenDim);
            for (int s = 0; s < seqLen; s++)
            {
                var row = normed2.Data.Slice(s * hiddenDim, hiddenDim);
                float[] gu = new float[2 * ffnDim];
                MatMulAddBias(row, gateUp, bg, hiddenDim, 2 * ffnDim, gu, 0);
                float[] mid = new float[ffnDim];
                for (int d = 0; d < ffnDim; d++)
                {
                    float gate = Silu(gu[d]);
                    mid[d] = gate * gu[ffnDim + d];
                }
                float[] down = new float[hiddenDim];
                MatMulAddBias(mid, wDown, bDown, ffnDim, hiddenDim, down, 0);
                var src = h1.Data.Slice(s * hiddenDim, hiddenDim);
                var dst = h2.Data.Slice(s * hiddenDim, hiddenDim);
                for (int d = 0; d < hiddenDim; d++) dst[d] = src[d] + down[d];
            }

            Console.WriteLine("pos  SM-out RMS   naive RMS   cos     maxAbsDiff");
            for (int s = 0; s < seqLen; s++)
            {
                var a = smOut.Data.Slice(s * hiddenDim, hiddenDim);
                var b = h2.Data.Slice(s * hiddenDim, hiddenDim);
                Console.WriteLine($"{s,3}  {Rms(a),10:F4}  {Rms(b),9:F4}  {Cos(a, b),6:F4}  {MaxAbs(a, b):E2}");
            }
        }

        private static float[]? BiasOrNull(LinearLayer layer)
            => layer.Bias is null ? null : layer.Bias.Data.ToArray();

        private static void MatMulAddBias(ReadOnlySpan<float> inp, ReadOnlySpan<float> w, float[]? bias, int inDim, int outDim, float[] outp, int offset)
        {
            // Float weights are stored (in, out) row-major: w[i * outDim + o].
            for (int o = 0; o < outDim; o++)
            {
                double acc = 0;
                for (int i = 0; i < inDim; i++) acc += inp[i] * w[i * outDim + o];
                if (bias != null) acc += bias[o];
                outp[offset + o] = (float)acc;
            }
        }

        private static float Silu(float x) => x / (1f + MathF.Exp(-x));

        // ---------------- helpers ----------------

        private static double MaxAbs(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            double m = 0;
            for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
            return m;
        }

        private static double Cos(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }
            return dot / Math.Sqrt(na * nb);
        }

        private static double Rms(ReadOnlySpan<float> v)
        {
            double sum = 0;
            foreach (var f in v) sum += (double)f * f;
            return Math.Sqrt(sum / v.Length);
        }
    }
}
