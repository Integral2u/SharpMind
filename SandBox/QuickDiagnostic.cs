using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System;
using System.IO;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace SandBox;

public static class QuickDiagnostic
{
    private static global::SharpMind.SharpMindConfig DeriveSharpMindConfig(ModelConfig mc, HardwareTier hw)
    {
        var attn = mc.NumKvHeads switch { 1 => AttentionKind.MQA, _ when mc.NumKvHeads == mc.NumHeads => AttentionKind.MHA, _ => AttentionKind.GQA };
        return new global::SharpMind.SharpMindConfig { Activation = ActivationKind.SiLU, Gate = GateKind.SwiGLU, Ffn = FfnKind.Gated, Attention = attn, Norm = NormKind.RMSNorm, Arch = ArchKind.Decoder, Hardware = hw };
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported) return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }

    private static float MaxAbs(Span<float> data) { float m = 0f; for (int i = 0; i < data.Length; i++) { float a = Math.Abs(data[i]); if (a > m) m = a; } return m; }

    public static void RunQwenDiagnostic()
    {
        string[] models = ["qwen2-0_5b-instruct-q8_0", "qwen2.5-1.5b-instruct-q8_0", "tinyllama-1.1b-chat-v1.0.f16"];

        foreach (var modelName in models)
        {
            string path = Path.Combine(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets", $"{modelName}.gguf");
            if (!File.Exists(path)) { Console.WriteLine($"[SKIP] {modelName} not found"); continue; }

            Console.WriteLine($"\n═══ {modelName} ═══");

            GgufLoader.Load(path, null, out ModelMetaData meta, out ModelConfig mc, out Tokenizer? tokenizer);
            if (tokenizer == null) { Console.WriteLine("No tokenizer"); continue; }

            var hw = DetectBestHardware();
            var cfg = DeriveSharpMindConfig(mc, hw);
            // Print GGUF tensor names
            var metaDiag = GgufLoader.LoadMeta(path);
            foreach (var t in metaDiag.Tensors)
                Console.WriteLine($"  Tensor: {t.Name} shape=[{string.Join(",", t.Shape)}] dtype={t.Dtype}");
            Console.WriteLine($"  Config: HiddenDim={mc.HiddenDim} FfnDim={mc.FfnDim} NumLayers={mc.NumLayers} VocabSize={mc.VocabSize}");

            using var weights = GgufLoader.LoadWeightsToTransformerWeights(path, mc);
            var model = ModelFactory.CreateSession(weights, cfg, null, null, false);

            // LM head diagnostic
            var lmHead = model.LmHead;
            int hd = mc.HiddenDim;
            Console.WriteLine($"LM head: {(lmHead != null ? $"non-null [{lmHead.Shape[0]},{lmHead.Shape[1]}] mean={string.Format("{0:G6}", lmHead.Data.ToArray().Average())}" : "null")}");
            Console.WriteLine($"Embedding shape=[{model.EmbeddingWeight.Shape[0]},{model.EmbeddingWeight.Shape[1]}] mean={string.Format("{0:G6}", model.EmbeddingWeight.Data.ToArray().Average())}");
            // Check embedding data layout: token 0's first 5 values
            var embData = model.EmbeddingWeight.Data;
            Console.WriteLine($"  Token 0 emb[0..4]: {embData[0]:F6}, {embData[1]:F6}, {embData[2]:F6}, {embData[3]:F6}, {embData[4]:F6}");
            Console.WriteLine($"  Token 0 emb[{hd - 5}..{hd - 1}]: {embData[hd - 5]:F6}, {embData[hd - 4]:F6}, {embData[hd - 3]:F6}, {embData[hd - 2]:F6}, {embData[hd - 1]:F6}");
            Console.WriteLine($"  Embedding at [1,0..4] (token 1): {embData[1*hd]:F6}, {embData[1*hd + 1]:F6}, {embData[1*hd + 2]:F6}, {embData[1*hd + 3]:F6}, {embData[1*hd + 4]:F6}");

            // Build prompt via formatter
            var formatter = ChatPromptFormatterFactory.Create(meta.GetChatTemplate());
            bool addBos = meta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            string prompt = formatter.Format(
                [new() { Role = ChatRole.User, Content = "hello" }], tokenizer, addBos);
            Console.WriteLine($"Prompt: {prompt.Replace("\n", "\\n")}");
            Console.WriteLine($"Formatter: {formatter.GetType().Name}");

            // Tokenize
            var tokenIds = tokenizer.Encode(prompt, addBos: false);
            Console.WriteLine($"Tokens ({tokenIds.Length}): {string.Join(", ", tokenIds.Take(48))}");
            bool hasUnk = tokenIds.Any(t => t == 0);
            Console.WriteLine($"UNK tokens: {(hasUnk ? "YES ✗" : "none ✓")}");

            // Multi-step generation diagnostic
            int hiddenDim = mc.HiddenDim;
            using var input = Tensor<int>.From(tokenIds, 1, tokenIds.Length);
            var caches = new KVCache[mc.NumLayers];
            for (int i = 0; i < mc.NumLayers; i++)
                caches[i] = new KVCache(1, mc.NumKvHeads, 256, mc.HeadDim);

            int promptLen = tokenIds.Length;
            var logits = model.ForwardLastLogits(input, caches, 0);
            int vocabSize = logits.Shape[1];

            Console.Write("Gen: ");
            int[] genScratch = new int[1];
            for (int step = 0; step < 8; step++)
            {
                int topId = 0; float topVal = float.NegativeInfinity;
                for (int j = 0; j < vocabSize; j++) { if (logits.Data[j] > topVal) { topVal = logits.Data[j]; topId = j; } }
                string tokenStr = tokenizer.Decode([topId], skipSpecials: false);
                Console.Write($"[{topId}:{tokenStr.Trim()}] ");

                if (step > 0)
                {
                    genScratch[0] = topId;
                    logits.Dispose();
                    int newPos = 0 + promptLen + step;
                    using var nextInput = Tensor<int>.From(genScratch, 1, 1);
                    logits = model.ForwardLastLogits(nextInput, caches, newPos);
                }
                else
                {
                    // First step: advance to next
                    genScratch[0] = topId;
                    logits.Dispose();
                    int newPos = 0 + promptLen + 0;
                    using var nextInput = Tensor<int>.From(genScratch, 1, 1);
                    logits = model.ForwardLastLogits(nextInput, caches, newPos);
                }

                int nanCnt = 0; for (int j = 0; j < logits.Data.Length; j++) if (float.IsNaN(logits.Data[j])) nanCnt++;
                if (nanCnt > 0) { Console.Write($"(NaN!={nanCnt}) "); break; }
            }
            Console.WriteLine();

            logits.Dispose();
            foreach (var c in caches) c.Dispose();
            model.Dispose();
        }
    }

    public static void CheckEmbeddingLayout(string modelName)
    {
        string path = Path.Combine(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets", $"{modelName}.gguf");
        if (!File.Exists(path)) { Console.WriteLine($"[SKIP] {modelName} not found"); return; }

        Console.WriteLine($"\n═══ {modelName} Embedding Layout Diagnostic ═══");

        var meta = GgufLoader.LoadMeta(path);
        var info = meta.Tensors.First(t => t.Name == "token_embd.weight");
        Console.WriteLine($"GGUF token_embd.weight shape=[{string.Join(",", info.Shape)}] dtype={info.Dtype}");

        // Read raw Q8_0 data and dequantize first 32 elements
        using var mmf = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(path, System.IO.FileMode.Open);
        using var stream = mmf.CreateViewStream(0, 0, System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);
        stream.Position = meta.DataOffset + info.Offset;
        float[] first32 = new float[32];
        GgufLoader.ReadQ8_0(reader, first32, 32);
        Console.WriteLine($"  First 32 dequantized values: {string.Format("{0:F6} {1:F6} {2:F6} {3:F6}...", first32[0], first32[1], first32[2], first32[3])}");

        // Read next block (elements 32..63)
        float[] next32 = new float[32];
        GgufLoader.ReadQ8_0(reader, next32, 32);
        Console.WriteLine($"  Next 32 (32..63): {string.Format("{0:F6} {1:F6} {2:F6} {3:F6}...", next32[0], next32[1], next32[2], next32[3])}");

        // Seek to position 896 (dim0=0, dim1=896) — should be vocab 896's first hidden value if [hidden,vocab]
        stream.Position = meta.DataOffset + info.Offset + (896 / 32) * 34; // Q8_0 block size = 34, skip first 28 blocks
        float[] at896 = new float[32];
        GgufLoader.ReadQ8_0(reader, at896, 32);
        Console.WriteLine($"  At offset 896 (vocab 896?): {string.Format("{0:F6} {1:F6} {2:F6} {3:F6}...", at896[0], at896[1], at896[2], at896[3])}");

        // Seek to element 151936 (dim0=1, dim1=0) — if [hidden,vocab], this is hidden=1, vocab=0
        long elemIdx = 151936;
        long blockIdx = elemIdx / 32;
        long byteOffset = blockIdx * 34;
        stream.Position = meta.DataOffset + info.Offset + byteOffset;
        float[] at151936 = new float[32];
        GgufLoader.ReadQ8_0(reader, at151936, 32);
        int subIdx = (int)(elemIdx % 32);
        Console.WriteLine($"  At offset 151936 ({subIdx}): {at151936[subIdx]:F6}");

        Console.WriteLine("  ---");
        Console.WriteLine("  If [vocab,hidden] row-major:");
        Console.WriteLine($"    emb[0] = token 0, hidden 0 = {first32[0]:F6}");
        Console.WriteLine($"    emb[1] = token 0, hidden 1 = {first32[1]:F6}");
        Console.WriteLine($"    emb[896] = token 1, hidden 0 = {at896[0]:F6}");

        Console.WriteLine("  If [hidden,vocab] row-major:");
        Console.WriteLine($"    emb[0] = hidden 0, vocab 0 = {first32[0]:F6}");
        Console.WriteLine($"    emb[1] = hidden 0, vocab 1 = {first32[1]:F6}");
        Console.WriteLine($"    emb[151936] = hidden 1, vocab 0 = {at151936[subIdx]:F6}");
    }

    public static void Run()
    {
        string modelPath = Path.Combine(
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets",
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");

        Console.WriteLine("═══ Quick Q6_K Dequant Diagnostic ═══");

        // Load just output.weight directly
        var meta = GgufLoader.LoadMeta(modelPath);
        var weights = GgufLoader.LoadWeights(modelPath);
        var tensorMap = meta.Tensors.ToDictionary(t => t.Name);

        if (!weights.TryGetValue("output.weight", out var outputWeight))
        {
            Console.WriteLine("output.weight not found!");
            return;
        }

        Console.WriteLine($"output.weight shape: [{outputWeight.Shape[0]}, {outputWeight.Shape[1]}]");
        var data = outputWeight.Data;

        // Token 71486 ("Alright")
        int checkToken = 71486;
        int hiddenDim = 1536;

        var row = data.Slice(checkToken * hiddenDim, hiddenDim);
        float min = float.MaxValue, max = float.MinValue;
        double mean = 0;
        for (int i = 0; i < row.Length; i++)
        {
            float v = row[i];
            if (v < min) min = v;
            if (v > max) max = v;
            mean += v;
        }
        mean /= row.Length;

        Console.WriteLine($"\nToken {checkToken} LM head weight stats:");
        Console.WriteLine($"  min={min:G4} max={max:G4} mean={mean:G6}");
        Console.Write("  First 10: ");
        for (int k = 0; k < 10; k++)
            Console.Write($"{row[k]:F10} ");
        Console.WriteLine();

        // Expected values from Python gguf.dequantize
        float[] expected = new float[] {
            0.022630691528320312f, -0.009429454803466797f, -0.0603485107421875f, -0.024516582489013672f,
            0.024516582489013672f, 0.02640247344970703f,  0.011315345764160156f, -0.0075435638427734375f,
            0.015087127685546875f, 0.0018858909606933594f
        };

        Console.WriteLine("\nComparison with Python gguf.dequantize:");
        int matches = 0;
        for (int i = 0; i < 10; i++)
        {
            bool match = Math.Abs(row[i] - expected[i]) < 0.0001f;
            if (match) matches++;
            Console.WriteLine($"  [{i}] SharpMind={row[i]:F10}  Python={expected[i]:F10}  {(match ? "PASS" : "FAIL")}");
        }
        Console.WriteLine($"\n{matches}/10 match with Python gguf.dequantize");
        Console.WriteLine(matches == 10 ? "✓ Q6_K FIX IS CORRECT!" : "✗ Q6_K FIX STILL WRONG!");
    }
}
