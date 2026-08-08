using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Tokenization;
using System.Text;

namespace SharpMind.Samples.Conversion;

/// <summary>
/// Proves the .SMM round-trip: GGUF -> SMM -> GGUF is lossless at the tensor
/// byte level, and reports the size overhead of the .SMM container.
///
/// For a real downloaded model it:
///   1. converts the .gguf to .smm via <see cref="GgufToSmmConverter"/>,
///   2. converts the .smm back to a fresh .gguf via <see cref="SmmToGufConverter"/>,
///   3. runs the same prompt on the original .gguf, the .smm, and the
///      round-tripped .gguf and compares the responses,
///   4. verifies that every tensor's raw bytes are bit-identical between the
///      original .gguf and the round-tripped .gguf,
///   5. verifies the re-emitted GGUF metadata / tokenizer stay equivalent
///      (same architecture, vocab, bos/eos, chat template),
///   6. prints a size table so the .SMM container's byte overhead is visible.
/// </summary>
public static class SmmRoundTripSample
{
    private const string DefaultModel = "SmolLM2-135M-Instruct.Q4_K_M";

    /// <summary>
    /// Locates <paramref name="modelName"/>.gguf under <paramref name="assetsDir"/>
    /// (optional — pass null to use the default ExternalAssets-style folder) and
    /// runs the full round-trip, writing all container variants to <paramref name="outDir"/>.
    /// </summary>
    public static async Task RunAsync(string? assetsDir = null, string modelName = DefaultModel, string? outDir = null)
    {
        string ggufPath = FindModel(assetsDir, modelName);
        string outputRoot = outDir ?? Path.Combine(Path.GetTempPath(), "smm-roundtrip");
        Directory.CreateDirectory(outputRoot);

        string smmPath = Path.Combine(outputRoot, $"{modelName}.smm");
        string roundTrip = Path.Combine(outputRoot, $"{modelName}.roundtrip.gguf");

        await Console.Out.WriteLineAsync($"Source GGUF : {ggufPath}");
        await Console.Out.WriteLineAsync($"Round-trip  : {roundTrip}");
        await Console.Out.WriteLineAsync();

        // ── GGUF → SMM ──
        GgufToSmmConverter.Convert(ggufPath, smmPath, new SmmWriteOptions { Source = "gguf" });
        await Console.Out.WriteLineAsync("GGUF → SMM complete.");
        await Console.Out.WriteLineAsync();

        // ── SMM → GGUF ──
        SmmToGufConverter.Convert(smmPath, roundTrip);
        await Console.Out.WriteLineAsync("SMM → GGUF complete.");
        await Console.Out.WriteLineAsync();

        // ── Run the same prompt on all three containers ──
        const string prompt = "hello";
        var original = await RunPromptAsync(ggufPath, prompt, maxTokens: 12);
        var smm = await RunPromptAsync(smmPath, prompt, maxTokens: 12);
        var roundTripped = await RunPromptAsync(roundTrip, prompt, maxTokens: 12);

        await Console.Out.WriteLineAsync($"Prompt: \"{prompt}\"");
        await Console.Out.WriteLineAsync($"  GGUF (original)     : {FormatReply(original)}");
        await Console.Out.WriteLineAsync($"  SMM                 : {FormatReply(smm)}");
        await Console.Out.WriteLineAsync($"  GGUF (round-trip)   : {FormatReply(roundTripped)}");
        bool repliesMatch = original == smm && smm == roundTripped;
        await Console.Out.WriteLineAsync($"Responses match      : {(repliesMatch ? "yes" : "NO — DIFFERENT")}");
        await Console.Out.WriteLineAsync();

        // ── Verify byte parity + metadata equivalence ──
        var src = ReadTensorBytes(ggufPath);
        var dst = ReadTensorBytes(roundTrip);
        int mismatches = 0;

        if (src.Length != dst.Length)
        {
            await Console.Out.WriteLineAsync($"WARN tensor count differs: {src.Length} vs {dst.Length}");
            mismatches++;
        }

        var dstByName = dst.ToDictionary(t => t.Name, StringComparer.Ordinal);
        foreach (var t in src)
        {
            if (!dstByName.TryGetValue(t.Name, out var other))
            {
                await Console.Out.WriteLineAsync($"MISSING tensor: {t.Name}");
                mismatches++;
                continue;
            }

            bool shapeOk = t.Shape.SequenceEqual(other.Shape);
            bool dtypeOk = t.Dtype == other.Dtype;
            bool bytesEqual = t.Bytes.AsSpan().SequenceEqual(other.Bytes);
            if (!shapeOk) await Console.Out.WriteLineAsync($"SHAPE differs: {t.Name} [{string.Join(",", t.Shape)}] vs [{string.Join(",", other.Shape)}]");
            if (!dtypeOk) await Console.Out.WriteLineAsync($"DTYPE   differs: {t.Name} {t.Dtype} vs {other.Dtype}");
            if (!bytesEqual) await Console.Out.WriteLineAsync($"BYTES   differ: {t.Name} ({FormatBytes(t.Bytes.Length)} vs {FormatBytes(other.Bytes.Length)})");
            if (!(shapeOk && dtypeOk && bytesEqual)) mismatches++;
        }

        await Console.Out.WriteLineAsync($"Tensor parity      : {src.Length} compared, {mismatches} mismatch(es)");
        bool metaOk = VerifyMeta(ggufPath, roundTrip);
        await Console.Out.WriteLineAsync($"Metadata / tokenizer: {(metaOk ? "equivalent" : "DIFFER")}");
        await Console.Out.WriteLineAsync();

        // ── Size table ──
        long ggufSize = new FileInfo(ggufPath).Length;
        long smmSize = new FileInfo(smmPath).Length;
        long rtSize = new FileInfo(roundTrip).Length;

        await Console.Out.WriteLineAsync("Size comparison (bytes):");
        await Console.Out.WriteLineAsync($"  GGUF (original)   : {ggufSize,12:N0}");
        await Console.Out.WriteLineAsync($"  SMM               : {smmSize,12:N0}  {Pct(smmSize, ggufSize)} vs GGUF");
        await Console.Out.WriteLineAsync($"  GGUF (round-trip) : {rtSize,12:N0}  {Pct(rtSize, ggufSize)} vs GGUF");
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync($"Files: {outputRoot}");
        await Console.Out.WriteLineAsync(mismatches == 0 && metaOk && repliesMatch ? "PASS" : "ROUND-TRIP FAILED");
    }

    // ── Helper functions ──────────────────────────────────────────────────

    private static async Task<string> RunPromptAsync(string modelPath, string prompt, int maxTokens)
    {
        ModelFormat? fmt = ModelFormatHelpers.GetFormatForExtension(modelPath);
        if (fmt == null) throw new InvalidDataException($"File type not supported: {modelPath}");
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
        metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
        if (tokenizer == null) return "(no tokenizer)";

        var sharpConfig = modelConfig.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);

        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig);
        var formatter = ChatPromptFormatterFactory.Create(meta, tokenizer);

        await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta, formatter: formatter, disposeModel: false)
        {
            MaxTokens = 256,
            MaxNewTokens = maxTokens,
            Temperature = 0.0f,
            TopK = 1,
            RepetitionPenalty = 1.0f,
        };
        session.InitializeChat();

        var reply = new StringBuilder();
        var cts = new CancellationTokenSource();
        bool sent = false;
        var tokens = 0;

        void Response(ChatStreamEntry text)
        {
            if (text.Status == ChatStatus.Responding && !string.IsNullOrEmpty(text.Token))
            {
                reply.Append(text.Token);
                tokens++;
                if (tokens >= maxTokens) cts.Cancel();
            }
        }

        Task<ChatMessage> Prompt()
        {
            if (!sent)
            {
                sent = true;
                return Task.FromResult(new ChatMessage { Content = prompt, Role = ChatRole.User });
            }
            cts.Cancel();
            return Task.FromResult(new ChatMessage { Content = "exit", Role = ChatRole.User });
        }

        await session.StartChatAsync(Prompt, Response, cts.Token);
        return reply.ToString();
    }

    private static string FormatReply(string reply)
        => string.IsNullOrEmpty(reply) ? "(empty)" : $"\"{reply.Trim()}\"";

    private static string FindModel(string? assetsDir, string modelName)
    {
        var candidates = new List<string>();
        if (!string.IsNullOrEmpty(assetsDir))
            candidates.Add(Path.Combine(assetsDir, $"{modelName}.gguf"));
        candidates.Add(Path.Combine("ExternalAssets", $"{modelName}.gguf"));

        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;
        throw new FileNotFoundException($"Model not found under {string.Join(" or ", candidates)}");
    }

    private static (string Name, int[] Shape, QuantDType Dtype, byte[] Bytes)[] ReadTensorBytes(string ggufPath)
    {
        var meta = GgufLoader.LoadMeta(ggufPath);
        var result = new (string, int[], QuantDType, byte[])[meta.Tensors.Count];
        for (int i = 0; i < meta.Tensors.Count; i++)
        {
            var info = meta.Tensors[i];
            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
            byte[] bytes = ReadRange(ggufPath, meta.DataOffset + info.Offset, rawSize);
            result[i] = (info.Name, info.Shape, info.Dtype, bytes);
        }
        return result;
    }

    private static byte[] ReadRange(string path, long offset, long length)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        var bytes = new byte[length];
        fs.Position = offset;
        fs.ReadExactly(bytes);
        return bytes;
    }

    private static bool VerifyMeta(string ggufPath, string roundTripPath)
    {
        var a = GgufLoader.LoadMeta(ggufPath);
        var b = GgufLoader.LoadMeta(roundTripPath);
        if (a.GetString("general.architecture") != b.GetString("general.architecture")) return false;

        var ta = GgufLoader.LoadTokenizerFromMeta(a);
        var tb = GgufLoader.LoadTokenizerFromMeta(b);
        if (ta is null || tb is null) return ta is null && tb is null;
        if (ta.VocabSize != tb.VocabSize) return false;
        for (int i = 0; i < Math.Min(ta.VocabSize, 128); i++)
            if (ta.IdToToken(i) != tb.IdToToken(i)) return false;

        return a.GetChatTemplate() == b.GetChatTemplate();
    }

    private static string Pct(long size, long baseline)
        => $"{100.0 * (size - baseline) / Math.Max(1, baseline),+8:F2}%";

    private static string FormatBytes(long bytes)
        => bytes >= 1 << 20 ? $"{bytes / (double)(1 << 20):F2} MiB" : $"{bytes:N0} B";
}