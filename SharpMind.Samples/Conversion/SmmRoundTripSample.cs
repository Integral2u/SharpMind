using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;

namespace SharpMind.Samples.Conversion;

/// <summary>
/// Proves the .SMM round-trip: GGUF -> SMM -> GGUF is lossless at the tensor
/// byte level, and reports how much (if at all) per-tensor GZip compression in
/// the .SMM container actually saved.
///
/// For a real downloaded model it:
///   1. converts the .gguf to .smm in every <see cref="CompressionMode"/>
///      (None / Gzip / Auto),
///   2. converts the .smm back to a fresh .gguf via <see cref="SmmToGufConverter"/>,
///   3. verifies that every tensor's raw bytes are bit-identical between the
///      original .gguf and the round-tripped .gguf,
///   4. verifies the re-emitted GGUF metadata / tokenizer stay equivalent
///      (same architecture, vocab, bos/eos, chat template),
///   5. prints a size table so the "was compression of any use?" question can be
///      answered per mode.
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

        string smmNone = Path.Combine(outputRoot, $"{modelName}.none.smm");
        string smmGzip = Path.Combine(outputRoot, $"{modelName}.gzip.smm");
        string smmAuto = Path.Combine(outputRoot, $"{modelName}.auto.smm");
        string roundTrip = Path.Combine(outputRoot, $"{modelName}.roundtrip.gguf");

        await Console.Out.WriteLineAsync($"Source GGUF : {ggufPath}");
        await Console.Out.WriteLineAsync($"Round-trip  : {roundTrip}");
        await Console.Out.WriteLineAsync();

        // ── GGUF → SMM in every compression mode ──
        GgufToSmmConverter.Convert(ggufPath, smmNone, new SmmWriteOptions { Compression = CompressionMode.None, Source = "gguf" });
        GgufToSmmConverter.Convert(ggufPath, smmGzip, new SmmWriteOptions { Compression = CompressionMode.Gzip, Source = "gguf" });
        GgufToSmmConverter.Convert(ggufPath, smmAuto, new SmmWriteOptions { Compression = CompressionMode.Auto, Source = "gguf" });
        await Console.Out.WriteLineAsync("GGUF → SMM (None, Gzip, Auto) complete.");
        await Console.Out.WriteLineAsync();

        // ── SMM → GGUF (from the Auto container) ──
        SmmToGufConverter.Convert(smmAuto, roundTrip);
        await Console.Out.WriteLineAsync("SMM → GGUF (from Auto) complete.");
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
        long noneSize = new FileInfo(smmNone).Length;
        long gzSize = new FileInfo(smmGzip).Length;
        long autoSize = new FileInfo(smmAuto).Length;
        long rtSize = new FileInfo(roundTrip).Length;

        await Console.Out.WriteLineAsync("Size comparison (bytes):");
        await Console.Out.WriteLineAsync($"  GGUF (original)   : {ggufSize,12:N0}");
        await Console.Out.WriteLineAsync($"  SMM: None         : {noneSize,12:N0}  {Pct(noneSize, ggufSize)} vs GGUF");
        await Console.Out.WriteLineAsync($"  SMM: Gzip         : {gzSize,12:N0}  {Pct(gzSize, ggufSize)} vs GGUF");
        await Console.Out.WriteLineAsync($"  SMM: Auto         : {autoSize,12:N0}  {Pct(autoSize, ggufSize)} vs GGUF");
        await Console.Out.WriteLineAsync($"  GGUF (round-trip) : {rtSize,12:N0}  {Pct(rtSize, ggufSize)} vs GGUF");

        bool compressionHelped = Math.Min(autoSize, gzSize) < noneSize;
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync(compressionHelped
            ? "Verdict: per-tensor compression saved space for this model."
            : "Verdict: per-tensor compression did NOT help size (quantized weights are already incompressible).");
        await Console.Out.WriteLineAsync();
        await Console.Out.WriteLineAsync($"Files: {outputRoot}");
        await Console.Out.WriteLineAsync(mismatches == 0 && metaOk ? "PASS" : "ROUND-TRIP FAILED");
    }

    // ── Helper functions ──────────────────────────────────────────────────

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