using SharpMind.Core;
using SharpMind.Core.Quantization;
using System.Text;

namespace SharpMind.Model.Format;

/// <summary>
/// Re-quantizes an .SMM container: every tensor's data region is read,
/// dequantized back to floats and re-encoded to a target dtype via
/// <see cref="TensorQuantizer.Quantize"/>, then streamed into a fresh container,
/// while the meta / tokenizer / plugin regions are copied through verbatim. The
/// new file is written to a <c>.tmp</c> path and atomically moved into place, so
/// a failure (or cancelled operation) never trashes the model.
///
/// Sources that are already quantized are re-quantized whenever the target is
/// actually leaner for that tensor's shape — this is how a Q8_0 .SMM (e.g. one
/// produced from a Q8_0 GGUF) can be shrunk to Q4_K. Re-encoding is lossy
/// twice, so tensors whose target is not leaner (coarser → finer, or unchanged)
/// are copied verbatim instead of being degraded for nothing. When a target
/// dtype cannot encode a tensor's shape (K-quant needs a flattened length
/// divisible by 256), the tensor falls back to F16 — but only when F16 is still
/// leaner than the source; otherwise it stays as-is.
/// </summary>
public static class SmmQuantizer
{
    private static readonly QuantizationOps _readOps = QuantizationFactory.Create(HardwareTier.Scalar);

    /// <summary>
    /// Quantizes every tensor in <paramref name="path"/> (in place) according to
    /// <paramref name="options"/> (per-role manual dtypes or a byte budget; see
    /// <see cref="SmmQuantOptions"/>). <paramref name="progress"/> reports 0..1
    /// as tensors are processed; <paramref name="ct"/> cancels the rewrite — the
    /// partial temp file is deleted and <paramref name="path"/> stays untouched.
    /// </summary>
    public static void Quantize(
        string path,
        SmmQuantOptions options,        
        IProgress<float>? progress = null, CancellationToken ct = default)
        => Quantize(path, path, options, progress, ct);

    /// <summary>
    /// Quantizes every tensor in <paramref name="sourcePath"/> to a fresh file at
    /// <paramref name="destPath"/> (leaving the source untouched) according to
    /// <paramref name="options"/>. <paramref name="progress"/> reports 0..1 as
    /// tensors are processed; <paramref name="ct"/> cancels the rewrite.
    /// </summary>
    public static void Quantize(
        string sourcePath,
        string destPath,
        SmmQuantOptions options,        
        IProgress<float>? progress = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var entries = SmmLoader.ReadTensorIndex(sourcePath);
        long sourceLength = new FileInfo(sourcePath).Length;
        var plan = SmmQuantPlan.Resolve(entries, options, sourceLength);
        Quantize(sourcePath, destPath, name => plan.GetValueOrDefault(name, QuantDType.F16), progress, ct);
    }

    /// <summary>
    /// Quantizes every tensor in <paramref name="path"/> (in place) to a single
    /// target <paramref name="target"/>. <paramref name="progress"/> reports
    /// 0..1 as tensors are processed; <paramref name="ct"/> cancels the rewrite.
    /// </summary>
    public static void Quantize(
        string path,
        QuantDType target,       
        IProgress<float>? progress = null, CancellationToken ct = default)
        => Quantize(path, path, target, progress, ct);

    /// <summary>
    /// Quantizes every tensor in <paramref name="sourcePath"/> to a fresh file at
    /// <paramref name="destPath"/> (leaving the source untouched) to a single
    /// target <paramref name="target"/>. <paramref name="progress"/> reports 0..1
    /// as tensors are processed; <paramref name="ct"/> cancels the rewrite.
    /// </summary>
    public static void Quantize(
        string sourcePath,
        string destPath,
        QuantDType target,        
        IProgress<float>? progress = null, CancellationToken ct = default)
    {
        if (target == QuantDType.F32 || !TensorQuantizer.IsSupportedTarget(target))
            throw new NotSupportedException(
                $"Quantization level {target} is not supported. Use F16 or a K-quant (Q2_K..Q8_K).");

        Quantize(sourcePath, destPath, _ => target, progress, ct);
    }

    private static void Quantize(
        string sourcePath,
        string destPath,
        Func<string, QuantDType> targetFor,        
        IProgress<float>? progress = null, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destPath);
        if (!File.Exists(sourcePath)) throw new FileNotFoundException(sourcePath);

        string tmpPath = destPath + ".tmp";
        try
        {
            WriteInternal(sourcePath, tmpPath, targetFor, ct, progress);
            MoveWithRetry(tmpPath, destPath);
        }
        catch
        {
            TryDelete(tmpPath);
            throw;
        }
    }

    private static void WriteInternal(
        string sourcePath, string tmpPath, Func<string, QuantDType> targetFor,
        CancellationToken ct, IProgress<float>? progress)
    {
        using var src = File.OpenRead(sourcePath);
        using var srcReader = new BinaryReader(src);
        ReadHeader(srcReader, out long metaLen, out long tokenizerLen, out long pluginAsmCount, out long tensorCount, out _, out long dataOffset);

        byte[] metaBytes = srcReader.ReadBytes(checked((int)metaLen));
        byte[] tokenizerBytes = srcReader.ReadBytes(checked((int)tokenizerLen));

        // Everything between the tokenizer and the data region (plugin manifest
        // + its alignment padding) is copied verbatim.
        long pluginAndPaddingLen = dataOffset - src.Position;
        if (pluginAndPaddingLen < 0)
            throw new InvalidDataException("Not SMM: malformed data offset in " + sourcePath);
        byte[] pluginAndPadding = srcReader.ReadBytes(checked((int)pluginAndPaddingLen));

        var entries = SmmLoader.ReadTensorIndex(sourcePath);
        if (entries.Count != tensorCount)
            throw new InvalidDataException("Not SMM: tensor index mismatch in " + sourcePath);

        using var outFs = new FileStream(tmpPath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(outFs);

        long headerPos = outFs.Position;
        writer.Write(new byte[SmmConstants.HeaderSize]);
        writer.Write(metaBytes);
        writer.Write(tokenizerBytes);
        writer.Write(pluginAndPadding);

        // Align to the same block size the writer uses, then stream tensor data.
        long newDataOffset = Align(outFs.Position, SmmConstants.DefaultAlignment);
        if (newDataOffset > outFs.Position)
            writer.Write(new byte[newDataOffset - outFs.Position]);

        var newIndex = new List<SmmTensorIndexEntry>(entries.Count);
        long dataCursor = 0;
        for (int i = 0; i < entries.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report((float)i / entries.Count);

            var entry = entries[i];
            long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
            if (rawSize <= 0)
                throw new InvalidDataException($"Tensor '{entry.Name}' has an unsupported dtype: {entry.Dtype}.");

            src.Position = dataOffset + entry.Offset;
            byte[] raw = srcReader.ReadBytes(checked((int)rawSize));

            QuantDType target = targetFor(entry.Name);
            byte[] outBytes = raw;
            QuantDType outDtype = entry.Dtype;
            // F32 sources are always candidates (they can at least go to F16);
            // already-quantized sources are re-encoded only when the target is
            // actually leaner for this shape (never upscale / re-encode for nothing).
            if (entry.Dtype != target &&
                (entry.Dtype == QuantDType.F32 ||
                 QuantizationOps.GetRawTensorByteCount(entry.Shape, target) < rawSize))
                (outBytes, outDtype) = QuantizeTensor(raw, entry.Shape, entry.Dtype, target);

            writer.Write(outBytes);
            newIndex.Add(new SmmTensorIndexEntry(entry.Name, outDtype, entry.Shape, dataCursor));
            dataCursor += outBytes.Length;
        }

        // ── Tensor index (end of file) ──
        long indexStart = outFs.Position;
        foreach (var entry in newIndex)
        {
            WriteString(writer, entry.Name);
            writer.Write((int)entry.Dtype);
            writer.Write(entry.Shape.Length);
            foreach (int dim in entry.Shape) writer.Write(dim);
            writer.Write(entry.Offset);
        }
        long newIndexLen = outFs.Position - indexStart;

        // ── Rewrite the header with final values ──
        outFs.Position = headerPos;
        writer.Write(SmmConstants.Magic);
        writer.Write(SmmConstants.Version);
        writer.Write((long)metaBytes.Length);
        writer.Write((long)tokenizerBytes.Length);
        writer.Write(pluginAsmCount); // plugin assemblies copied verbatim
        writer.Write((long)newIndex.Count);
        writer.Write(newIndexLen);
        writer.Write(newDataOffset);
        writer.Write(0L); // reserved
        writer.Flush();
        progress?.Report(1f);
    }

    private static (byte[] bytes, QuantDType dtype) QuantizeTensor(byte[] raw, int[] shape, QuantDType source, QuantDType target)
    {
        int count = 1;
        foreach (int dim in shape) count *= dim;

        // Dequantize the source (F32 = verbatim floats) so any quantized source
        // (Q8_0 from a GGUF, F16, etc.) can be re-encoded to a leaner target.
        var values = new float[count];
        if (source == QuantDType.F32)
        {
            if (raw.Length != count * 4)
                throw new InvalidDataException(
                    $"Tensor claims F32 ({shape.Length}-D, {count} elements) but has {raw.Length} bytes.");
            Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        }
        else
        {
            try
            {
                using var ms = new MemoryStream(raw, writable: false);
                using var reader = new BinaryReader(ms);
                _readOps.ReadFor(source, reader, values, count);
            }
            catch (Exception ex) when (ex is EndOfStreamException or IOException)
            {
                throw new InvalidDataException(
                    $"Tensor has unsupported source dtype {source} ({shape.Length}-D, {count} elements).", ex);
            }
        }

        try
        {
            return (TensorQuantizer.Quantize(values, shape, target), target);
        }
        catch (InvalidOperationException)
        {
            // The target layout cannot encode this tensor's shape (e.g. a
            // K-quant needs 256-divisible flattened length). F16 is always safe —
            // but only if it is still leaner than the source, so we never
            // upscale a tensor that is already smaller than F16 would be.
            if (QuantizationOps.GetRawTensorByteCount(shape, QuantDType.F16) < raw.Length)
                return (TensorQuantizer.Quantize(values, shape, QuantDType.F16), QuantDType.F16);
            return (raw, source);
        }
    }

    private static void ReadHeader(
        BinaryReader reader,
        out long metaLen, out long tokenizerLen, out long pluginAsmCount,
        out long tensorCount, out long indexLen, out long dataOffset)
    {
        uint magic = reader.ReadUInt32();
        if (magic != SmmConstants.Magic)
            throw new InvalidDataException("Not SMM: " + magic.ToString("X8"));
        uint version = reader.ReadUInt32();
        if (version != SmmConstants.Version)
            throw new InvalidDataException("Unsupported SMM version: " + version);

        metaLen = reader.ReadInt64();
        tokenizerLen = reader.ReadInt64();
        pluginAsmCount = reader.ReadInt64();
        tensorCount = reader.ReadInt64();
        indexLen = reader.ReadInt64();
        dataOffset = reader.ReadInt64();
        reader.ReadInt64(); // reserved
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static long Align(long position, int alignment)
        => (position + alignment - 1) & ~(alignment - 1L);

    private static void MoveWithRetry(string sourcePath, string destPath)
    {
        const int maxAttempts = 6;
        const int delayMs = 100;
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                File.Move(sourcePath, destPath, overwrite: true);
                return;
            }
            catch (IOException) when (attempt < maxAttempts)
            {
                System.Threading.Thread.Sleep(delayMs);
            }
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* best effort cleanup */ }
    }
}