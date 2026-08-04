using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.IO.MemoryMappedFiles;

namespace SharpMind.Model.Format.Conversion;

/// <summary>
/// Converts a GGUF model into a SharpMind Model (.SMM) container.
///
/// The GGUF tokenizer and chat template are embedded in the container and
/// every tensor's raw bytes are streamed verbatim from the source file (no
/// re-quantization, no full-model buffer), so the result loads via the same
/// <see cref="SmmLoader"/> path used for training exports.
/// </summary>
public static class GgufToSmmConverter
{
    /// <summary>
    /// Converts <paramref name="ggufPath"/> to <paramref name="smmPath"/>.
    /// Pass <see cref="SmmWriteOptions"/> to control per-tensor compression
    /// (default <see cref="CompressionMode.Auto"/>).
    /// </summary>
    public static void Convert(string ggufPath, string smmPath, SmmWriteOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ggufPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(smmPath);
        if (!File.Exists(ggufPath)) throw new FileNotFoundException(ggufPath);

        var meta = GgufLoader.LoadMeta(ggufPath);
        var config = GgufLoader.LoadConfig(meta)
            ?? throw new InvalidDataException("GGUF is missing architecture metadata.");

        var tokenizer = GgufLoader.LoadTokenizerFromMeta(meta, config.VocabSize);
        string? chatTemplate = meta.GetChatTemplate();

        using var mmf = MemoryMappedFile.CreateFromFile(
            ggufPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);

        var tensors = new List<SmmTensorData>(meta.Tensors.Count);
        foreach (var info in meta.Tensors)
        {
            long rawSize = QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
            if (rawSize <= 0) continue;
            tensors.Add(new SmmTensorData
            {
                Name = info.Name,
                Shape = info.Shape,
                Dtype = info.Dtype,
                GetBytes = () => ReadTensorBytes(stream, meta, info, rawSize),
            });
        }

        SmmWriter.Write(smmPath, config, tokenizer, chatTemplate, tensors, options);
    }

    private static byte[] ReadTensorBytes(MemoryMappedViewStream stream, ModelMetaData meta, TensorInfo info, long rawSize)
    {
        long position = meta.DataOffset + info.Offset;
        if (position < 0 || position + rawSize > stream.Length)
            throw new InvalidDataException($"Tensor '{info.Name}' range is out of bounds.");
        stream.Position = position;
        var bytes = new byte[rawSize];
        stream.ReadExactly(bytes);
        return bytes;
    }
}
