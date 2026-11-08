using SharpMind.Core.Quantization;
using SharpMind.Model.Format;

namespace SharpMind.Model.Analysis;

/// <summary>Summary of one pruned GGUF export.</summary>
public sealed record SurgeryExportResult(
    string Path,
    int OriginalLayers,
    int KeptLayers,
    int RemovedLayers,
    int Tensors,
    long Bytes);

/// <summary>
/// Writes a new GGUF with selected transformer blocks removed. Block tensors are renumbered,
/// <c>{arch}.block_count</c> is updated and any per-layer metadata array is filtered with the
/// same indices, so the result loads as an ordinary smaller model. Global tensors and the
/// tokenizer are copied byte for byte (still quantized).
/// </summary>
public static class ModelSurgeryExporter
{
    public static SurgeryExportResult ExportPruned(
        string sourcePath,
        string destinationPath,
        IReadOnlySet<int> removeLayers,
        Action<int, int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);
        ArgumentNullException.ThrowIfNull(removeLayers);
        if (!File.Exists(sourcePath)) throw new FileNotFoundException("Source model not found.", sourcePath);
        if (Path.GetFullPath(sourcePath).Equals(Path.GetFullPath(destinationPath), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("The destination must differ from the source.", nameof(destinationPath));

        ModelMetaData meta = GgufLoader.LoadMeta(sourcePath);
        string arch = meta.GetString("general.architecture");
        if (string.IsNullOrWhiteSpace(arch))
            throw new InvalidOperationException("The source model has no architecture key.");
        int originalLayers = (int)meta.GetLong($"{arch}.block_count");
        if (originalLayers <= 0)
            throw new InvalidOperationException("The source model reports no transformer blocks.");
        if (removeLayers.Any(layer => layer < 0 || layer >= originalLayers))
            throw new ArgumentOutOfRangeException(nameof(removeLayers), "A removed layer is outside the model.");

        var keep = Enumerable.Range(0, originalLayers)
            .Where(layer => !removeLayers.Contains(layer))
            .ToArray();
        if (keep.Length == 0)
            throw new ArgumentException("At least one layer must remain.", nameof(removeLayers));
        var newIndex = new Dictionary<int, int>(keep.Length);
        for (int index = 0; index < keep.Length; index++) newIndex[keep[index]] = index;

        var kvPairs = new List<GgufKvPair>(meta.KvPairs.Count + 1);
        bool blockCountWritten = false;
        foreach (KvPair pair in meta.KvPairs)
        {
            object value = pair.Value;
            if (pair.Key.Equals($"{arch}.block_count", StringComparison.OrdinalIgnoreCase))
            {
                value = keep.Length;
                blockCountWritten = true;
            }
            else if (IsPerLayerArray(pair.Key, arch, originalLayers, value))
            {
                value = FilterArray(value, removeLayers);
            }
            kvPairs.Add(new GgufKvPair { Key = pair.Key, Value = value });
        }
        if (!blockCountWritten)
            kvPairs.Add(new GgufKvPair { Key = $"{arch}.block_count", Value = keep.Length });

        string? directory = Path.GetDirectoryName(Path.GetFullPath(destinationPath));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        using var source = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        var tensors = new List<GgufTensor>(meta.Tensors.Count);
        foreach (TensorInfo tensor in meta.Tensors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string name = tensor.Name;
            if (TryParseBlock(name, out int layer, out string suffix))
            {
                if (!newIndex.TryGetValue(layer, out int replacement)) continue;
                name = $"blk.{replacement}.{suffix}";
            }

            long offset = meta.DataOffset + tensor.Offset;
            long byteCount = QuantizationOps.GetRawTensorByteCount(tensor.Shape, tensor.Dtype);
            if (byteCount <= 0 || byteCount > int.MaxValue)
                throw new InvalidDataException($"Tensor '{tensor.Name}' has an invalid byte size {byteCount}.");

            tensors.Add(new GgufTensor
            {
                Name = name,
                Shape = tensor.Shape,
                Dtype = tensor.Dtype,
                GetBytes = () =>
                {
                    var bytes = new byte[byteCount];
                    source.Position = offset;
                    source.ReadExactly(bytes);
                    return bytes;
                },
            });
            progress?.Invoke(tensors.Count, meta.Tensors.Count);
        }

        GgufWriter.Write(destinationPath, kvPairs, tensors);
        return new SurgeryExportResult(
            destinationPath,
            originalLayers,
            keep.Length,
            originalLayers - keep.Length,
            tensors.Count,
            new FileInfo(destinationPath).Length);
    }

    private static bool TryParseBlock(string name, out int layer, out string suffix)
    {
        layer = -1;
        suffix = "";
        if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
        int dot = name.IndexOf('.', 4);
        if (dot < 0) return false;
        if (!int.TryParse(name.AsSpan(4, dot - 4), out layer)) return false;
        suffix = name[(dot + 1)..];
        return true;
    }

    private static bool IsPerLayerArray(string key, string arch, int layers, object value)
    {
        if (!key.StartsWith(arch + ".", StringComparison.OrdinalIgnoreCase)) return false;
        if (!(key.Contains("attention", StringComparison.OrdinalIgnoreCase) ||
              key.Contains("layer", StringComparison.OrdinalIgnoreCase) ||
              key.Contains("block", StringComparison.OrdinalIgnoreCase) ||
              key.Contains("sliding", StringComparison.OrdinalIgnoreCase))) return false;
        return value is Array array && array.Length == layers;
    }

    private static object FilterArray(object value, IReadOnlySet<int> remove)
    {
        static T[] Keep<T>(T[] source, IReadOnlySet<int> remove)
        {
            var result = new List<T>(source.Length - remove.Count);
            for (int i = 0; i < source.Length; i++)
                if (!remove.Contains(i)) result.Add(source[i]);
            return [.. result];
        }

        return value switch
        {
            int[] array => Keep(array, remove),
            uint[] array => Keep(array, remove),
            long[] array => Keep(array, remove),
            float[] array => Keep(array, remove),
            string[] array => Keep(array, remove),
            bool[] array => Keep(array, remove),
            _ => value,
        };
    }
}
