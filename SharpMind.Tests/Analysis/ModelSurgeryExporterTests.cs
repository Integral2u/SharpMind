using System.Buffers.Binary;
using SharpMind.Core.Quantization;
using SharpMind.Model.Analysis;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Analysis;

public sealed class ModelSurgeryExporterTests
{
    [Fact]
    public void ExportPruned_RenumbersBlocksAndFiltersPerLayerArrays()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        string destination = Path.Combine(temp.Path, "pruned.gguf");
        WriteSyntheticModel(source, layers: 3);

        SurgeryExportResult result = ModelSurgeryExporter.ExportPruned(source, destination, new HashSet<int> { 1 });

        Assert.True(File.Exists(destination));
        Assert.Equal(3, result.OriginalLayers);
        Assert.Equal(2, result.KeptLayers);
        Assert.Equal(1, result.RemovedLayers);

        ModelMetaData meta = GgufLoader.LoadMeta(destination);
        Assert.Equal(2L, meta.GetLong("qwen2.block_count"));
        Assert.Contains(meta.Tensors, tensor => tensor.Name == "blk.0.attn_q.weight");
        Assert.Contains(meta.Tensors, tensor => tensor.Name == "blk.1.attn_q.weight");
        Assert.DoesNotContain(meta.Tensors, tensor => tensor.Name.StartsWith("blk.2.", StringComparison.Ordinal));
        Assert.Equal(2, meta.Tensors.Count(tensor => tensor.Name.StartsWith("blk.", StringComparison.Ordinal)));

        int[] headCounts = Assert.IsType<int[]>(meta.KvPairs.First(pair => pair.Key == "qwen2.attention.head_count_kv").Value);
        Assert.Equal([4, 1], headCounts);
        bool[] sliding = Assert.IsType<bool[]>(meta.KvPairs.First(pair => pair.Key == "qwen2.attention.sliding_window").Value);
        Assert.Equal([false, true], sliding);
    }

    [Fact]
    public void ExportPruned_RejectsRemovingEveryLayer()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        string destination = Path.Combine(temp.Path, "pruned.gguf");
        WriteSyntheticModel(source, layers: 2);

        Assert.Throws<ArgumentException>(() =>
            ModelSurgeryExporter.ExportPruned(source, destination, new HashSet<int> { 0, 1 }));
    }

    private static void WriteSyntheticModel(string path, int layers)
    {
        var tensors = new List<GgufTensor>();
        for (int layer = 0; layer < layers; layer++)
        {
            byte[] bytes = TensorBytes(layer);
            tensors.Add(new GgufTensor
            {
                Name = $"blk.{layer}.attn_q.weight",
                Shape = [2, 2],
                Dtype = QuantDType.F32,
                GetBytes = () => bytes,
            });
        }
        tensors.Add(new GgufTensor
        {
            Name = "output_norm.weight",
            Shape = [2],
            Dtype = QuantDType.F32,
            GetBytes = () => TensorBytes(99)[..8],
        });

        Assert.Equal(layers, tensors.Count(tensor => tensor.Name.StartsWith("blk.", StringComparison.Ordinal)));

        GgufWriter.Write(
            path,
            [
                new GgufKvPair { Key = "general.architecture", Value = "qwen2" },
                new GgufKvPair { Key = "qwen2.block_count", Value = layers },
                new GgufKvPair { Key = "qwen2.attention.head_count_kv", Value = Enumerable.Range(0, layers).Select(i => i == 0 ? 4 : i == 1 ? 2 : 1).ToArray() },
                new GgufKvPair { Key = "qwen2.attention.sliding_window", Value = Enumerable.Range(0, layers).Select(i => i > 0).ToArray() },
            ],
            tensors);
    }

    private static byte[] TensorBytes(int seed)
    {
        var bytes = new byte[16];
        for (int i = 0; i < 4; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), seed + i);
        return bytes;
    }
}
