using System.Buffers.Binary;
using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

public sealed class GgufQuantizerTests
{
    [Fact]
    public void QuantizeFixed_WritesF16AndKeepsValues()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        string destination = Path.Combine(temp.Path, "f16.gguf");
        float[] attention = WriteSynthetic(source);

        GgufQuantizeResult result = GgufQuantizer.Quantize(source, destination, QuantDType.F16);

        Assert.Equal(3, result.Tensors);
        Assert.Equal(3, result.Requantized);
        Assert.Equal(0, result.Copied);
        ModelMetaData meta = GgufLoader.LoadMeta(destination);
        Assert.All(meta.Tensors, tensor => Assert.Equal(QuantDType.F16, tensor.Dtype));
        Assert.Equal(3, result.Counts[QuantDType.F16]);
        using var reader = new GgufTensorReader(destination);
        AssertClose(attention, reader.ReadFloat32("blk.0.attn_q.weight"), 0.01f);
    }

    [Fact]
    public void QuantizeFixed_KeepsNormsAtF16ForAQ8Target()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        string destination = Path.Combine(temp.Path, "q8.gguf");
        WriteSynthetic(source);

        GgufQuantizeResult result = GgufQuantizer.Quantize(source, destination, QuantDType.Q8_0);

        ModelMetaData meta = GgufLoader.LoadMeta(destination);
        Assert.Equal(QuantDType.Q8_0, Dtype(meta, "token_embd.weight"));
        Assert.Equal(QuantDType.Q8_0, Dtype(meta, "blk.0.attn_q.weight"));
        Assert.Equal(QuantDType.F16, Dtype(meta, "blk.0.attn_norm.weight"));
        Assert.Equal(2, result.Counts[QuantDType.Q8_0]);
        Assert.Equal(1, result.Counts[QuantDType.F16]);
    }

    [Fact]
    public void QuantizeFixed_FallsBackFromKQuantOnUnalignedRows()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        string destination = Path.Combine(temp.Path, "q4k.gguf");
        WriteSynthetic(
            source,
            ("blk.0.attn_q.weight", [64, 4]),
            ("blk.0.ffn_down.weight", [256, 2]),
            ("blk.0.attn_norm.weight", [32]));

        GgufQuantizer.Quantize(source, destination, QuantDType.Q4_K);

        ModelMetaData meta = GgufLoader.LoadMeta(destination);
        Assert.Equal(QuantDType.Q8_0, Dtype(meta, "blk.0.attn_q.weight"));
        Assert.Equal(QuantDType.Q4_K, Dtype(meta, "blk.0.ffn_down.weight"));
        Assert.Equal(QuantDType.F16, Dtype(meta, "blk.0.attn_norm.weight"));
    }

    [Fact]
    public void Quantize_RejectsTheSameSourceAndDestination()
    {
        using var temp = new TempDirectory();
        string source = Path.Combine(temp.Path, "source.gguf");
        WriteSynthetic(source);

        Assert.Throws<ArgumentException>(() =>
            GgufQuantizer.Quantize(source, source, QuantDType.F16));
    }

    [Fact]
    public void Plan_BudgetPicksTheCoarsestFittingKQuant()
    {
        var tensors = new List<TensorInfo>
        {
            new() { Name = "blk.0.attn_q.weight", Dtype = QuantDType.F32, Shape = [256, 16], Offset = 0 },
            new() { Name = "blk.0.attn_norm.weight", Dtype = QuantDType.F32, Shape = [256], Offset = 0 },
        };
        // 100,000-byte source minus the 16,384-byte quantizable tensor = 83,616 fixed.
        // A 1,500-byte data budget fits Q2_K (1,344) but not Q3_K (1,760).
        var options = new GgufQuantOptions
        {
            TargetBytes = 85_116,
            Floor = QuantDType.Q2_K,
        };

        Dictionary<string, QuantDType> plan = GgufQuantPlan.Resolve(tensors, options, sourceLength: 100_000);

        Assert.Equal(QuantDType.Q2_K, plan["blk.0.attn_q.weight"]);
        Assert.Equal(QuantDType.F16, plan["blk.0.attn_norm.weight"]);
    }

    [Fact]
    public void Plan_HonoursPerRoleOverrides()
    {
        var tensors = new List<TensorInfo>
        {
            new() { Name = "token_embd.weight", Dtype = QuantDType.F32, Shape = [256, 8], Offset = 0 },
            new() { Name = "blk.0.attn_q.weight", Dtype = QuantDType.F32, Shape = [256, 8], Offset = 0 },
            new() { Name = "blk.0.attn_norm.weight", Dtype = QuantDType.F32, Shape = [256], Offset = 0 },
        };
        var options = new GgufQuantOptions
        {
            DefaultLevel = QuantDType.Q4_K,
            RoleLevels = new Dictionary<SmmTensorRole, QuantDType>
            {
                [SmmTensorRole.Embedding] = QuantDType.Q8_K,
            },
        };

        Dictionary<string, QuantDType> plan = GgufQuantPlan.Resolve(tensors, options, sourceLength: 1_000_000);

        Assert.Equal(QuantDType.Q8_K, plan["token_embd.weight"]);
        Assert.Equal(QuantDType.Q4_K, plan["blk.0.attn_q.weight"]);
        Assert.Equal(QuantDType.F16, plan["blk.0.attn_norm.weight"]);
    }

    private static QuantDType Dtype(ModelMetaData meta, string name) =>
        meta.Tensors.First(tensor => tensor.Name == name).Dtype;

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, tolerance);
    }

    private static float[] WriteSynthetic(
        string path,
        params (string Name, int[] Shape)[]? shapes)
    {
        var entries = shapes is { Length: > 0 }
            ? shapes
            :
            [
                ("token_embd.weight", new[] { 32, 8 }),
                ("blk.0.attn_q.weight", new[] { 64, 4 }),
                ("blk.0.attn_norm.weight", new[] { 32 }),
            ];

        var tensors = new List<GgufTensor>();
        float[] attention = [];
        int seed = 17;
        foreach ((string name, int[] shape) in entries)
        {
            int count = shape.Aggregate(1, (total, dimension) => total * dimension);
            var values = new float[count];
            var random = new Random(seed++);
            for (int i = 0; i < count; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
            if (name == "blk.0.attn_q.weight") attention = values;

            byte[] bytes = FloatBytes(values);
            tensors.Add(new GgufTensor
            {
                Name = name,
                Shape = shape,
                Dtype = QuantDType.F32,
                GetBytes = () => bytes,
            });
        }

        GgufWriter.Write(
            path,
            [new GgufKvPair { Key = "general.architecture", Value = "qwen2" }],
            tensors);
        return attention;
    }

    private static byte[] FloatBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        return bytes;
    }
}
