using System.Buffers.Binary;
using SharpMind.Core.Quantization;
using SharpMind.Model.Analysis;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Analysis;

public sealed class QuantizationAnalyzerTests
{
    [Fact]
    public void Compare_F32AgainstF16_ReportsASmallError()
    {
        using var temp = new TempDirectory();
        string reference = Path.Combine(temp.Path, "reference.gguf");
        string candidate = Path.Combine(temp.Path, "candidate.gguf");
        var values = new float[16];
        for (int i = 0; i < values.Length; i++) values[i] = (i - 8) * 0.125f;

        WriteGguf(reference, values, QuantDType.F32);
        WriteGguf(candidate, values, QuantDType.F16);

        QuantizationComparisonReport report = QuantizationAnalyzer.Compare(reference, candidate);

        Assert.Equal(1, report.Compared);
        Assert.Equal(0, report.Skipped);
        Assert.Equal(16, report.Elements);
        Assert.True(report.GlobalRelativeL2 < 0.01, $"relative L2 {report.GlobalRelativeL2}");
        Assert.True(report.MeanCosine > 0.999, $"cosine {report.MeanCosine}");
        TensorQuantizationError tensor = Assert.Single(report.Tensors);
        Assert.Equal("w", tensor.Name);
        Assert.Equal("F32", tensor.ReferenceDtype);
        Assert.Equal("F16", tensor.CandidateDtype);
        Assert.True(report.SameModel);
        Assert.Null(report.CompatibilityNote);
    }

    [Fact]
    public void Compare_FlagsDifferentShapesAsIncompatible()
    {
        using var temp = new TempDirectory();
        string reference = Path.Combine(temp.Path, "reference.gguf");
        string candidate = Path.Combine(temp.Path, "candidate.gguf");
        WriteGguf(reference, [1, 2, 3, 4], QuantDType.F32, shape: [2, 2]);
        WriteGguf(candidate, [1, 2, 3, 4], QuantDType.F16, shape: [4]);

        QuantizationComparisonReport report = QuantizationAnalyzer.Compare(reference, candidate);

        Assert.False(report.SameModel);
        Assert.NotNull(report.CompatibilityNote);
        Assert.Equal(["w"], report.MismatchedTensors);
    }

    [Fact]
    public void Compare_FlagsDifferentArchitecturesAsIncompatible()
    {
        using var temp = new TempDirectory();
        string reference = Path.Combine(temp.Path, "reference.gguf");
        string candidate = Path.Combine(temp.Path, "candidate.gguf");
        WriteGguf(reference, [1, 2, 3, 4], QuantDType.F32, architecture: "qwen2", shape: [2, 2]);
        WriteGguf(candidate, [1, 2, 3, 4], QuantDType.F16, architecture: "llama", shape: [2, 2]);

        QuantizationComparisonReport report = QuantizationAnalyzer.Compare(reference, candidate);

        Assert.False(report.SameModel);
        Assert.Contains("architecture", report.CompatibilityNote, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Compare_SkipsATensorMissingFromTheCandidate()
    {
        using var temp = new TempDirectory();
        string reference = Path.Combine(temp.Path, "reference.gguf");
        string candidate = Path.Combine(temp.Path, "candidate.gguf");
        WriteGguf(reference, [1, 2, 3, 4], QuantDType.F32);
        WriteGguf(candidate, [1, 2, 3, 4], QuantDType.F32, name: "other");

        QuantizationComparisonReport report = QuantizationAnalyzer.Compare(reference, candidate);

        Assert.Equal(0, report.Compared);
        Assert.Equal(1, report.Skipped);
        Assert.False(report.SameModel);
    }

    private static void WriteGguf(
        string path, float[] values, QuantDType dtype,
        string name = "w", int[]? shape = null, string architecture = "qwen2")
    {
        byte[] bytes;
        if (dtype == QuantDType.F32)
        {
            bytes = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }
        else if (dtype == QuantDType.F16)
        {
            bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(i * 2), (Half)values[i]);
        }
        else
        {
            throw new NotSupportedException(dtype.ToString());
        }

        GgufWriter.Write(
            path,
            [new GgufKvPair { Key = "general.architecture", Value = architecture }],
            [new GgufTensor { Name = name, Shape = shape ?? [4, 4], Dtype = dtype, GetBytes = () => bytes }]);
    }
}
