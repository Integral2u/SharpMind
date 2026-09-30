namespace SharpMind.Tests.ModelFormat;

using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using Xunit;

/// <summary>
/// Regression tests for the canonical GGUF type id ↔ <see cref="QuantDType"/> bridge.
/// The internal enum is a convenience index (I8=16 etc.), NOT the GGUF wire format,
/// so raw casts mis-decode real IQ2/IQ3 tensors as integer types. Every GGUF read
/// and write must go through <see cref="GgufTypeMap"/>, and unsupported-but-real
/// types must fail loudly instead of silently producing garbage weights.
/// </summary>
public class GgufTypeMapTests
{
    /// <summary>Canonical <c>ggml_type</c> ids SharpMind supports, and the internal dtype each must decode to.</summary>
    public static IEnumerable<object[]> SupportedIds() => new (uint, QuantDType)[]
    {
        (0, QuantDType.F32),
        (1, QuantDType.F16),
        (2, QuantDType.Q4_0),
        (3, QuantDType.Q4_1),
        (6, QuantDType.Q5_0),
        (7, QuantDType.Q5_1),
        (8, QuantDType.Q8_0),
        (9, QuantDType.Q8_1),
        (10, QuantDType.Q2_K),
        (11, QuantDType.Q3_K),
        (12, QuantDType.Q4_K),
        (13, QuantDType.Q5_K),
        (14, QuantDType.Q6_K),
        (15, QuantDType.Q8_K),
        (19, QuantDType.IQ1_S),
        (20, QuantDType.IQ4_NL),
        (24, QuantDType.I8),
        (25, QuantDType.I16),
        (26, QuantDType.I32),
        (29, QuantDType.IQ1_M),
        (30, QuantDType.BF16),
        (34, QuantDType.TQ1_0),
        (35, QuantDType.TQ2_0),
        (41, QuantDType.Q1_0),
    }.Select(p => new object[] { p.Item1, p.Item2 });

    /// <summary>Real <c>ggml_type</c> ids SharpMind intentionally does not implement — these must not silently mis-decode.</summary>
    public static IEnumerable<object[]> UnsupportedIds() => new uint[]
    {
        16, // IQ2_XXS
        17, // IQ2_XS
        18, // IQ3_XXS
        21, // IQ3_S
        22, // IQ2_S
        23, // IQ4_XS
        27, // I64
        28, // F64
        31, // Q4_0_4_4
        32, // Q4_0_4_8
        33, // Q4_0_8_8
        36, // IQ4_NL_4_4
        37, // IQ4_NL_4_8
        38, // IQ4_NL_8_8
        39, // MXFP4
        40, // NVFP4
    }.Select(id => new object[] { id });

    /// <summary>Ids GGML has never assigned — treat as unknown, not as a supported type.</summary>
    public static IEnumerable<object[]> UnknownIds() => new uint[]
    {
        4, 5, 42, 1000, uint.MaxValue,
    }.Select(id => new object[] { id });

    [Theory]
    [MemberData(nameof(SupportedIds))]
    public void FromGgufId_MapsSupportedIdsToCorrectDtype(uint id, QuantDType expected)
    {
        QuantDType dtype = GgufTypeMap.FromGgufId(id, "blk.0.attn_q.weight");
        Assert.Equal(expected, dtype);
    }

    [Theory]
    [MemberData(nameof(SupportedIds))]
    public void ToGgmlType_RoundTripsCanonicalIds(uint id, QuantDType dtype)
    {
        Assert.Equal(id, GgufTypeMap.ToGgmlType(dtype));
    }

    [Theory]
    [MemberData(nameof(SupportedIds))]
    public void ToGgmlType_FromGgufId_RoundTrip(uint id, QuantDType dtype)
    {
        // Ignore `id` here — it only documents the canonical value for each row;
        // the round-trip must hold for the dtype regardless of the original id.
        _ = id;
        Assert.Equal(dtype, GgufTypeMap.FromGgufId(GgufTypeMap.ToGgmlType(dtype)));
    }

    [Theory]
    [MemberData(nameof(UnsupportedIds))]
    public void FromGgufId_UnsupportedTypesFailLoudly(uint id)
    {
        const string tensorName = "output.weight";
        var ex = Assert.Throws<NotSupportedException>(() => GgufTypeMap.FromGgufId(id, tensorName));
        Assert.Contains(tensorName, ex.Message);
        Assert.Contains("is not supported by SharpMind", ex.Message);
    }

    [Theory]
    [MemberData(nameof(UnknownIds))]
    public void FromGgufId_UnknownIdFailsWithUnknownType(uint id)
    {
        var ex = Assert.Throws<NotSupportedException>(() => GgufTypeMap.FromGgufId(id));
        Assert.Contains("unknown GGML type id", ex.Message);
        Assert.Contains("is not supported by SharpMind", ex.Message);
    }

    [Fact]
    public void KQuantAliases_WriteAsBaseType()
    {
        Assert.Equal(10u, GgufTypeMap.ToGgmlType(QuantDType.Q2_K_S));
        Assert.Equal(11u, GgufTypeMap.ToGgmlType(QuantDType.Q3_K_S));
        Assert.Equal(11u, GgufTypeMap.ToGgmlType(QuantDType.Q3_K_M));
        Assert.Equal(11u, GgufTypeMap.ToGgmlType(QuantDType.Q3_K_L));
        Assert.Equal(12u, GgufTypeMap.ToGgmlType(QuantDType.Q4_K_S));
        Assert.Equal(12u, GgufTypeMap.ToGgmlType(QuantDType.Q4_K_M));
        Assert.Equal(13u, GgufTypeMap.ToGgmlType(QuantDType.Q5_K_S));
        Assert.Equal(13u, GgufTypeMap.ToGgmlType(QuantDType.Q5_K_M));
        Assert.Equal(14u, GgufTypeMap.ToGgmlType(QuantDType.Q6_K_S));

        // And they read back as their base type, not the alias.
        Assert.Equal(QuantDType.Q4_K, GgufTypeMap.FromGgufId(GgufTypeMap.ToGgmlType(QuantDType.Q4_K_M)));
    }

    [Theory]
    [InlineData(QuantDType.F32)]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.BF16)]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.Q4_K_M)]
    public void GgufWriter_ToGgufLoader_RoundTripsDtype(QuantDType dtype)
    {
        // A synthetic GGUF (no physical model involved) written by GgufWriter must
        // be read back by GgufLoader with the same dtype. The alias Q4_K_M round-trips
        // as its base Q4_K, matching what the wire format actually stores.
        string path = Path.Combine(Path.GetTempPath(), $"GgufTypeMap_rt_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensor = new GgufTensor
            {
                Name = "blk.0.attn_q.weight",
                Shape = [64, 64],
                Dtype = dtype,
                GetBytes = () => new byte[64 * 64 * 4],
            };
            GgufWriter.Write(path, [], [tensor]);

            var meta = GgufLoader.LoadMeta(path);
            var info = Assert.Single(meta.Tensors);
            Assert.Equal("blk.0.attn_q.weight", info.Name);
            Assert.Equal(new[] { 64, 64 }, info.Shape);
            QuantDType expected = dtype is QuantDType.Q4_K_M ? QuantDType.Q4_K : dtype;
            Assert.Equal(expected, info.Dtype);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Theory]
    [InlineData(16u, "IQ2_XXS")]
    [InlineData(18u, "IQ3_XXS")]
    [InlineData(39u, "MXFP4")]
    public void GgufLoadMeta_UnsupportedTensorTypeFailsLoudly(uint ggmlId, string expectedName)
    {
        // Hand-crafted GGUF v3 header with a single tensor of an unsupported type.
        // This is the file shape real llama.cpp models expose (e.g. IQ2_XXS quants).
        string path = Path.Combine(Path.GetTempPath(), $"GgufTypeMap_bad_{Guid.NewGuid():N}.gguf");
        try
        {
            using (var fs = File.Create(path))
            using (var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false))
            {
                const uint magic = 0x46554747;
                w.Write(magic);            // "GGUF"
                w.Write(3u);               // version
                w.Write(1L);               // tensor count
                w.Write(0L);               // kv count

                WriteGgufString(w, "output.weight");
                w.Write(2u);               // n_dims
                w.Write(8UL);              // dim 0
                w.Write(8UL);              // dim 1
                w.Write(ggmlId);           // canonical ggml_type id
                w.Write(0UL);              // offset
            }

            var ex = Assert.Throws<NotSupportedException>(() => GgufLoader.LoadMeta(path));
            Assert.Contains(expectedName, ex.Message);
            Assert.Contains("output.weight", ex.Message);
            Assert.Contains("is not supported by SharpMind", ex.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static void WriteGgufString(BinaryWriter w, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }
}