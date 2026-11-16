using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// Which kernel a linear layer runs is decided by the mapping's suffix, and the
/// suffix used to be guessed from the host ISA. That is the same question only
/// on x86: every <c>Fma/Avx2/Sse3.IsSupported</c> check is false on Arm64, so
/// every quantized layer selected <c>_scalar</c> and the NEON matmuls were
/// reachable from <c>kbench</c> alone. These tests pin the mapping as the
/// source of truth and check that whatever it names is actually registered.
/// </summary>
public class QuantCompoundSelectionTests
{
    private static readonly QuantDType[] LayerDtypes =
    [
        QuantDType.F32, QuantDType.F16, QuantDType.BF16,
        QuantDType.Q8_0, QuantDType.Q8_1, QuantDType.Q4_0, QuantDType.Q4_1,
        QuantDType.Q5_0, QuantDType.Q5_1,
        QuantDType.Q2_K, QuantDType.Q3_K, QuantDType.Q4_K_M, QuantDType.Q5_K_M,
        QuantDType.Q6_K, QuantDType.Q8_K,
        QuantDType.IQ4_NL, QuantDType.MXFP4, QuantDType.Q1_0, QuantDType.TQ2_0,
    ];

    private static PuzzleCornerPiece LinearTable() =>
        typeof(InferenceLinearLayer)
            .GetMethod(nameof(InferenceLinearLayer.QuantizedMatMulFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!;

    private static Dictionary<string, string> ScalarMapping() =>
        new MappingBuilder(HardwareTier.Scalar).ApplyQuantPreset().Build();

    [Fact]
    public void CompoundFollowsTheMappingTier_NotTheHostIsa()
    {
        // Splice in what the Arm64 preset produces for the dtypes that have a
        // NEON matmul, without needing Arm64 hardware to ask.
        var mapping = ScalarMapping();
        foreach (string prefix in new[] { "q4k", "q6k", "q8_0", "mxfp4" })
            mapping[$"qmatmul_{prefix}"] = $"qmatmul_{prefix}_parallel_arm64";

        Assert.Equal("q4k_parallel_arm64", QuantHelper.GetCompound(QuantDType.Q4_K_M, mapping));
        Assert.Equal("q6k_parallel_arm64", QuantHelper.GetCompound(QuantDType.Q6_K, mapping));
        Assert.Equal("q8_0_parallel_arm64", QuantHelper.GetCompound(QuantDType.Q8_0, mapping));
        Assert.Equal("mxfp4_parallel_arm64", QuantHelper.GetCompound(QuantDType.MXFP4, mapping));

        // A dtype with no NEON kernel must fall back to the tier's scalar target
        // rather than inventing an `_arm64` name that is not registered.
        Assert.Equal("q2k_parallel_scalar", QuantHelper.GetCompound(QuantDType.Q2_K, mapping));
    }

    [Fact]
    public void SerialPreset_KeepsTheSerialSuffix()
    {
        var mapping = new MappingBuilder(HardwareTier.Scalar).ApplyQuantPreset(parallel: false).Build();
        Assert.Equal("mxfp4_serial_scalar", QuantHelper.GetCompound(QuantDType.MXFP4, mapping));
    }

    [Theory]
    [InlineData(HardwareTier.Scalar)]
    [InlineData(HardwareTier.SSE)]
    [InlineData(HardwareTier.AVX2)]
    [InlineData(HardwareTier.FMA)]
    public void EveryCompound_ResolvesToARegisteredKernel(HardwareTier tier)
    {
        if (!HardwareTierHelpers.IsSupported(tier)) return;

        var table = LinearTable().KeyValues.Keys;
        foreach (bool parallel in new[] { true, false })
        {
            var mapping = new MappingBuilder(tier).ApplyQuantPreset(parallel).Build();
            foreach (var dtype in LayerDtypes)
            {
                string compound = QuantHelper.GetCompound(dtype, mapping);
                Assert.True(table.Contains(compound),
                    $"{dtype} at {tier} (parallel: {parallel}) selects '{compound}', which the " +
                    "linear-layer dispatch table does not register.");
            }
        }
    }

    [Fact]
    public void EveryLayerDtype_HasItsOwnPrefix()
    {
        // The old catch-all returned "f32", so an unlisted dtype silently ran its
        // raw block bytes through the float kernel. No two layer dtypes may share
        // a prefix, and none may borrow F32's.
        var prefixes = LayerDtypes.ToDictionary(d => d, d => QuantHelper.DtypeToCompound(d));
        Assert.Equal(LayerDtypes.Length, prefixes.Values.Distinct().Count());
        Assert.DoesNotContain("f32", prefixes.Where(kv => kv.Key is not QuantDType.F32).Select(kv => kv.Value));
    }

    [Theory]
    [InlineData(QuantDType.IQ1_S)]
    [InlineData(QuantDType.IQ1_M)]
    [InlineData(QuantDType.TQ1_0)]
    public void DecodeOnlyDtypes_FailLoudlyInsteadOfUsingTheFloatKernel(QuantDType dtype)
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => QuantHelper.GetCompound(dtype, ScalarMapping()));
        Assert.Contains(dtype.ToString(), ex.Message);
        Assert.Contains("no matmul target", ex.Message);
    }
}
