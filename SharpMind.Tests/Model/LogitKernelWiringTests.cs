using System.Linq;
using System.Reflection;
using JigSawDotNet;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The LM head and the layer projections are the same operation on different
/// tensors, so their JigSaw dispatch tables must name the same kernel for the
/// same dtype+ISA key. They are written out as two separate literal attribute
/// lists, which is exactly how they drift: the F32 and F16 rows in
/// <see cref="LogitOps"/> once pointed at the scalar kernels on every tier,
/// including <c>_fma</c>, so the largest matmul in the model — the LM head, over
/// a 151936-row vocab — silently ran unvectorised while every layer beside it
/// did not.
/// </summary>
public class LogitKernelWiringTests
{
    [Fact]
    public void LogitAndLinearDispatchTablesAgree()
    {
        var logit = typeof(LogitOps)
            .GetMethod(nameof(LogitOps.ProjectFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!;
        var linear = typeof(InferenceLinearLayer)
            .GetMethod(nameof(InferenceLinearLayer.QuantizedMatMulFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!;

        Assert.Empty(logit.KeyValues.Keys.Except(linear.KeyValues.Keys));
        Assert.Empty(linear.KeyValues.Keys.Except(logit.KeyValues.Keys));

        var mismatched = logit.KeyValues.Keys
            .Where(k => logit.KeyValues[k] != linear.KeyValues[k])
            .Select(k => $"{k}: logit={Leaf(logit.KeyValues[k])} linear={Leaf(linear.KeyValues[k])}")
            .ToList();

        Assert.True(mismatched.Count == 0,
            "LogitOps and InferenceLinearLayer must dispatch the same key to the same kernel:\n  " +
            string.Join("\n  ", mismatched));
    }

    /// <summary>
    /// Pins the specific regression: a vectorised-tier key must never resolve to
    /// a scalar kernel. Table equality alone would still pass if both tables
    /// were downgraded together.
    ///
    /// The grid-based formats (IQ1_S, IQ1_M, TQ1_0, IQ2_XXS, IQ2_XS, IQ3_XXS)
    /// have no vector kernel yet — the grid lookup is a gather — so their
    /// tier keys deliberately resolve to the scalar implementation, exactly as
    /// they do in <see cref="SharpMind.Core.Quantization.QuantizationOps"/>.
    /// </summary>
    [Fact]
    public void FmaAndAvx2KeysNeverResolveToScalarKernels()
    {
        string[] scalarOnly =
            ["iq1_s_", "iq1_m_", "tq1_0_", "iq2_xxs_", "iq2_xs_", "iq3_xxs_", "iq2_s_", "iq3_s_"];

        var logit = typeof(LogitOps)
            .GetMethod(nameof(LogitOps.ProjectFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!;

        var downgraded = logit.KeyValues
            .Where(kv => (kv.Key.EndsWith("_fma") || kv.Key.EndsWith("_avx2"))
                      && !scalarOnly.Any(p => kv.Key.StartsWith(p, StringComparison.Ordinal))
                      && Leaf(kv.Value).EndsWith("_Scalar"))
            .Select(kv => $"{kv.Key} -> {Leaf(kv.Value)}")
            .ToList();

        Assert.True(downgraded.Count == 0,
            "These vectorised-tier keys resolve to a scalar kernel:\n  " + string.Join("\n  ", downgraded));
    }

    /// <summary>
    /// The layer table is a second copy of the kernel choice that
    /// <see cref="SharpMind.Core.Quantization.QuantizationOps"/> already makes,
    /// so the two must name the same implementation for the same suffix. A
    /// mismatch means a projection runs a different kernel than the standalone
    /// ops path — the shape of bug that put IQ1_S/TQ1_0 tensors through the F32
    /// kernel, and that left the Arm64 tiers on scalar.
    /// </summary>
    [Fact]
    public void LinearDispatchMatchesTheQuantizationOpsTable()
    {
        const string prefix = "qmatmul_";
        var ops = typeof(SharpMind.Core.Quantization.QuantizationOps)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Select(m => m.GetCustomAttribute<PuzzleCornerPiece>())
            .Where(a => a != null)
            .SelectMany(a => a!.KeyValues)
            .Where(kv => kv.Key.StartsWith(prefix, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[prefix.Length..], kv => kv.Value, StringComparer.Ordinal);

        var linear = typeof(InferenceLinearLayer)
            .GetMethod(nameof(InferenceLinearLayer.QuantizedMatMulFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!
            .KeyValues;

        // The raw integer types are tensor storage, never a linear weight, and
        // have no compound name; everything else must be wired.
        string[] notLayerWeights = ["i8_", "i16_", "i32_"];

        var missing = ops.Keys
            .Where(k => !linear.ContainsKey(k))
            .Where(k => !notLayerWeights.Any(p => k.StartsWith(p, StringComparison.Ordinal)))
            .OrderBy(k => k, StringComparer.Ordinal)
            .ToList();
        Assert.True(missing.Count == 0,
            "These QuantizationOps matmul keys have no linear-layer row:\n  " + string.Join("\n  ", missing));

        var mismatched = linear
            .Where(kv => ops.TryGetValue(kv.Key, out string? impl) && impl != kv.Value)
            .Select(kv => $"{kv.Key}: linear={Leaf(kv.Value)} ops={Leaf(ops[kv.Key])}")
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToList();
        Assert.True(mismatched.Count == 0,
            "The linear layer and QuantizationOps must dispatch a key to the same kernel:\n  " +
            string.Join("\n  ", mismatched));
    }

    private static string Leaf(string fullName) => fullName[(fullName.LastIndexOf('.') + 1)..];
}
