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
    /// Pins the specific regression: an FMA-tier key must never resolve to a
    /// scalar kernel. Table equality alone would still pass if both tables were
    /// downgraded together.
    /// </summary>
    [Fact]
    public void FmaAndAvx2KeysNeverResolveToScalarKernels()
    {
        var logit = typeof(LogitOps)
            .GetMethod(nameof(LogitOps.ProjectFn), BindingFlags.Public | BindingFlags.Instance)!
            .GetCustomAttribute<PuzzleCornerPiece>()!;

        var downgraded = logit.KeyValues
            .Where(kv => (kv.Key.EndsWith("_fma") || kv.Key.EndsWith("_avx2")) && Leaf(kv.Value).EndsWith("_Scalar"))
            .Select(kv => $"{kv.Key} -> {Leaf(kv.Value)}")
            .ToList();

        Assert.True(downgraded.Count == 0,
            "These vectorised-tier keys resolve to a scalar kernel:\n  " + string.Join("\n  ", downgraded));
    }

    private static string Leaf(string fullName) => fullName[(fullName.LastIndexOf('.') + 1)..];
}
