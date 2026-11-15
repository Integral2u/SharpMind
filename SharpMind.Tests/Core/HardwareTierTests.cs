using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.Tests.Core;

/// <summary>
/// The hardware tier is a static, process-wide decision. Half of every assertion
/// here only means something on the other architecture, which is why the same
/// source also proves that x86 rejects the Arm64 tier and that Arm64 rejects the
/// x86 tiers - CI runs this file on both, so neither branch goes unchecked.
/// </summary>
public class HardwareTierTests
{
    [Fact]
    public void TierValuesAreFrozen_BecauseSavedSessionsPersistTheNumber()
    {
        // SessionOptions is serialized as plain JSON, so HardwareTier round-trips
        // as its numeric value. Renumbering would silently turn a saved "SSE"
        // preset into a different tier; Arm64 is appended for that reason.
        Assert.Equal(0, (int)HardwareTier.Auto);
        Assert.Equal(1, (int)HardwareTier.FMA);
        Assert.Equal(2, (int)HardwareTier.AVX2);
        Assert.Equal(3, (int)HardwareTier.SSE);
        Assert.Equal(4, (int)HardwareTier.Scalar);
        Assert.Equal(5, (int)HardwareTier.Arm64);
    }

    [Fact]
    public void DetectBestTier_IsAlwaysATierThisCpuCanRun()
    {
        var tier = HardwareTierHelpers.DetectBestTier();

        Assert.NotEqual(HardwareTier.Auto, tier);
        Assert.True(HardwareTierHelpers.IsSupported(tier));
        Assert.Equal(tier, HardwareTierHelpers.EnsureUsable(tier));
    }

    [Fact]
    public void DetectBestTier_ReportsTheTierOfTheProcessArchitecture()
    {
        bool isArm64 = RuntimeInformation.ProcessArchitecture == Architecture.Arm64;

        Assert.Equal(isArm64, HardwareTierHelpers.DetectBestTier() == HardwareTier.Arm64);
    }

    [Fact]
    public void SelectableTiers_OffersEverySupportedTierBestFirst()
    {
        var selectable = HardwareTierHelpers.SelectableTiers;

        // The options screen renders this list as a radio group, so Auto leads and
        // the detector's answer is the next entry down.
        Assert.Equal(HardwareTier.Auto, selectable[0]);
        Assert.Equal(HardwareTierHelpers.DetectBestTier(), selectable[1]);
        Assert.Equal(selectable.Count, selectable.Distinct().Count());
        Assert.All(selectable, tier => Assert.True(HardwareTierHelpers.IsSupported(tier)));

        // Every tier this CPU can run is offered; a new enum member that is not
        // added to the list would otherwise vanish from the screen silently.
        var supported = Enum.GetValues<HardwareTier>()
            .Where(HardwareTierHelpers.IsSupported)
            .OrderBy(tier => tier);
        Assert.Equal(supported, selectable.OrderBy(tier => tier));
    }

    [Fact]
    public void IsSupported_IsExclusivePerInstructionSetFamily()
    {
        // AdvSimd and AVX2 are never both available. That mutual exclusion is what
        // lets "Arm64 ranks below AVX2" be a static decision instead of a runtime
        // comparison between architectures that cannot coexist in one process.
        Assert.False(Avx2.IsSupported && AdvSimd.Arm64.IsSupported);
        Assert.Equal(Avx2.IsSupported, HardwareTierHelpers.IsSupported(HardwareTier.AVX2));
        Assert.Equal(Fma.IsSupported, HardwareTierHelpers.IsSupported(HardwareTier.FMA));
        Assert.Equal(AdvSimd.Arm64.IsSupported, HardwareTierHelpers.IsSupported(HardwareTier.Arm64));
        Assert.True(HardwareTierHelpers.IsSupported(HardwareTier.Scalar));
    }

    [Fact]
    public void EnsureUsable_RejectsATierFromTheOtherArchitecture()
    {
        HardwareTier foreign = AdvSimd.Arm64.IsSupported ? HardwareTier.AVX2 : HardwareTier.Arm64;

        var ex = Assert.Throws<PlatformNotSupportedException>(
            () => HardwareTierHelpers.EnsureUsable(foreign));

        Assert.Contains(foreign.ToString(), ex.Message);
        Assert.Contains(RuntimeInformation.ProcessArchitecture.ToString(), ex.Message);
    }

    [Fact]
    public void MappingBuilder_ThrowsWhenForcedToATierThisCpuCannotRun()
    {
        // A forced tier is a benchmarking escape hatch; reaching a kernel that
        // cannot exist on this CPU must be a message, not a fault.
        HardwareTier foreign = AdvSimd.Arm64.IsSupported ? HardwareTier.FMA : HardwareTier.Arm64;

        Assert.Throws<PlatformNotSupportedException>(() => new MappingBuilder(foreign));
    }

    [Fact]
    public void MappingBuilder_AutoAssemblesWhateverThisCpuIs()
    {
        var mapping = new MappingBuilder(HardwareTier.Auto).ApplyQuantPreset().Build();

        Assert.NotEmpty(mapping);
        Assert.Equal(new MappingBuilder(HardwareTierHelpers.DetectBestTier()).ApplyQuantPreset().Build(), mapping);
    }

    [Fact]
    public void Arm64_BuildsTheSameTargetsAsScalar_UntilNeonKernelsLand()
    {
        if (!AdvSimd.Arm64.IsSupported) return;

        // The whole point of the first Arm64 change is that it is a no-op: every
        // quant key resolves to its scalar target, so nothing can fail to
        // assemble. When NEON kernels start being registered per key this test is
        // the place that records the change deliberately.
        var arm = new MappingBuilder(HardwareTier.Arm64).ApplyQuantPreset().Build();
        var scalar = new MappingBuilder(HardwareTier.Scalar).ApplyQuantPreset().Build();

        Assert.Equal(Describe(scalar), Describe(arm));
        Assert.All(arm.Values, value => Assert.Contains("scalar", value));
    }

    [Fact]
    public void Arm64_DoesNotUseTheWideQ8_0Path()
    {
        if (!AdvSimd.Arm64.IsSupported) return;

        // The wide path is built on Vector256, which is two emulated Vector128
        // operations on Arm64 - slower than the plain kernel, not faster.
        Assert.False(Q8_0WideWeights.IsSupported(inFeatures: 256, outFeatures: 256));
    }

    private static string[] Describe(Dictionary<string, string> mapping) =>
        [.. mapping.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}")];
}
