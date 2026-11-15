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
    public void Arm64_UsesNeonForTheKernelsThatExistAndScalarElsewhere()
    {
        if (!AdvSimd.Arm64.IsSupported) return;

        // The deliberate tripwire for NEON coverage: the only keys that differ
        // from Scalar are the ones with an Arm64 kernel. Switching a key that has
        // no registered target fails to assemble, so this list has to grow with
        // each kernel in both places.
        var arm = new MappingBuilder(HardwareTier.Arm64).ApplyQuantPreset().Build();
        var scalar = new MappingBuilder(HardwareTier.Scalar).ApplyQuantPreset().Build();

        Assert.Equal("q4k_arm64", arm["vecdot_q4k"]);
        Assert.Equal("q6k_arm64", arm["vecdot_q6k"]);
        Assert.Equal("q8_0_arm64", arm["vecdot_q8_0"]);
        Assert.Equal("qmatmul_q4k_parallel_arm64", arm["qmatmul_q4k"]);
        Assert.Equal("qmatmul_q6k_parallel_arm64", arm["qmatmul_q6k"]);
        Assert.Equal("qmatmul_q8_0_parallel_arm64", arm["qmatmul_q8_0"]);

        string[] differing =
            [.. arm.Where(kv => scalar[kv.Key] != kv.Value)
                   .Select(kv => kv.Key)
                   .OrderBy(key => key, StringComparer.Ordinal)];
        Assert.Equal(new[]
        {
            "qmatmul_q4k", "qmatmul_q6k", "qmatmul_q8_0",
            "vecdot_q4k", "vecdot_q6k", "vecdot_q8_0",
        }, differing);

        // ...and the mapping assembles, which is the part JigSaw rejects when an
        // option is unregistered rather than falling back to the key's default.
        Assert.NotNull(QuantizationFactory.Create(HardwareTier.Arm64));
    }

    [Fact]
    public void Arm64_DoesNotUseTheWideQ8_0Path()
    {
        if (!AdvSimd.Arm64.IsSupported) return;

        // The wide path is built on Vector256, which is two emulated Vector128
        // operations on Arm64 - slower than the plain kernel, not faster.
        Assert.False(Q8_0WideWeights.IsSupported(inFeatures: 256, outFeatures: 256));
    }
}
