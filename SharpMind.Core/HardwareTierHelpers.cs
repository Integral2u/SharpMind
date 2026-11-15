using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core;

public static class HardwareTierHelpers
{
    /// <summary>
    /// The fastest tier this process can run. Arm64 sits below AVX2 and above
    /// SSE: NEON is 128 bits per vector against AVX2's 256, but it is still far
    /// ahead of the scalar path an SSE-only host would otherwise get. The x86
    /// intrinsics report <see langword="false"/> on Arm64 and vice versa, so the
    /// two families never compete at runtime.
    /// </summary>
    public static HardwareTier DetectBestTier()
    {
        if (Fma.IsSupported) return HardwareTier.FMA;
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (AdvSimd.Arm64.IsSupported) return HardwareTier.Arm64;
        if (Sse.IsSupported) return HardwareTier.SSE;
        return HardwareTier.Scalar;
    }

    /// <summary>Whether this process can execute <paramref name="tier"/>'s kernels.</summary>
    public static bool IsSupported(HardwareTier tier) => tier switch
    {
        HardwareTier.Auto or HardwareTier.Scalar => true,
        HardwareTier.SSE => Sse.IsSupported,
        HardwareTier.AVX2 => Avx2.IsSupported,
        HardwareTier.FMA => Fma.IsSupported,
        HardwareTier.Arm64 => AdvSimd.Arm64.IsSupported,
        _ => false,
    };

    /// <summary>
    /// The tiers this process can run, in the order they should be presented:
    /// <see cref="HardwareTier.Auto"/> first, then fastest to slowest. This is
    /// the single list the options screen renders, because a radio group is an
    /// index into it and offering a tier the CPU cannot run would fault at
    /// launch. It is computed once, since instruction-set support never changes
    /// within a process.
    /// </summary>
    public static IReadOnlyList<HardwareTier> SelectableTiers { get; } =
    [
        HardwareTier.Auto,
        .. new[]
        {
            HardwareTier.FMA, HardwareTier.AVX2, HardwareTier.Arm64,
            HardwareTier.SSE, HardwareTier.Scalar,
        }.Where(IsSupported),
    ];

    /// <summary>
    /// Throws for a tier this process physically cannot execute. Forcing a tier
    /// is a benchmarking and diagnostic escape hatch, and the wrong one is an
    /// easy mistake to make (a saved preset authored on Arm64, a
    /// <c>kbench</c> argument, a test that forgot a guard). Without this check
    /// the first kernel call dies as an illegal instruction or a JIT
    /// <see cref="PlatformNotSupportedException"/> with no clue where it came from.
    /// </summary>
    /// <exception cref="PlatformNotSupportedException">The CPU lacks the tier's instruction set.</exception>
    public static HardwareTier EnsureUsable(HardwareTier tier)
    {
        if (tier != HardwareTier.Auto && !IsSupported(tier))
            throw new PlatformNotSupportedException(
                $"HardwareTier.{tier} is not supported by this process " +
                $"({RuntimeInformation.ProcessArchitecture}). The fastest tier here is " +
                $"{DetectBestTier()}; use HardwareTier.Auto, or a tier this CPU implements.");
        return tier;
    }
}
