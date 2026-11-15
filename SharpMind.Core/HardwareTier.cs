namespace SharpMind.Core;

/// <summary>
/// The kernel tier a mapping is built for. These are hardware families rather
/// than a linear ranking: an Arm64 tier and an x86 tier can never be selected
/// in the same process, because <c>Avx2</c>/<c>Fma</c> report
/// <see langword="false"/> on Arm64 and <c>AdvSimd</c> reports
/// <see langword="false"/> on x86.
/// </summary>
/// <remarks>
/// The numeric values are frozen: saved sessions and presets persist this enum
/// as a number, so inserted members would silently change what an existing file
/// means. Order of preference lives in
/// <see cref="HardwareTierHelpers.DetectBestTier"/>: FMA, AVX2, Arm64, SSE,
/// Scalar.
/// </remarks>
public enum HardwareTier
{
    /// <summary>Use the fastest tier this process can run.</summary>
    Auto = 0,

    /// <summary>x86-64 with AVX2 and FMA, the widest x86 vector path.</summary>
    FMA = 1,

    /// <summary>x86-64 with AVX2 but no FMA.</summary>
    AVX2 = 2,

    /// <summary>x86-64 with SSE only. Ranks below <see cref="Arm64"/>, which has the same 128-bit width but a cleaner instruction set.</summary>
    SSE = 3,

    /// <summary>Portable scalar kernels; available on every platform.</summary>
    Scalar = 4,

    /// <summary>Arm64 NEON (AdvSimd). No dedicated kernels ship yet, so it currently maps to the scalar targets.</summary>
    Arm64 = 5,
}
