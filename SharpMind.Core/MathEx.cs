using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core;

/// <summary>
/// Canonical scalar and SIMD fast-math helpers used across the activation, gradient
/// and autograd kernels. Before this class existed the degree-6 exp polynomial was
/// re-implemented three times (two of them byte-identical) and the GELU constants in
/// four files; the helpers here are the single source of truth. All entries keep the
/// exact arithmetic of the originals so outputs are unchanged.
/// </summary>
public static class MathEx
{
    // GELU tanh-approximation constants (0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))).
    public const float SqrtTwoPiInv = 0.7978845608f;
    public const float GeluCoeff = 0.044715f;

    /// <summary>exp(x) via range-reduced degree-6 polynomial, ≈5 ULP over [-88, 88].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastExp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-88.0f)), Vector256.Create(88.0f));

        // exp(x) = 2^(x * log2(e))
        var z = Avx.Multiply(x, Vector256.Create(1.4426950408889634f));

        // Round z to nearest int via magic-bias trick
        var magic = Vector256.Create(12582912.0f);
        var nF = Avx.Subtract(Avx.Add(z, magic), magic);
        var nI = Avx2.ConvertToVector256Int32(nF);

        // r = z - n  in [-0.5, 0.5]
        var r = Avx.Subtract(z, nF);

        // u = r * ln(2)  in [-0.35, 0.35]
        var u = Avx.Multiply(r, Vector256.Create(0.6931471805599453f));

        // exp(u) Horner degree-6 — error << 1 ULP on this domain
        var p = Avx.Add(Vector256.Create(1.0f),
            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f),
                Avx.Multiply(u, Avx.Add(Vector256.Create(0.5f),
                    Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 6.0f),
                        Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 24.0f),
                            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 120.0f),
                                Avx.Multiply(u, Vector256.Create(1.0f / 720.0f))
                            ))
                        ))
                    ))
                ))
            ))
        );

        // Multiply by 2^n: build 2^n as a float, then multiply
        var expAdj = Avx2.Add(nI, Vector256.Create(127));
        expAdj = Avx2.Min(Avx2.Max(expAdj, Vector256.Create(0)), Vector256.Create(254));
        var pow2nBits = Avx2.ShiftLeftLogical(expAdj, 23);
        return Avx.Multiply(p, Vector256.AsSingle(pow2nBits));
    }

    /// <summary>tanh(z) = (exp(2z) - 1) / (exp(2z) + 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> FastTanh(Vector256<float> z)
    {
        z = Avx.Min(Avx.Max(z, Vector256.Create(-9.0f)), Vector256.Create(9.0f));
        var twoZ = Avx.Multiply(z, Vector256.Create(2.0f));
        var e2z = FastExp(twoZ);
        var one = Vector256.Create(1.0f);
        return Avx.Divide(Avx.Subtract(e2z, one), Avx.Add(e2z, one));
    }

    /// <summary>
    /// Scalar exp(x) via the same degree-6 polynomial as <see cref="FastExp"/>, with the
    /// magic-bias step replaced by <see cref="MathF.Round"/> and the 2^n scale by
    /// <see cref="MathF.Pow"/>. Used for softmax tail lanes when the SIMD path is absent.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExpScalar(float x)
    {
        x = Math.Clamp(x, -88f, 88f);
        var z = x * 1.4426950408889634f;
        var n = MathF.Round(z);
        var r = z - n;
        var u = r * 0.6931471805599453f;
        var p = 1f + u * (1f + u * (0.5f + u * ((1f / 6f) + u * ((1f / 24f) + u * ((1f / 120f) + u * (1f / 720f))))));
        return p * MathF.Pow(2f, n);
    }

    /// <summary>Sigmoid: 1 / (1 + exp(-v)).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    /// <summary>GELU tanh approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x) => 0.5f * x * (1f + MathF.Tanh(SqrtTwoPiInv * (x + GeluCoeff * x * x * x)));

    /// <summary>Derivative of the GELU tanh approximation (the d/dx factor a caller multiplies by its gradient).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float GeluDerivative(float x)
    {
        float x3 = x * x * x;
        float inner = SqrtTwoPiInv * (x + GeluCoeff * x3);
        float tanh = MathF.Tanh(inner);
        float dtanh = 1f - tanh * tanh;
        float dInner = SqrtTwoPiInv * (1f + 3f * GeluCoeff * x * x);
        return 0.5f * (1f + tanh) + 0.5f * x * dtanh * dInner;
    }
}