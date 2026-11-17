using System;
using SharpMind.Core.Embeddings;
using Xunit;

namespace SharpMind.Tests.Core;

/// <summary>
/// YaRN is what makes a DeepSeek-V2 file usable at all: `factor = 40` scales
/// every rotary angle by 1/40 unless the dimension is in the extrapolated band,
/// and the tables carry an "attention factor" on top. Getting either wrong does
/// not crash anything - it produces a model that is fluent but cannot use
/// position.
///
/// The expected values here come from reimplementing ggml's loop verbatim
/// (<c>ggml_rope_cache_init</c> / <c>rope_yarn</c> in ggml-cpu/ops.cpp): walk the
/// pair index, blend interpolated and extrapolated angles across the correction
/// band, scale cos and sin by mscale. RoPE does the same thing folded into a
/// per-dimension frequency, so agreement means the fold is right.
/// </summary>
public class YarnRopeTests
{
    private const float Theta = 10_000f;
    private const int RopeDim = 64;
    private const float Factor = 40f;
    private const int OriginalContext = 4096;
    private const float LogMultiplier = 0.0707f;

    /// <summary>ggml's reference: one cos/sin pair per (position, pair index).</summary>
    private static (float[] cos, float[] sin) Reference(int maxSeqLen)
    {
        const float betaFast = 32f, betaSlow = 1f;
        int nDims = RopeDim;
        float low = MathF.Floor(nDims * MathF.Log(OriginalContext / (betaFast * 2f * MathF.PI)) / (2f * MathF.Log(Theta)));
        float high = MathF.Ceiling(nDims * MathF.Log(OriginalContext / (betaSlow * 2f * MathF.PI)) / (2f * MathF.Log(Theta)));

        float factor = Factor;
        float freqScale = 1f / factor;
        static float MS(float scale, float m) => scale <= 1f ? 1f : 0.1f * m * MathF.Log(scale) + 1f;
        // llama-context.cpp: the ratio, then the 1/(1 + 0.1 ln f) cancellation...
        float mscale = MS(factor, LogMultiplier) / MS(factor, LogMultiplier);
        mscale *= 1f / (1f + 0.1f * MathF.Log(factor));
        // ...which ggml's rope_yarn undoes whenever ext_factor != 0
        // (mscale *= 1 + 0.1 ln(1/freq_scale)). The tables therefore carry the
        // plain ratio; MLA pre-scales kq_scale by the mscale² instead.
        mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);

        int halfDim = RopeDim / 2;
        float thetaScale = MathF.Pow(Theta, -2f / RopeDim);
        var cos = new float[maxSeqLen * halfDim];
        var sin = new float[maxSeqLen * halfDim];

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            float theta = pos;
            for (int i = 0; i < halfDim; i++)
            {
                float thetaExtrap = theta;
                float thetaInterp = freqScale * thetaExtrap;
                float blended = thetaInterp;
                float y = (i - low) / MathF.Max(0.001f, high - low);
                float ramp = 1f - MathF.Min(1f, MathF.Max(0f, y));
                blended = thetaInterp * (1f - ramp) + thetaExtrap * ramp;

                cos[pos * halfDim + i] = MathF.Cos(blended) * mscale;
                sin[pos * halfDim + i] = MathF.Sin(blended) * mscale;
                theta *= thetaScale;
            }
        }
        return (cos, sin);
    }

    [Fact]
    public void CosSinTables_MatchGgmlsYarnLoop()
    {
        const int maxSeqLen = 64;
        var rope = new RoPE(RopeDim, maxSeqLen, Theta, ropeDim: RopeDim, ropeScalingType: "yarn",
            ropeScalingFactor: Factor, ropeOriginalContextLength: OriginalContext,
            yarnLogMultiplier: LogMultiplier);
        var (cos, sin) = Reference(maxSeqLen);

        Assert.Equal(cos.Length, rope.CosTable.Length);
        double worst = 0;
        for (int k = 0; k < cos.Length; k++)
        {
            worst = Math.Max(worst, Math.Abs(cos[k] - rope.CosTable[k]));
            worst = Math.Max(worst, Math.Abs(sin[k] - rope.SinTable[k]));
        }
        Assert.True(worst < 1e-5, $"worst table difference {worst:E3}");
    }

    [Fact]
    public void YarnCompressesEveryAngle_ExceptTheExtrapolatedBand()
    {
        // Pair 0 sits in the extrapolated band, so its angle is untouched: the
        // table is the plain rotation times the YaRN ratio (1 for DeepSeek-V2,
        // where mscale == mscale_all_dim). The low-frequency pairs are
        // compressed, which is the whole point of YaRN.
        const int maxSeqLen = 2;
        var yarn = new RoPE(RopeDim, maxSeqLen, Theta, ropeDim: RopeDim, ropeScalingType: "yarn",
            ropeScalingFactor: Factor, ropeOriginalContextLength: OriginalContext,
            yarnLogMultiplier: LogMultiplier);
        var plain = new RoPE(RopeDim, maxSeqLen, Theta, ropeDim: RopeDim);
        float mscale = RoPE.YarnMscale(Factor, LogMultiplier);
        Assert.Equal(1f, mscale, 6);

        // pos 1, pair 0: ramp == 1 -> the plain rotation, times mscale.
        Assert.Equal(plain.CosTable[32] * mscale, yarn.CosTable[32], 6);
        Assert.Equal(plain.SinTable[32] * mscale, yarn.SinTable[32], 6);

        // pos 1, last pair: fully interpolated, so the angle shrinks by 1/40 and
        // the rotation is nowhere near the plain one.
        float yarnAngle = MathF.Atan2(yarn.SinTable[63], yarn.CosTable[63]);
        float plainAngle = MathF.Atan2(plain.SinTable[63], plain.CosTable[63]);
        Assert.Equal(plainAngle / Factor, yarnAngle, 7);
    }

    [Fact]
    public void WithoutYarnTheTablesAreThePlainRope()
    {
        var plain = new RoPE(RopeDim, 8, Theta, ropeDim: RopeDim);
        for (int k = 0; k < plain.CosTable.Length; k++)
        {
            // No mscale: every entry is a pure rotation, |cos| + |sin| > 0 and
            // cos^2 + sin^2 == 1.
            float c = plain.CosTable[k], s = plain.SinTable[k];
            Assert.Equal(1f, c * c + s * s, 5);
        }
    }
}
