using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

/// <summary>
/// MXFP4 (OCP micro-scaling FP4, GGML type 39). A block is 32 values in 17
/// bytes: one E8M0 exponent, then sixteen bytes of E2M1 nibbles. Nibble
/// <c>j</c> holds element <c>j</c> in its low half and element <c>j + 16</c> in
/// its high half — the same half-split the 32-value Q4 formats use.
/// </summary>
public static partial class QuantizationKernels
{
    /// <summary>
    /// The E2M1 values a nibble indexes: <c>{0, .5, 1, 1.5, 2, 3, 4, 6}</c> for
    /// the low half of the range and their negations for the high half. Every
    /// value is exactly representable in float, so the decode has no rounding.
    /// </summary>
    private static readonly float[] kvalues_mxfp4 =
        [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    public static void ReadMXFP4_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 17;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = E8M0ToFloat_Scalar(buf[0]);
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int nib = (j < half) ? (buf[1 + j] & 0x0F) : (buf[1 + j - half] >> 4);
                data[blockStart + j] = d * kvalues_mxfp4[nib];
            }
        }
    }
}
