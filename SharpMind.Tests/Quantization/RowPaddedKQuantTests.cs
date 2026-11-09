using SharpMind.Core;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The flat K-quant kernels index blocks by <c>col * K / blockSize</c>. A GGUF
/// tensor whose row length is not 256-aligned is row-padded instead; feeding the
/// kernel the padded K and a zero-padded input row must produce the same result
/// as the row-padded blocks. This is the core of the 896-wide Qwen2 case.
/// </summary>
public sealed class RowPaddedKQuantTests
{
    [Fact]
    public unsafe void Q4KRowPaddedMatMulMatchesTheDequantizedReference()
    {
        const int rowLength = 896;
        const int paddedRow = 1024;
        const int outFeatures = 4;

        var random = new Random(1234);
        var weight = new float[rowLength, outFeatures];
        for (int i = 0; i < rowLength; i++)
            for (int o = 0; o < outFeatures; o++)
                weight[i, o] = (float)(random.NextDouble() * 2 - 1);

        var raw = new byte[outFeatures * 576]; // 4 blocks * 144 bytes per row
        var paddedRowValues = new float[paddedRow];
        for (int o = 0; o < outFeatures; o++)
        {
            Array.Clear(paddedRowValues);
            for (int i = 0; i < rowLength; i++)
                paddedRowValues[i] = weight[i, o];
            byte[] encoded = TensorQuantizer.Quantize(
                paddedRowValues, [paddedRow], QuantDType.Q4_K);
            Assert.Equal(576, encoded.Length);
            encoded.CopyTo(raw, o * 576);
        }

        var input = new float[rowLength];
        for (int i = 0; i < rowLength; i++)
            input[i] = (float)(random.NextDouble() * 2 - 1);
        var paddedInput = new float[paddedRow];
        input.CopyTo(paddedInput, 0);

        var actual = new float[outFeatures];
        fixed (float* pInput = paddedInput)
        fixed (byte* pRaw = raw)
        fixed (float* pOutput = actual)
        {
            QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar(
                pInput, pRaw, pOutput, 1, paddedRow, outFeatures);
        }

        QuantizationOps ops = QuantizationFactory.Create(HardwareTier.Scalar);
        var dequantized = new float[paddedRow];
        for (int o = 0; o < outFeatures; o++)
        {
            using var memory = new MemoryStream(raw, o * 576, 576, writable: false);
            using var reader = new BinaryReader(memory);
            ops.ReadFor(QuantDType.Q4_K, reader, dequantized, paddedRow);
            double expected = 0;
            for (int i = 0; i < paddedRow; i++)
                expected += dequantized[i] * paddedInput[i];
            Assert.InRange(Math.Abs(actual[o] - expected), 0, 1e-3);
        }
    }
}
