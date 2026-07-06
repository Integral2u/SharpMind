using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Tests.Quantization;

public class QuantizedMatMulRefTests
{
    private const string ExternalAssets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
    private const string Fp16File = "qwen2-0_5b-instruct-fp16.gguf";
    private const string TensorName = "blk.0.attn_q.weight";

    private static readonly Lazy<GgufLoader> _loader = new(() => GgufLoaderFactory.Create());

    private static bool TryGetFp16File(out string path)
    {
        path = Path.Combine(ExternalAssets, Fp16File);
        return File.Exists(path);
    }

    private static TensorInfo? FindTensor(GgufLoader loader, string path)
    {
        var meta = loader.LoadMeta(path);
        return meta.Tensors.FirstOrDefault(t => t.Name == TensorName);
    }

    private static unsafe float[] RunFloatForward(GgufLoader loader, string path, TensorInfo info, Tensor<float> input)
    {
        using var ggufStream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(ggufStream);
        var meta = loader.LoadMeta(path);
        ggufStream.Position = meta.DataOffset + info.Offset;
        int count = info.Shape[0] * info.Shape[1];
        var buf = new float[count];
        loader.ReadTensorInto(reader, info.Dtype, info.Shape, buf.AsSpan());

        // GGUF stores weight in transposed layout [outFeatures, inFeatures].
        // MatMulWithBT expects bt in [N, K] layout, so the GGUF buffer
        // can be used directly as bt without further transposition.
        int inF = info.Shape[0], outF = info.Shape[1];
        var bt = Tensor<float>.From(buf.AsSpan(), outF, inF);
        var ops = TensorOpsFactory.Create(global::SharpMind.SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar });
        using var output = ops.MatMulWithBT(input, bt);
        return output.Data.ToArray();
    }

    [Fact]
    public void Fp16_QuantizedMatMul_Matches_Float32_Forward()
    {
        if (!TryGetFp16File(out var path))
        {
            Assert.True(true, $"SKIP: {Fp16File} not found");
            return;
        }

        var loader = _loader.Value;
        var info = FindTensor(loader, path);
        Assert.NotNull(info);

        int inF = info.Value.Shape[0];
        int outF = info.Value.Shape[1];

        using var input = new Tensor<float>(1, inF);
        var rng = new Random(42);
        for (int i = 0; i < input.ElementCount; i++)
            input.Data[i] = (float)(rng.NextDouble() * 2 - 1);

        var floatResult = RunFloatForward(loader, path, info.Value, input);

        using var ggufStream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(ggufStream);
        var meta = loader.LoadMeta(path);
        ggufStream.Position = meta.DataOffset + info.Value.Offset;
        long rawSize = loader.GetRawTensorByteCount(info.Value.Shape, info.Value.Dtype);
        var rawData = new byte[rawSize];
        reader.ReadExactly(rawData);

        var qOps = QuantizationFactory.Create();
        using var qResult = new Tensor<float>(1, outF);
        unsafe
        {
            fixed (byte* pRaw = rawData)
            {
                qOps.QuantizedMatMulF16(input.DataPtr, pRaw, qResult.DataPtr, 1, inF, outF);
            }
        }

        double maxDiff = 0;
        for (int i = 0; i < outF; i++)
        {
            double diff = Math.Abs(floatResult[i] - qResult.Data[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        Assert.True(maxDiff < 0.02,
            $"F16 QuantizedMatMul deviates from float32 reference. MaxDiff: {maxDiff:F6}. " +
            "If this fails, the true float32 output needs human verification.");
    }

    [Fact]
    public void Fp16_QuantizedMatMul_MultiBatch_Matches_Float32()
    {
        if (!TryGetFp16File(out var path))
        {
            Assert.True(true, $"SKIP: {Fp16File} not found");
            return;
        }

        var loader = _loader.Value;
        var info = FindTensor(loader, path);
        Assert.NotNull(info);

        int inF = info.Value.Shape[0];
        int outF = info.Value.Shape[1];
        const int M = 3;

        using var input = new Tensor<float>(M, inF);
        var rng = new Random(42);
        for (int i = 0; i < input.ElementCount; i++)
            input.Data[i] = (float)(rng.NextDouble() * 2 - 1);

        using var ggufStream = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(ggufStream);
        var meta = loader.LoadMeta(path);
        ggufStream.Position = meta.DataOffset + info.Value.Offset;
        long rawSize = loader.GetRawTensorByteCount(info.Value.Shape, info.Value.Dtype);
        var rawData = new byte[rawSize];
        reader.ReadExactly(rawData);

        var qOps = QuantizationFactory.Create();
        using var qResult = new Tensor<float>(M, outF);
        unsafe
        {
            fixed (byte* pRaw = rawData)
            {
                qOps.QuantizedMatMulF16(input.DataPtr, pRaw, qResult.DataPtr, M, inF, outF);
            }
        }

        for (int row = 0; row < M; row++)
        {
            var rowData = new float[inF];
            input.Data.Slice(row * inF, inF).CopyTo(rowData);
            using var rowInput = Tensor<float>.From(rowData.AsSpan(), 1, inF);
            var floatResult = RunFloatForward(loader, path, info.Value, rowInput);

            double maxDiff = 0;
            for (int i = 0; i < outF; i++)
            {
                double diff = Math.Abs(floatResult[i] - qResult.Data[row * outF + i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            Assert.True(maxDiff < 0.02,
                $"Row {row} deviates. MaxDiff: {maxDiff:F6}");
        }
    }
}
