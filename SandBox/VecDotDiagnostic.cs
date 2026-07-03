using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using System.IO.MemoryMappedFiles;

namespace SandBox;

public static class VecDotDiagnostic
{
    private static (int blockSize, int bytesPerBlock) GetBlockInfo(GgufDtype dtype) => dtype switch
    {
        GgufDtype.Q5_0 => (32, 22),
        GgufDtype.Q4_0 => (32, 18),
        GgufDtype.Q4_1 => (32, 20),
        GgufDtype.Q5_1 => (32, 24),
        GgufDtype.Q8_0 => (32, 34),
        GgufDtype.Q8_1 => (32, 36),
        GgufDtype.IQ4_NL => (32, 18),
        GgufDtype.Q2_K or GgufDtype.Q2_K_S => (256, 84),
        GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => (256, 110),
        GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => (256, 144),
        GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => (256, 176),
        GgufDtype.Q6_K or GgufDtype.Q6_K_S => (256, 210),
        GgufDtype.Q8_K => (256, 292),
        _ => (0, 0)
    };

    private static long GetRawTensorByteCount(int[] shape, GgufDtype dtype)
    {
        var (blockSize, bytesPerBlock) = GetBlockInfo(dtype);
        if (blockSize == 0) return 0;
        long totalElements = 1;
        foreach (int d in shape) totalElements *= d;
        long nBlocks = (totalElements + blockSize - 1) / blockSize;
        return nBlocks * bytesPerBlock;
    }

    public static unsafe void Run()
    {
        string ggufPath = Path.Combine(
            @"C:\Integral2u\source\repos\SharpMind\ExternalAssets",
            "SmolLM2-135M-Instruct.Q4_K_M.gguf");

        if (!File.Exists(ggufPath))
        {
            Console.Error.WriteLine($"File not found: {ggufPath}");
            return;
        }

        var meta = GgufLoader.LoadMeta(ggufPath);
        var config = GgufLoader.LoadConfig(meta);
        if (config == null) { Console.Error.WriteLine("Failed to load config"); return; }

        int hiddenDim = config.HiddenDim;
        int ffnDim = config.FfnDim;
        Console.Error.WriteLine($"Config: HiddenDim={hiddenDim} FfnDim={ffnDim} NumLayers={config.NumLayers}");

        // --- Find ffn_gate.weight for layer 0 ---
        var gateInfo = meta.Tensors.FirstOrDefault(t =>
            t.Name.StartsWith("blk.0.ffn_gate", StringComparison.OrdinalIgnoreCase));
        if (gateInfo.Name == null) { Console.Error.WriteLine("blk.0.ffn_gate.weight not found!"); return; }

        int ggufRows = gateInfo.Shape[0];
        int ggufCols = gateInfo.Shape[1];
        bool hasTransposedShape = (ggufRows == hiddenDim && ggufCols == ffnDim);

        Console.Error.WriteLine($"\nTensor: {gateInfo.Name}");
        Console.Error.WriteLine($"  Shape: [{string.Join(",", gateInfo.Shape)}]");
        Console.Error.WriteLine($"  Dtype: {gateInfo.Dtype} ({(uint)gateInfo.Dtype})");
        Console.Error.WriteLine($"  Interpretation: GGUF stores [{ggufRows}, {ggufCols}]");
        Console.Error.WriteLine($"  hiddenDim={hiddenDim}, ffnDim={ffnDim}");
        Console.Error.WriteLine($"  This is [InDim={ggufRows}, OutDim={ggufCols}] (@@ hasTransposedShape={hasTransposedShape})");
        Console.Error.WriteLine($"  Standard GGUF would store [OutDim, InDim] = [{ffnDim}, {hiddenDim}]");

        // --- Q5_0 block layout analysis ---
        const int Q5_BLOCK_BYTES = 22;
        int nBlocksQ5 = (hiddenDim + 31) / 32; // ceil(hiddenDim / 32) = 18
        int ggufRowBlocks = (ggufCols + 31) / 32; // blocks per GGUF row = ceil(shape[1] / 32) = 48
        long totalElements = (long)ggufRows * ggufCols;
        long totalQ5Blocks = (totalElements + 31) / 32;

        Console.Error.WriteLine($"\n  === Q5_0 block layout analysis ===");
        Console.Error.WriteLine($"  nBlocks per VecDot column (ceil(hiddenDim/32)):      {nBlocksQ5}");
        Console.Error.WriteLine($"  Blocks per GGUF row (ceil(shape[1]/32)):            {ggufRowBlocks}");
        Console.Error.WriteLine($"  Total Q5_0 blocks in tensor:                        {totalQ5Blocks}");
        Console.Error.WriteLine($"  Q5_0 col stride bytes:                              {nBlocksQ5 * Q5_BLOCK_BYTES}");
        Console.Error.WriteLine($"  GGUF row stride bytes:                              {ggufRowBlocks * Q5_BLOCK_BYTES}");
        Console.Error.WriteLine($"  Column strides match -> VecDot indexing IS correct: {nBlocksQ5 == ggufRowBlocks}");
        Console.Error.WriteLine($"  CRITICAL: VecDot stride ({nBlocksQ5} blks) != GGUF row stride ({ggufRowBlocks} blks)");

        // --- Read raw data and float dequant ---
        using var mmf = MemoryMappedFile.CreateFromFile(ggufPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var stream = mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read);
        using var reader = new BinaryReader(stream);

        long dataPos = meta.DataOffset + gateInfo.Offset;
        stream.Position = dataPos;
        long actualRawSize = GetRawTensorByteCount(gateInfo.Shape, gateInfo.Dtype);
        byte[] rawData = new byte[actualRawSize];
        stream.ReadExactly(rawData);

        var allWeights = GgufLoader.LoadWeights(ggufPath);
        if (!allWeights.TryGetValue(gateInfo.Name, out var gateFloat))
        {
            Console.Error.WriteLine("Failed to dequantize ffn_gate.weight");
            return;
        }
        var dequant = gateFloat.Data;

        // --- Create random test input ---
        Random rng = new(42);
        float[] input = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        Console.Error.WriteLine($"\n  === Dot product comparisons with RANDOM input ===\n");

        // Test columns 0, 1, 576, 1024
        int[] testCols = [0, 1, 576, 1024, 1535];
        fixed (float* pInput = input)
        fixed (byte* pRaw = rawData)
        {
            foreach (int col in testCols)
            {
                if (col >= ggufCols) continue;

                // VecDot reads: blocks at offsets [col * nBlocksQ5 * Q5_BLOCK_BYTES, ...]
                // These are 18 contiguous blocks (576 elements) starting at block col*18
                int vecDotBlockStart = col * nBlocksQ5;
                int vecDotFlatStart = vecDotBlockStart * 32;
                // VecDot computes: sum input[i] * dequantBlock[vecDotBlockStart * 32 + i] for i in [0, 576)
                // where dequantBlock is the raw Q5_0 dequantized data (same as GGUF flat element order)

                // Correct column col in GGUF [InDim, OutDim] layout:
                // weight[i, col] = dequant[i * OutDim + col] for i in [0, InDim)
                // Float dot = sum input[i] * dequant[i * ggufCols + col]
                double correctFloatDot = 0;
                for (int i = 0; i < hiddenDim; i++)
                    correctFloatDot += input[i] * dequant[i * ggufCols + col];

                // What VecDot actually reads & computes:
                // VecDot reads elements [vecDotFlatStart .. vecDotFlatStart + 575]
                // These are NOT the same as the correct column elements
                double vecDotCrunch = QuantizationKernels.VecDotQ5_0_Scalar(pInput, pRaw, col, hiddenDim);

                // What VecDot *would* read if the strides matched (= what it DOES read, reinterpreted):
                int vecDotBlock0 = vecDotBlockStart;
                Console.Error.WriteLine($"  Col {col}: block stride test");
                Console.Error.WriteLine($"    VecDot reads contiguous blocks [{vecDotBlock0}..{vecDotBlock0 + nBlocksQ5 - 1}]");
                Console.Error.WriteLine($"    = GGUF flat elements [{vecDotFlatStart}..{vecDotFlatStart + hiddenDim - 1}]");
                Console.Error.WriteLine($"    Correct col elements at GGUF flat indices: {col}, {ggufCols + col}, {2 * ggufCols + col}, ...");
                Console.Error.WriteLine($"    Correct float dot: {correctFloatDot:F12}");
                Console.Error.WriteLine($"    VecDotQ5_0 value:  {vecDotCrunch:F12}");
                Console.Error.WriteLine($"    MATCH:             {(Math.Abs(correctFloatDot - vecDotCrunch) < 1e-4f ? "YES (correct indexing)" : "NO (indexing MISMATCH)")}");
                Console.Error.WriteLine();
            }
        }

        // --- Fused buffer simulation ---
        var upInfo = meta.Tensors.FirstOrDefault(t =>
            t.Name.StartsWith("blk.0.ffn_up", StringComparison.OrdinalIgnoreCase));

        if (upInfo.Name != null)
        {
            long upRawSize = GetRawTensorByteCount(upInfo.Shape, upInfo.Dtype);
            long expectedFusedBytes = 2L * ffnDim * nBlocksQ5 * Q5_BLOCK_BYTES;
            long gateRawSize = actualRawSize;
            long concatBytes = gateRawSize + upRawSize;

            Console.Error.WriteLine($"  === Fused buffer simulation (gate+up) ===");
            Console.Error.WriteLine($"  ffn_up shape=[{string.Join(",", upInfo.Shape)}] dtype={upInfo.Dtype}");
            Console.Error.WriteLine($"  ffn_up raw bytes: {upRawSize}");
            Console.Error.WriteLine($"  ffn_gate raw bytes: {gateRawSize}");
            Console.Error.WriteLine($"  Fused (concat) bytes: {concatBytes}");
            Console.Error.WriteLine($"  VecDot expects ({2 * ffnDim} cols * {nBlocksQ5} blocks/col * {Q5_BLOCK_BYTES}B): {expectedFusedBytes}");
            Console.Error.WriteLine($"  Simple concat matches VecDot expectation: {concatBytes == expectedFusedBytes}");
            Console.Error.WriteLine($"  (Fused layout is consistent with VecDot indexing even though each");
            Console.Error.WriteLine($"   individual tensor's data is in [InDim,OutDim] row-major order —");
            Console.Error.WriteLine($"   the column stride only depends on hiddenDim, not on ffnDim)");
        }

        // --- Summary ---
        Console.Error.WriteLine($"\n  === SUMMARY ===");
        Console.Error.WriteLine($"  GGUF shape is [{ggufRows}, {ggufCols}] = [InDim, OutDim] = [{hiddenDim}, {ffnDim}]");
        Console.Error.WriteLine($"  VecDot nBlocks = ceil(hiddenDim/32) = {nBlocksQ5}");
        Console.Error.WriteLine($"  GGUF row blocks = ceil(OutDim/32) = {ggufRowBlocks}");
        bool stridesMatch = nBlocksQ5 == ggufRowBlocks;
        Console.Error.WriteLine($"  The two strides {(stridesMatch ? "MATCH" : "DIFFER")} — indexing is {(stridesMatch ? "CORRECT" : "INCORRECT")}");
        Console.Error.WriteLine($"  This means VecDotQ5_0 reads the wrong elements for all columns except");
        Console.Error.WriteLine($"  those where the contiguous block group happens to align with the");
        Console.Error.WriteLine($"  correct scattered column data (which is essentially never for non-trivial shapes).");
        Console.Error.WriteLine($"  The Q5_0 quantized forward path needs a layout transpose/repack step");

        foreach (var kv in allWeights) kv.Value.Dispose();
        Console.Error.WriteLine("\n  Done.");
    }
}
