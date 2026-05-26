using SharpMind.Model.Format;
using System.Text;

namespace SharpMind.Diagnostic;

public static class GgufDiagnostic
{
    public static void DumpFirstBlock(string ggufPath, string tensorName, int blockIndex = 0)
    {
        var meta = GgufLoader.LoadMeta(ggufPath);

        // Find the tensor
        var info = meta.Tensors.FirstOrDefault(t =>
            t.Name.Contains(tensorName, StringComparison.OrdinalIgnoreCase));

        if (info.Name == null)
        {
            Console.Error.WriteLine($"Tensor '{tensorName}' not found.");
            Console.Error.WriteLine("Available tensors:");
            foreach (var t in meta.Tensors.Take(20))
                Console.Error.WriteLine($"  {t.Name} [{t.Dtype}] shape=[{string.Join(",", t.Shape)}] offset={t.Offset}");
            return;
        }

        Console.WriteLine($"Tensor: {info.Name}");
        Console.WriteLine($"  Dtype: {info.Dtype}");
        Console.WriteLine($"  Shape: [{string.Join(", ", info.Shape)}]");
        Console.WriteLine($"  Offset: {info.Offset}");

        int bytesPerBlock = GgufLoader.GetRawTensorByteCountForBlock(info.Dtype);
        Console.WriteLine($"  Bytes per block: {bytesPerBlock}");

        long dataPos = meta.DataOffset + info.Offset;
        using var fs = new FileStream(ggufPath, FileMode.Open, FileAccess.Read);
        fs.Position = dataPos + blockIndex * bytesPerBlock;

        byte[] blockData = new byte[bytesPerBlock];
        int read = fs.Read(blockData, 0, bytesPerBlock);
        Console.WriteLine($"  Read {read} bytes from file position {fs.Position - read}");

        Console.WriteLine("  Raw hex dump:");
        for (int i = 0; i < Math.Min(128, read); i += 16)
        {
            var hex = Convert.ToHexString(blockData, i, Math.Min(16, read - i));
            var asc = Encoding.ASCII.GetString(blockData, i, Math.Min(16, read - i))
                .Replace('\0', '.').Replace('\n', '.').Replace('\r', '.');
            Console.WriteLine($"    {i:X4}: {hex,-48} {asc}");
        }

        // Dump d and dmin for Q4_K
        if (info.Dtype == GgufDtype.Q4_K)
        {
            var d = BitConverter.ToUInt16(blockData, 0);
            var dmin = BitConverter.ToUInt16(blockData, 2);
            Console.WriteLine($"  d (raw half): 0x{d:X4}");
            Console.WriteLine($"  dmin (raw half): 0x{dmin:X4}");

            // Dump scales
            Console.WriteLine("  Scales (12 bytes):");
            for (int i = 0; i < 12; i++)
                Console.WriteLine($"    scales[{i}] = 0x{blockData[4 + i]:X2} ({blockData[4 + i]})");

            // Dump first 16 qs bytes
            Console.WriteLine("  First 16 qs bytes:");
            for (int i = 0; i < Math.Min(16, read - 16); i++)
                Console.WriteLine($"    qs[{i}] = 0x{blockData[16 + i]:X2}");
        }
        else if (info.Dtype == GgufDtype.Q3_K)
        {
            var d = BitConverter.ToUInt16(blockData, 108);
            Console.WriteLine($"  d (raw half): 0x{d:X4}");

            // Dump first 32 hmask bytes
            Console.WriteLine("  hmask[0..7]:");
            for (int i = 0; i < 8; i++)
                Console.WriteLine($"    hmask[{i}] = 0x{blockData[i]:X2}");

            // Dump scales
            Console.WriteLine("  Scales (12 bytes):");
            for (int i = 0; i < 12; i++)
                Console.WriteLine($"    scales[{i}] = 0x{blockData[96 + i]:X2} ({blockData[96 + i]})");

            // Dump first 16 qs bytes
            Console.WriteLine("  First 16 qs bytes:");
            for (int i = 0; i < 16; i++)
                Console.WriteLine($"    qs[{i}] = 0x{blockData[32 + i]:X2}");
        }
    }
}
