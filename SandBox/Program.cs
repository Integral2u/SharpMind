using SharpMind.Model.Format;

await SharpMind.Samples.Examples.MultiTestInteractive.RunAsync("hello");
return;

string assets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
var dsPath = Path.Combine(assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");

var meta = GgufLoader.LoadMeta(dsPath);
var embInfo = meta.Tensors.First(t => t.Name == "token_embd.weight");

// Read first 100 blocks and dump min/max per block, plus raw d and scales
using (var fs = File.OpenRead(dsPath))
{
    fs.Position = meta.DataOffset + embInfo.Offset;
    using var reader = new BinaryReader(fs);
    var data = new float[256 * 100];
    GgufLoader.ReadQ3_K(reader, data.AsSpan(), data.Length);
    
    double overallSumSq = 0;
    float overallMin = float.MaxValue, overallMax = float.MinValue;
    
    // Go back and read raw d values
    fs.Position = meta.DataOffset + embInfo.Offset;
    
    for (int b = 0; b < 100; b++)
    {
        // Read raw block
        ushort dRaw = reader.ReadUInt16();
        float d = GgufLoader.HalfToFloat(dRaw);
        reader.BaseStream.Position += 108; // skip rest of block (hmask+qs+scales)
        
        int bo = b * 256;
        float bmin = float.MaxValue, bmax = float.MinValue;
        double bsumSq = 0;
        for (int i = 0; i < 256; i++)
        {
            float v = data[bo + i];
            bmin = Math.Min(bmin, v); bmax = Math.Max(bmax, v);
            bsumSq += v * v;
            overallMin = Math.Min(overallMin, v); overallMax = Math.Max(overallMax, v);
        }
        overallSumSq += bsumSq;
        if (d > 10 || Math.Abs(bmax) > 1000 || Math.Abs(bmin) > 1000)
            Console.WriteLine($"  Block {b,3}: d={d,10:G6} min={bmin,12:G6} max={bmax,12:G6} norm={Math.Sqrt(bsumSq),10:G6}");
    }
    Console.WriteLine($"Overall 100 blocks ({data.Length} elems): min={overallMin:G6} max={overallMax:G6} norm={Math.Sqrt(overallSumSq):G6}");
    Console.Write("First 20: ");
    for (int i = 0; i < 20; i++) Console.Write($"{data[i]:G6} ");
    Console.WriteLine();
}
