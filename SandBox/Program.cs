// See https://aka.ms/new-console-template for more information

using SharpMind.GPU;
using System.Runtime.Intrinsics.X86;

Console.WriteLine($"AVX2 {Avx2.IsSupported}");
Console.WriteLine($"FMA  {Fma.IsSupported}");
Console.WriteLine(Enum.GetName(GPUSharpMindConfig.BestBackend));
Console.In.ReadLine();