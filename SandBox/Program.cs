// See https://aka.ms/new-console-template for more information

using System.Runtime.Intrinsics.X86;

Console.WriteLine($"AVX2 {Avx2.IsSupported}");
Console.WriteLine($"FMA  {Fma.IsSupported}");
Console.In.ReadLine();