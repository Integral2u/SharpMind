// See https://aka.ms/new-console-template for more information

using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.GPU;
using System.Runtime.Intrinsics.X86;

Console.WriteLine($"AVX2 {Avx2.IsSupported}");
Console.WriteLine($"FMA  {Fma.IsSupported}");
Console.WriteLine(Enum.GetName(GPUSharpMindConfig.BestBackend));

var config = new VocabConfig { VocabSize = 500 };
var gen = new PseudoLanguageGenerator(config);
var rec = gen.GetModelSizeRecommendation();

Console.WriteLine($"Vocab: {rec.VocabSize}");
Console.WriteLine($"Embedding: {rec.EmbeddingDim}");
Console.WriteLine($"Layers: {rec.NumLayers}");
Console.WriteLine($"Est. Params: {rec.EstimatedParams:N0}");

Console.In.ReadLine();