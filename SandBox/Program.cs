//await SharpMind.Samples.Examples.QwenOnGpu.RunAsync("hello");
using SharpMind.Tests.Diagnostic;

await SharpMind.Samples.Examples.KnownFailingModels.RunAsync("Hello");// .BuilderOptions.RunAsync("Hello");
Console.In.ReadLine();
