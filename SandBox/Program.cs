using SandBox;
using SandBox.RunModels;
await SharpMind.Samples.Examples.BuilderOptions.RunAsync("Hello", @"C:\Integral2u\source\repos\SharpMind\ExternalAssets", "qwen2-0_5b-instruct-q8_0");//  KnownWorkingModels.RunAsync("Hello");
//VecDotQ4KDiagnostic.Run();
Console.In.ReadLine();
return;