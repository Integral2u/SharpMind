using LLama;
using SandBox;
using SandBox.RunModels;
//await SharpMind.Samples.Examples.BuilderOptions.RunAsync("Hello", @"C:\Integral2u\source\repos\SharpMind\ExternalAssets", "qwen2-0_5b-instruct-q8_0");
Console.WriteLine("await KnownWorkingModels.RunAsync(\"Hello\");---------------------");
await KnownWorkingModels.RunAsync("Hello");
Console.WriteLine("await KnownGiberishModels.RunAsync(\"Hello\");---------------------");
await KnownGiberishModels.RunAsync("Hello");
Console.WriteLine("await KnownFailingModels.RunAsync(\"Hello\");---------------------");
await KnownFailingModels.RunAsync("Hello");

Console.In.ReadLine();
return;