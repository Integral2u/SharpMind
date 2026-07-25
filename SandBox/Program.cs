using LLama;
using SandBox;
using SandBox.RunModels;
await SandBox.RunModels.DiagnosticModelRunner.RunAsync("Hello", ["qwen2-0_5b-instruct-q8_0", "Qwen3-0.6B-Q2_K"]);

Console.In.ReadLine();
return;