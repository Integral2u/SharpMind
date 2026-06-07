await SharpMind.Samples.Examples.BuilderOptions.RunAsync("hello");
Console.In.ReadLine();
var m = SharpMind.Model.Format.GgufLoader.LoadMeta(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q8_0.gguf");
Console.WriteLine("=== Searching for output.* / head / lm_* ===");
foreach (var t in m.Tensors)
    if (t.Name.Contains("output") || t.Name.Contains("head") || t.Name.Contains("lm_"))
        Console.WriteLine($"  FOUND: {t.Name} shape=[{string.Join(",",t.Shape)}] dtype={t.Dtype}");
Console.In.ReadLine();
