using BenchmarkDotNet.Running;

if (args.Length > 0 && args[0] == "--bench-to-config")
{
    string outputPath = args.Length > 1 ? args[1] : "qconfig.json";
    SharpMind.Benchmarks.BenchToConfig.Run(outputPath);
    return;
}

BenchmarkRunner.Run<SharpMind.Benchmarks.KernelsBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q8_0Benchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q3KBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q4KBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.HSum256Benchmarks>();
Console.In.ReadLine();
