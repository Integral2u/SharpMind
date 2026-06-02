using BenchmarkDotNet.Running;

BenchmarkRunner.Run<SharpMind.Benchmarks.KernelsBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q8_0Benchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q3KBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.Q4KBenchmarks>();
//BenchmarkRunner.Run<SharpMind.Benchmarks.HSum256Benchmarks>();
Console.In.ReadLine();