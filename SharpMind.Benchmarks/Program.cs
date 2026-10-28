using BenchmarkDotNet.Running;

// Run in Release:  dotnet run -c Release --project SharpMind.Benchmarks -- --filter "*"
BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
