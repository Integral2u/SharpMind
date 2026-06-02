using BenchmarkDotNet.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpMind.Core.Quantization;

namespace SharpMind.Benchmarks
{
    public class KernelsBenchmarks
    {
        //| Method           | Mean     | Error     | StdDev   | Median   |
        //|----------------- |---------:|------d---:|---------:|---------:|
        //| HalfToFloat_F16C | 37.65 ns | 2.045  ns | 6.029 ns | 39.83 ns |
        //| HalfToFloat_F16C | 8.796 ns | 0.5179 ns | 1.527 ns | 9.549 ns |
        [Benchmark]
        public float HalfToFloat_F16C()
        {
            return QuantizationKernels.HalfToFloat_F16C((ushort)Random.Shared.Next(ushort.MinValue, ushort.MaxValue + 1));
        }
    }
}
