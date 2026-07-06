using System;
using System.Linq;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests;

public class QuickDtypeCheck
{
    private readonly ITestOutputHelper _output;
    public QuickDtypeCheck(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CheckQ2K_TensorTypes()
    {
        var meta = GgufLoaderFactory.Default.LoadMeta(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0.5b-instruct-q2_k.gguf");
        foreach (var g in meta.Tensors.GroupBy(t => t.Dtype).OrderBy(g => g.Key))
            _output.WriteLine($"  {g.Key}: {g.Count()} tensors");

        foreach (var t in meta.Tensors.Where(t => t.Name.Contains("blk.0.")))
            _output.WriteLine($"{t.Name}: {t.Dtype} [{string.Join(",", t.Shape)}]");

        foreach (var t in meta.Tensors.Where(t => t.Name.Contains("token_embd") || t.Name.Contains("output") || t.Name.Contains("lm_head")))
            _output.WriteLine($"{t.Name}: {t.Dtype} [{string.Join(",", t.Shape)}]");
    }
}
