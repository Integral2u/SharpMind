using System;
using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// MLA files carry <c>attn_q_a</c>/<c>attn_q_b</c> and
/// <c>attn_kv_a_mqa</c>/<c>attn_kv_b</c>. The attention layer unpacks weights by
/// substring, and every one of those names contains the plain
/// <c>attn_q</c>/<c>attn_k</c> prefix, so an MLA file used to load with its
/// sub-projections written into the wrong tensors and then answer from garbage.
/// The guard is by tensor, not by architecture name, so a future arch string
/// cannot sneak past it and no real architecture is denylisted.
/// </summary>
public sealed class MlaGuardTests
{
    private static ModelMetaData Meta(params string[] tensorNames)
    {
        var meta = new ModelMetaData
        {
            KvPairs =
            [
                new KvPair { Key = "general.architecture", Value = "deepseek2" },
                new KvPair { Key = "deepseek2.block_count", Value = 2L },
            ],
        };
        foreach (string name in tensorNames)
        {
            meta.Tensors.Add(new TensorInfo
            {
                Name = name,
                Dtype = QuantDType.F16,
                Shape = [64, 64],
                Offset = 0,
            });
        }
        return meta;
    }

    [Theory]
    [InlineData("blk.0.attn_kv_a_mqa.weight")]
    [InlineData("blk.13.attn_kv_a_mqa.weight")]
    public void LoadConfig_RefusesAnMlaTensorSet(string mlaTensor)
    {
        var meta = Meta("token_embd.weight", "blk.0.attn_q_a.weight", mlaTensor);

        var ex = Assert.Throws<NotSupportedException>(() => GgufLoader.LoadConfig(meta));

        Assert.Contains("Multi-head Latent Attention", ex.Message);
        Assert.Contains(mlaTensor, ex.Message);
    }

    [Fact]
    public void LoadConfig_AcceptsPlainAttention()
    {
        var meta = Meta("token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", "blk.0.attn_output.weight");

        // Throws for missing shapes/keys rather than for MLA; what matters is
        // that the MLA guard did not fire.
        var ex = Record.Exception(() => GgufLoader.LoadConfig(meta));
        Assert.False(ex is NotSupportedException { Message: var m } && m.Contains("Multi-head Latent Attention"));
    }
}
