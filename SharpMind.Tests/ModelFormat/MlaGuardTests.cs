using System;
using System.Linq;
using SharpMind.Core.Quantization;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// MLA binding is chosen by tensor, not by architecture name: the repo keeps no
/// architecture denylist, and a new arch string must not be able to slip past a
/// check. What the loader has to get right is the three-way decision — fused
/// kv_b (supported), split k_b/v_b (not bound yet), or MLA tensors with no
/// kv_lora_rank (cannot be sized) — plus the config values the layer needs.
/// </summary>
public sealed class MlaGuardTests
{
    private static ModelMetaData Meta(params (string Name, QuantDType Dtype)[] tensors)
    {
        var meta = new ModelMetaData
        {
            KvPairs =
            [
                new KvPair { Key = "general.architecture", Value = "deepseek2" },
                new KvPair { Key = "deepseek2.block_count", Value = 2L },
                new KvPair { Key = "deepseek2.attention.head_count", Value = 16L },
                new KvPair { Key = "deepseek2.attention.key_length", Value = 192L },
                new KvPair { Key = "deepseek2.attention.value_length", Value = 128L },
                new KvPair { Key = "deepseek2.rope.dimension_count", Value = 64L },
                new KvPair { Key = "deepseek2.expert_feed_forward_length", Value = 1408L },
                new KvPair { Key = "deepseek2.expert_shared_count", Value = 2L },
                new KvPair { Key = "deepseek2.leading_dense_block_count", Value = 1L },
            ],
        };
        foreach (var (name, dtype) in tensors)
        {
            meta.Tensors.Add(new TensorInfo { Name = name, Dtype = dtype, Shape = [64, 64], Offset = 0 });
        }
        return meta;
    }

    private static void Add(ModelMetaData meta, string key, long value)
        => meta.KvPairs.Add(new KvPair { Key = key, Value = value });

    [Fact]
    public void LoadConfig_ReadsTheMlaGeometry()
    {
        var meta = Meta(
            ("token_embd.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_a_mqa.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_a_norm.weight", QuantDType.F32),
            ("blk.0.attn_kv_b.weight", QuantDType.Q2_K));
        Add(meta, "deepseek2.attention.kv_lora_rank", 512);

        var config = GgufLoader.LoadConfig(meta);

        Assert.NotNull(config);
        Assert.True(config!.IsMla);
        Assert.Equal(512, config.KvLoraRank);
        Assert.Equal(192, config.HeadDim);
        Assert.Equal(128, config.ValueDim);
        Assert.Equal(64, config.MlaRopeHeadDim);
        Assert.Equal(128, config.MlaNopeHeadDim);
        // 2 shared experts fused into one 2816-wide tensor, not two layers.
        Assert.Equal(2816, config.SharedExpertFfnDim);
        Assert.Equal(1, config.LeadingDenseLayers);
        Assert.True(config.IsDenseFfnLayer(0));
        Assert.False(config.IsDenseFfnLayer(1));
    }

    [Fact]
    public void LoadConfig_AcceptsTheFusedKvBLayout()
    {
        var meta = Meta(
            ("token_embd.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_a_mqa.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_b.weight", QuantDType.Q2_K));
        Add(meta, "deepseek2.attention.kv_lora_rank", 512);

        // Must not throw: this is the layout both local MLA models ship.
        var config = GgufLoader.LoadConfig(meta);
        Assert.True(config!.IsMla);
    }

    [Fact]
    public void LoadConfig_RefusesTheSplitKvBLayoutByName()
    {
        var meta = Meta(
            ("token_embd.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_a_mqa.weight", QuantDType.Q2_K),
            ("blk.0.attn_k_b.weight", QuantDType.Q2_K),
            ("blk.0.attn_v_b.weight", QuantDType.Q2_K));
        Add(meta, "deepseek2.attention.kv_lora_rank", 512);

        var ex = Assert.Throws<NotSupportedException>(() => GgufLoader.LoadConfig(meta));
        Assert.Contains("split k_b/v_b", ex.Message);
        Assert.Contains("attn_kv_a_mqa", ex.Message);
    }

    [Fact]
    public void LoadConfig_RefusesMlaWithoutAKvLoraRank()
    {
        // No kv_lora_rank: the latent width is unknown, so the layer could not be
        // sized and the model would answer from garbage.
        var meta = Meta(
            ("token_embd.weight", QuantDType.Q2_K),
            ("blk.0.attn_kv_a_mqa.weight", QuantDType.Q2_K));

        var ex = Assert.Throws<NotSupportedException>(() => GgufLoader.LoadConfig(meta));
        Assert.Contains("kv_lora_rank", ex.Message);
    }

    [Fact]
    public void LoadConfig_AcceptsPlainAttention()
    {
        var meta = Meta("token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
            "blk.0.attn_v.weight", "blk.0.attn_output.weight");

        var config = GgufLoader.LoadConfig(meta);
        Assert.False(config!.IsMla);
        Assert.Equal(0, config.KvLoraRank);
    }

    private static ModelMetaData Meta(params string[] tensorNames)
        => Meta(tensorNames.Select(n => (n, QuantDType.F16)).ToArray());
}
