using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

public class WeightMapperTests
{
    [Fact]
    public void LlamaMapper_EmbedTokens_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.embed_tokens.weight", [32000, 4096]);

        Assert.NotNull(result);
        Assert.Equal("embedding.weight", result);
    }

    [Fact]
    public void LlamaMapper_FinalNorm_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.norm.weight", [4096]);

        Assert.NotNull(result);
        Assert.Equal("final_norm.weight", result);
    }

    [Fact]
    public void LlamaMapper_LmHead_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("lm_head.weight", [32000, 4096]);

        Assert.NotNull(result);
        Assert.Equal("lm_head.weight", result);
    }

    [Fact]
    public void LlamaMapper_AttentionQProj_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.0.self_attn.q_proj.weight", [4096, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.attention.Wq.weight", result);
    }

    [Fact]
    public void LlamaMapper_AttentionKProj_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.5.self_attn.k_proj.weight", [1024, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.5.attention.Wk.weight", result);
    }

    [Fact]
    public void LlamaMapper_AttentionVProj_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.10.self_attn.v_proj.weight", [1024, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.10.attention.Wv.weight", result);
    }

    [Fact]
    public void LlamaMapper_AttentionOProj_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.0.self_attn.o_proj.weight", [4096, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.attention.Wo.weight", result);
    }

    [Fact]
    public void LlamaMapper_MlpGate_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.0.mlp.gate_proj.weight", [11008, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.ffn.gate.weight", result);
    }

    [Fact]
    public void LlamaMapper_MlpUp_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.0.mlp.up_proj.weight", [11008, 4096]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.ffn.up.weight", result);
    }

    [Fact]
    public void LlamaMapper_MlpDown_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("model.layers.0.mlp.down_proj.weight", [4096, 11008]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.ffn.down.weight", result);
    }

    [Fact]
    public void LlamaMapper_LayerNorm_MapsCorrectly()
    {
        var mapper = new LlamaMapper();

        var result1 = mapper.MapWeight("model.layers.15.input_layernorm.weight", [4096]);
        var result2 = mapper.MapWeight("model.layers.15.post_attention_layernorm.weight", [4096]);

        Assert.NotNull(result1);
        Assert.NotNull(result2);
        Assert.Equal("blocks.15.attn_norm.weight", result1);
        Assert.Equal("blocks.15.ffn_norm.weight", result2);
    }

    [Fact]
    public void LlamaMapper_Unknown_ReturnsNull()
    {
        var mapper = new LlamaMapper();

        var result = mapper.MapWeight("unknown.layer.weight", [100, 100]);

        Assert.Null(result);
    }

    [Fact]
    public void Gpt2Mapper_Embed_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result = mapper.MapWeight("wte.weight", [50257, 768]);

        Assert.NotNull(result);
        Assert.Equal("embedding.weight", result);
    }

    [Fact]
    public void Gpt2Mapper_Attention_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result = mapper.MapWeight("h.0.attn.c_attn.weight", [2304, 768]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.attention.qkv.weight", result);
    }

    [Fact]
    public void Gpt2Mapper_OutputProj_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result = mapper.MapWeight("h.0.attn.c_proj.weight", [768, 768]);

        Assert.NotNull(result);
        Assert.Equal("blocks.0.attention.out.weight", result);
    }

    [Fact]
    public void Gpt2Mapper_Mlp_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result1 = mapper.MapWeight("h.0.mlp.c_fc.weight", [3072, 768]);
        var result2 = mapper.MapWeight("h.0.mlp.c_proj.weight", [768, 3072]);

        Assert.NotNull(result1);
        Assert.NotNull(result2);
        Assert.Equal("blocks.0.ffn.gate.weight", result1);
        Assert.Equal("blocks.0.ffn.down.weight", result2);
    }

    [Fact]
    public void Gpt2Mapper_LayerNorm_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result1 = mapper.MapWeight("h.0.ln_1.weight", [768]);
        var result2 = mapper.MapWeight("h.0.ln_2.weight", [768]);

        Assert.NotNull(result1);
        Assert.NotNull(result2);
        Assert.Equal("blocks.0.attn_norm.weight", result1);
        Assert.Equal("blocks.0.ffn_norm.weight", result2);
    }

    [Fact]
    public void Gpt2Mapper_FinalNorm_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result = mapper.MapWeight("ln_f.weight", [768]);

        Assert.NotNull(result);
        Assert.Equal("final_norm.weight", result);
    }

    [Fact]
    public void Gpt2Mapper_LmHead_MapsCorrectly()
    {
        var mapper = new Gpt2Mapper();

        var result = mapper.MapWeight("lm_head.weight", [50257, 768]);

        Assert.NotNull(result);
        Assert.Equal("lm_head.weight", result);
    }
}