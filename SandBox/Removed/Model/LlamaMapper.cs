namespace SharpMind.Model.Format;

/// <summary>Standard Llama weight mapper (used by most HF models).</summary>
public sealed class LlamaMapper() : WeightMapper
{

    public override string? MapWeight(string name, int[] shape)
    {
        if (name == "model.embed_tokens.weight")
            return "embedding.weight";
        if (name == "model.norm.weight")
            return "final_norm.weight";
        if (name == "lm_head.weight")
            return "lm_head.weight";

        if (name.EndsWith(".input_layernorm.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.attn_norm.weight";
        }
        if (name.EndsWith(".post_attention_layernorm.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.ffn_norm.weight";
        }

        if (name.EndsWith(".self_attn.q_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.attention.Wq.weight";
        }
        if (name.EndsWith(".self_attn.k_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.attention.Wk.weight";
        }
        if (name.EndsWith(".self_attn.v_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.attention.Wv.weight";
        }
        if (name.EndsWith(".self_attn.o_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.attention.Wo.weight";
        }

        if (name.EndsWith(".mlp.gate_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.ffn.gate.weight";
        }
        if (name.EndsWith(".mlp.up_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.ffn.up.weight";
        }
        if (name.EndsWith(".mlp.down_proj.weight"))
        {
            var idx = ParseLayerIdx(name, "model.layers");
            if (idx >= 0) return $"blocks.{idx}.ffn.down.weight";
        }

        return null;
    }

    private static int ParseLayerIdx(string name, string prefix)
    {
        if (!name.StartsWith(prefix)) return -1;
        var rel = name[prefix.Length..];
        if (rel.Length < 3) return -1;
        var parts = rel.Split('.');
        if (parts.Length >= 2 && int.TryParse(parts[1], out int idx))
            return idx;
        return -1;
    }
}

