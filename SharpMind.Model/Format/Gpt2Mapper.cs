namespace SharpMind.Model.Format;

/// <summary>GPT-2 weight mapper.</summary>
public sealed class Gpt2Mapper : WeightMapper
{
    public override string? MapWeight(string name, int[] shape)
    {
        if (name == "wte.weight")
            return "embedding.weight";
        if (name == "wpe.weight")
            return "position_emb.weight";
        if (name == "ln_f.weight")
            return "final_norm.weight";
        if (name == "lm_head.weight")
            return "lm_head.weight";

        if (name.StartsWith("h."))
        {
            var parts = name.Split('.');
            if (parts.Length >= 3 && int.TryParse(parts[1], out int layerIdx))
            {
                string last = string.Join(".", parts.Skip(2));
                if (last == "attn.c_attn.weight")
                    return $"blocks.{layerIdx}.attention.qkv.weight";
                if (last == "attn.c_proj.weight")
                    return $"blocks.{layerIdx}.attention.out.weight";
                if (last == "mlp.c_fc.weight")
                    return $"blocks.{layerIdx}.ffn.gate.weight";
                if (last == "mlp.c_proj.weight")
                    return $"blocks.{layerIdx}.ffn.down.weight";
                if (last == "ln_1.weight")
                    return $"blocks.{layerIdx}.attn_norm.weight";
                if (last == "ln_2.weight")
                    return $"blocks.{layerIdx}.ffn_norm.weight";
            }
        }

        return null;
    }
}