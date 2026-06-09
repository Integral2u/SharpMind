using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Model;

/// <summary>
/// Immutable container for a Transformer's weights. 
/// Designed to be shared across multiple Transformer sessions to avoid reloading from disk.
/// </summary>
public sealed class TransformerWeights : IDisposable
{
    public ModelConfig Config { get; }
    public Tensor<float> EmbeddingWeight { get; }
    public Tensor<float>? LmHeadWeight { get; private set; }
    public void SetLmHead(Tensor<float> head) => LmHeadWeight = head;
    public Tensor<float> FinalNormWeight { get; }
    public Tensor<float>? FinalNormBias { get; }
    
    // Weights for each block
    public BlockWeights[] Blocks { get; }

    public TransformerWeights(ModelConfig config, Tensor<float> embedding, Tensor<float>? lmHead, Tensor<float> finalNormW, Tensor<float>? finalNormB, BlockWeights[] blocks)
    {
        Config = config;
        EmbeddingWeight = embedding;
        LmHeadWeight = lmHead;
        FinalNormWeight = finalNormW;
        FinalNormBias = finalNormB;
        Blocks = blocks;
    }

    public void Dispose()
    {
        EmbeddingWeight.Dispose();
        LmHeadWeight?.Dispose();
        FinalNormWeight.Dispose();
        FinalNormBias?.Dispose();
        foreach (var block in Blocks) block.Dispose();
    }

    public (Tensor<float>? target, BlockWeights? block, string? rawField) ResolveTarget(string name)
    {
        if (name.Contains("token_embd", StringComparison.OrdinalIgnoreCase)) return (EmbeddingWeight, null, null);
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase)) return (FinalNormWeight, null, null);
        // Exact match only — "attn_output.weight" contains "output.weight" but is a block tensor
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return (LmHeadWeight, null, null);
        
        var match = System.Text.RegularExpressions.Regex.Match(name, @"blk\.(\d+)\.");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int bIdx) && bIdx < Blocks.Length)
        {
            var block = Blocks[bIdx];
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            {
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
            }
            else
            {
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWq");
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWk");
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWv");
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWo");
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWgate");
                if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWup");
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWf2");
            }
        }
        return (null, null, null);
    }

    public Tensor<float>? ResolveFloatTarget(string name)
    {
        if (name.Contains("token_embd", StringComparison.OrdinalIgnoreCase)) return EmbeddingWeight;
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase)) return FinalNormWeight;
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return LmHeadWeight;
        
        var match = System.Text.RegularExpressions.Regex.Match(name, @"blk\.(\d+)\.");
        if (match.Success && int.TryParse(match.Groups[1].Value, out int bIdx) && bIdx < Blocks.Length)
        {
            var b = Blocks[bIdx];
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            {
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return b.WqBias;
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return b.WkBias;
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return b.WvBias;
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return b.WoBias;
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm1B;
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm2B;
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return b.Wf1Bias;
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return b.Wf2Bias;
            }
            else
            {
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return b.Wq;
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return b.Wk;
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return b.Wv;
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return b.Wo;
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm1W;
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm2W;
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return b.Wf1;
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return b.Wf2;
            }
        }
        return null;
    }

    public void SetRawField(BlockWeights block, string field, byte[] data, Format.GgufDtype dtype)
    {
        switch (field)
        {
            case "RawWq": block.RawWq = data; break;
            case "RawWk": block.RawWk = data; break;
            case "RawWv": block.RawWv = data; break;
            case "RawWo": block.RawWo = data; break;
            case "RawWgate": block.RawWgate = data; break;
            case "RawWup": block.RawWup = data; break;
            case "RawWf1": block.RawWf1 = data; break;
            case "RawWf2": block.RawWf2 = data; break;
        }
        block.QuantDtype = dtype;
    }

    public sealed class BlockWeights : IDisposable
    {
        // Attention
        public Tensor<float> Wq { get; }
        public Tensor<float> Wk { get; }
        public Tensor<float> Wv { get; }
        public Tensor<float> Wo { get; }
        public Tensor<float> WqBias { get; }
        public Tensor<float> WkBias { get; }
        public Tensor<float> WvBias { get; }
        public Tensor<float> WoBias { get; }

        // FFN
        public Tensor<float> Wf1 { get; } // Gate/Up
        public Tensor<float> Wf2 { get; } // Down
        public Tensor<float> Wf1Bias { get; }
        public Tensor<float> Wf2Bias { get; }

        // Norms
        public Tensor<float> Norm1W { get; }
        public Tensor<float>? Norm1B { get; }
        public Tensor<float> Norm2W { get; }
        public Tensor<float>? Norm2B { get; }
        
        // Quantized data
        public byte[]? RawWq { get; set; }
        public byte[]? RawWk { get; set; }
        public byte[]? RawWv { get; set; }
        public byte[]? RawWo { get; set; }
        public byte[]? RawWgate { get; set; }
        public byte[]? RawWup { get; set; }
        public byte[]? RawWf1 { get; set; }
        public byte[]? RawWf2 { get; set; }
        public Format.GgufDtype? QuantDtype { get; set; }

        public BlockWeights(Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo, 
                            Tensor<float> wqB, Tensor<float> wkB, Tensor<float> wvB, Tensor<float> woB,
                            Tensor<float> wf1, Tensor<float> wf2, Tensor<float> wf1B, Tensor<float> wf2B,
                            Tensor<float> n1w, Tensor<float>? n1b, Tensor<float> n2w, Tensor<float>? n2b)
        {
            Wq = wq; Wk = wk; Wv = wv; Wo = wo;
            WqBias = wqB; WkBias = wkB; WvBias = wvB; WoBias = woB;
            Wf1 = wf1; Wf2 = wf2; Wf1Bias = wf1B; Wf2Bias = wf2B;
            Norm1W = n1w; Norm1B = n1b; Norm2W = n2w; Norm2B = n2b;
        }

        public void Dispose()
        {
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
            WqBias.Dispose(); WkBias.Dispose(); WvBias.Dispose(); WoBias.Dispose();
            Wf1.Dispose(); Wf2.Dispose(); Wf1Bias.Dispose(); Wf2Bias.Dispose();
            Norm1W.Dispose(); Norm1B?.Dispose(); Norm2W.Dispose(); Norm2B?.Dispose();
        }
    }
}
