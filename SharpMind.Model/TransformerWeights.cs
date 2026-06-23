using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Model;

/// <summary>
/// Immutable container for a Transformer's weights. 
/// Designed to be shared across multiple Transformer sessions to avoid reloading from disk.
/// </summary>
public sealed class TransformerWeights(ModelConfig config, Tensor<float> embedding, Tensor<float>? lmHead, Tensor<float> finalNormW, Tensor<float>? finalNormB, TransformerWeights.BlockWeights[] blocks) : IDisposable
{
    public ModelConfig Config { get; } = config;
    public Tensor<float> EmbeddingWeight { get; } = embedding;
    public Tensor<float>? LmHeadWeight { get; private set; } = lmHead;
    public void SetLmHead(Tensor<float> head) => LmHeadWeight = head;
    public Tensor<float> FinalNormWeight { get; } = finalNormW;
    public Tensor<float>? FinalNormBias { get; } = finalNormB;

    // Raw quantized data for non-block tensors (embedding, lm_head)
    public byte[]? RawEmbedding { get; set; }
    public Format.GgufDtype? RawEmbeddingDtype { get; set; }

    // Weights for each block
    public BlockWeights[] Blocks { get; } = blocks;

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

        var match = RegexGenerated.LayerIndexDotNDot.Match(name);// System.Text.RegularExpressions.Regex.Match(name, @"blk\.(\d+)\.");
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
                // Q/K norm checks must be BEFORE the broader attn_q/attn_k checks
                // to avoid "attn_q_norm" being misidentified as "attn_q".
                if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
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
        
        var match = RegexGenerated.LayerIndexDotNDot.Match(name); //System.Text.RegularExpressions.Regex.Match(name, @"blk\.(\d+)\.");
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
                if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.QNormW ??= new Tensor<float>(Config.HeadDim);
                    return b.QNormW;
                }
                if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.KNormW ??= new Tensor<float>(Config.HeadDim);
                    return b.KNormW;
                }
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

    public static void SetRawField(BlockWeights block, string field, byte[] data, Format.GgufDtype dtype)
    {
        switch (field)
        {
            case "RawWq": block.RawWq = data; block.QuantDtypeWq = dtype; break;
            case "RawWk": block.RawWk = data; block.QuantDtypeWk = dtype; break;
            case "RawWv": block.RawWv = data; block.QuantDtypeWv = dtype; break;
            case "RawWo": block.RawWo = data; block.QuantDtypeWo = dtype; break;
            case "RawWgate": block.RawWgate = data; block.QuantDtypeWgate = dtype; break;
            case "RawWup": block.RawWup = data; block.QuantDtypeWup = dtype; break;
            case "RawWf1": block.RawWf1 = data; block.QuantDtypeWf1 = dtype; break;
            case "RawWf2": block.RawWf2 = data; block.QuantDtypeWf2 = dtype; break;
        }
        block.QuantDtype = dtype; // backwards compat
    }

    public sealed class BlockWeights(Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo,
                        Tensor<float> wqB, Tensor<float> wkB, Tensor<float> wvB, Tensor<float> woB,
                        Tensor<float> wf1, Tensor<float> wf2, Tensor<float> wf1B, Tensor<float> wf2B,
                        Tensor<float> n1w, Tensor<float>? n1b, Tensor<float> n2w, Tensor<float>? n2b,
                        Tensor<float>? qNorm, Tensor<float>? kNorm) : IDisposable
    {
        // Attention
        public Tensor<float> Wq { get; } = wq; public Tensor<float> Wk { get; } = wk; public Tensor<float> Wv { get; } = wv; public Tensor<float> Wo { get; } = wo;
        public Tensor<float> WqBias { get; } = wqB; public Tensor<float> WkBias { get; } = wkB; public Tensor<float> WvBias { get; } = wvB; public Tensor<float> WoBias { get; } = woB;

        // FFN
        public Tensor<float> Wf1 { get; } = wf1; public Tensor<float> Wf2 { get; } = wf2; public Tensor<float> Wf1Bias { get; } = wf1B; public Tensor<float> Wf2Bias { get; } = wf2B;

        // Norms
        public Tensor<float> Norm1W { get; } = n1w; public Tensor<float>? Norm1B { get; } = n1b; public Tensor<float> Norm2W { get; } = n2w; public Tensor<float>? Norm2B { get; } = n2b;

        // Per-head Q/K normalization (Qwen3)
        // Settable so ResolveFloatTarget can create lazily.
        public Tensor<float>? QNormW { get; set; } = qNorm;
        public Tensor<float>? KNormW { get; set; } = kNorm;

        // Quantized data
        public byte[]? RawWq { get; set; }
        public byte[]? RawWk { get; set; }
        public byte[]? RawWv { get; set; }
        public byte[]? RawWo { get; set; }
        public byte[]? RawWgate { get; set; }
        public byte[]? RawWup { get; set; }
        public byte[]? RawWf1 { get; set; }
        public byte[]? RawWf2 { get; set; }

        // Per-tensor quantization dtype (one per raw field)
        public Format.GgufDtype? QuantDtypeWq { get; set; }
        public Format.GgufDtype? QuantDtypeWk { get; set; }
        public Format.GgufDtype? QuantDtypeWv { get; set; }
        public Format.GgufDtype? QuantDtypeWo { get; set; }
        public Format.GgufDtype? QuantDtypeWgate { get; set; }
        public Format.GgufDtype? QuantDtypeWup { get; set; }
        public Format.GgufDtype? QuantDtypeWf1 { get; set; }
        public Format.GgufDtype? QuantDtypeWf2 { get; set; }
        [Obsolete("Use per-tensor QuantDtype fields instead. This field is overwritten by the last tensor processed.")]
        public Format.GgufDtype? QuantDtype { get; set; }

        public void Dispose()
        {
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
            WqBias.Dispose(); WkBias.Dispose(); WvBias.Dispose(); WoBias.Dispose();
            Wf1.Dispose(); Wf2.Dispose(); Wf1Bias.Dispose(); Wf2Bias.Dispose();
            Norm1W.Dispose(); Norm1B?.Dispose(); Norm2W.Dispose(); Norm2B?.Dispose();
            QNormW?.Dispose(); KNormW?.Dispose();
        }
    }
}
