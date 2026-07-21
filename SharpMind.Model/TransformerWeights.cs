using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Model;

/// <summary>Metadata for one weight tensor: file offset, byte size, quantization dtype.</summary>
public readonly record struct TensorMeta(long Offset, int Size, QuantDType Dtype);

/// <summary>Container for a Transformer's weights — all weights loaded into memory.</summary>
public abstract class TransformerWeights : IDisposable
{
    public ModelConfig Config { get; }
    public Tensor<float> EmbeddingWeight { get; }
    public Tensor<float>? LmHeadWeight { get; protected set; }
    public void SetLmHead(Tensor<float> head) => LmHeadWeight = head;
    public Tensor<float> FinalNormWeight { get; }
    public Tensor<float>? FinalNormBias { get; }

    // Raw quantized data for non-block tensors (embedding, lm_head)
    public byte[]? RawEmbedding { get; set; }
    public QuantDType? RawEmbeddingDtype { get; set; }
    public byte[]? RawLmHead { get; set; }
    public QuantDType? RawLmHeadDtype { get; set; }

    // Per-block weights (Attention, FFN, Norms)
    public BlockWeights[] Blocks { get; }

    // GGUF metadata
    public Format.ModelMetaData? GgufMeta { get; set; }
    public string? GgufPath { get; set; }
    public bool IsMoE { get; set; }

    // The loader used during InitializeWeights
    protected IModelLoader? Loader { get; }

    protected TransformerWeights(
        ModelConfig config,
        Tensor<float> embedding,
        Tensor<float>? lmHead,
        Tensor<float> finalNormW,
        Tensor<float>? finalNormB,
        BlockWeights[] blocks,
        IModelLoader? loader)
    {
        Config = config;
        EmbeddingWeight = embedding;
        LmHeadWeight = lmHead;
        FinalNormWeight = finalNormW;
        FinalNormBias = finalNormB;
        Blocks = blocks;
        Loader = loader;
    }

    /// <summary>Initialises weights using the stored <see cref="IModelLoader"/>.
    /// Called after construction — must be called exactly once.</summary>
    public abstract void InitializeWeights(IProgress<float>? progress = null);

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            EmbeddingWeight.Dispose();
            LmHeadWeight?.Dispose();
            FinalNormWeight.Dispose();
            FinalNormBias?.Dispose();
            foreach (var block in Blocks) block.Dispose();
        }
    }

    public (Tensor<float>? target, BlockWeights? block, string? rawField) ResolveTarget(string name)
    {
        if (name.Contains("token_embd", StringComparison.OrdinalIgnoreCase)) return (EmbeddingWeight, null, null);
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return (FinalNormBias, null, null);
            return (FinalNormWeight, null, null);
        }
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return (LmHeadWeight, null, null);

        var match = RegexGenerated.LayerIndexDotNDot.Match(name);
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
                if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWq");
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWk");
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWv");
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWo");
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);

                if (IsMoE && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWgateExp_{expIdx}");
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWupExp_{expIdx}");
                        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWdownExp_{expIdx}");
                    }
                    return (null, block, null);
                }

                if (IsMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                    return (null, block, "RawRouter");

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
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return FinalNormBias;
            return FinalNormWeight;
        }
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return LmHeadWeight;

        var match = RegexGenerated.LayerIndexDotNDot.Match(name);
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

                if (IsMoE && (name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)))
                    return null;

                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return b.Wf1;
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return b.Wf2;
            }
        }
        return null;
    }

    public static void SetRawField(BlockWeights block, string field, byte[] data, QuantDType dtype)
    {
        if (field.StartsWith("RawWgateExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(12), out int gateExp))
        {
            block.RawWgateExp ??= [];
            block.RawWgateExp[gateExp] = data;
            block.QuantDtypeWgateExp ??= [];
            block.QuantDtypeWgateExp[gateExp] = dtype;
            return;
        }
        if (field.StartsWith("RawWupExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(10), out int upExp))
        {
            block.RawWupExp ??= [];
            block.RawWupExp[upExp] = data;
            block.QuantDtypeWupExp ??= [];
            block.QuantDtypeWupExp[upExp] = dtype;
            return;
        }
        if (field.StartsWith("RawWdownExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(12), out int downExp))
        {
            block.RawWdownExp ??= [];
            block.RawWdownExp[downExp] = data;
            block.QuantDtypeWdownExp ??= [];
            block.QuantDtypeWdownExp[downExp] = dtype;
            return;
        }

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
            case "RawRouter": block.RawRouter = data; block.QuantDtypeRouter = dtype; break;
        }
    }

    /// <summary>Records tensor metadata in the block's <see cref="BlockWeights.TensorMeta"/>
    /// dictionary (field → {offset, size, dtype}) without loading data.</summary>
    public static void SetTensorMeta(BlockWeights block, string field, long offset, int size, QuantDType dtype)
    {
        block.TensorMeta[field] = new TensorMeta(offset, size, dtype);
    }

    public sealed class BlockWeights : IDisposable
    {
        // Attention float tensors (nullable — Full mode populates all; Cached mode populates on demand)
        public Tensor<float>? Wq { get; set; }
        public Tensor<float>? Wk { get; set; }
        public Tensor<float>? Wv { get; set; }
        public Tensor<float>? Wo { get; set; }
        public Tensor<float>? WqBias { get; set; }
        public Tensor<float>? WkBias { get; set; }
        public Tensor<float>? WvBias { get; set; }
        public Tensor<float>? WoBias { get; set; }

        // FFN float tensors
        public Tensor<float>? Wf1 { get; set; }
        public Tensor<float>? Wf2 { get; set; }
        public Tensor<float>? Wf1Bias { get; set; }
        public Tensor<float>? Wf2Bias { get; set; }

        // Norm float tensors
        public Tensor<float>? Norm1W { get; set; }
        public Tensor<float>? Norm1B { get; set; }
        public Tensor<float>? Norm2W { get; set; }
        public Tensor<float>? Norm2B { get; set; }

        // Per-head Q/K normalization (Qwen3)
        public Tensor<float>? QNormW { get; set; }
        public Tensor<float>? KNormW { get; set; }

        // Quantized data (byte arrays)
        public byte[]? RawWq { get; set; }
        public byte[]? RawWk { get; set; }
        public byte[]? RawWv { get; set; }
        public byte[]? RawWo { get; set; }
        public byte[]? RawWgate { get; set; }
        public byte[]? RawWup { get; set; }
        public byte[]? RawWf1 { get; set; }
        public byte[]? RawWf2 { get; set; }

        // MoE expert quantized data
        public Dictionary<int, byte[]>? RawWgateExp { get; set; }
        public Dictionary<int, byte[]>? RawWupExp { get; set; }
        public Dictionary<int, byte[]>? RawWdownExp { get; set; }
        public byte[]? RawRouter { get; set; }

        // Per-tensor quantization dtype
        public QuantDType? QuantDtypeWq { get; set; }
        public QuantDType? QuantDtypeWk { get; set; }
        public QuantDType? QuantDtypeWv { get; set; }
        public QuantDType? QuantDtypeWo { get; set; }
        public QuantDType? QuantDtypeWgate { get; set; }
        public QuantDType? QuantDtypeWup { get; set; }
        public QuantDType? QuantDtypeWf1 { get; set; }
        public QuantDType? QuantDtypeWf2 { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWgateExp { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWupExp { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWdownExp { get; set; }
        public QuantDType? QuantDtypeRouter { get; set; }

        // Tensor metadata (offset, size, dtype) populated by IModelLoader.PreInit
        public Dictionary<string, TensorMeta> TensorMeta { get; } = [];

        public BlockWeights() { }

        public BlockWeights(
            Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo,
            Tensor<float> wqB, Tensor<float> wkB, Tensor<float> wvB, Tensor<float> woB,
            Tensor<float> wf1, Tensor<float> wf2, Tensor<float> wf1B, Tensor<float> wf2B,
            Tensor<float> n1w, Tensor<float>? n1b, Tensor<float> n2w, Tensor<float>? n2b,
            Tensor<float>? qNorm, Tensor<float>? kNorm)
        {
            Wq = wq; Wk = wk; Wv = wv; Wo = wo;
            WqBias = wqB; WkBias = wkB; WvBias = wvB; WoBias = woB;
            Wf1 = wf1; Wf2 = wf2; Wf1Bias = wf1B; Wf2Bias = wf2B;
            Norm1W = n1w; Norm1B = n1b; Norm2W = n2w; Norm2B = n2b;
            QNormW = qNorm; KNormW = kNorm;
        }

        public void Dispose()
        {
            Wq?.Dispose(); Wk?.Dispose(); Wv?.Dispose(); Wo?.Dispose();
            WqBias?.Dispose(); WkBias?.Dispose(); WvBias?.Dispose(); WoBias?.Dispose();
            Wf1?.Dispose(); Wf2?.Dispose(); Wf1Bias?.Dispose(); Wf2Bias?.Dispose();
            Norm1W?.Dispose(); Norm1B?.Dispose(); Norm2W?.Dispose(); Norm2B?.Dispose();
            QNormW?.Dispose(); KNormW?.Dispose();
        }
    }
}

/// <summary>Loads all weights into memory at once.</summary>
public sealed class TransformerWeightsFull : TransformerWeights
{
    public TransformerWeightsFull(
        ModelConfig config,
        Tensor<float> embedding,
        Tensor<float>? lmHead,
        Tensor<float> finalNormW,
        Tensor<float>? finalNormB,
        BlockWeights[] blocks,
        IModelLoader loader)
        : base(config, embedding, lmHead, finalNormW, finalNormB, blocks, loader) { }

    public override void InitializeWeights(IProgress<float>? progress = null)
    {
        Loader!.LoadAllWeights(this, progress);
    }
}


