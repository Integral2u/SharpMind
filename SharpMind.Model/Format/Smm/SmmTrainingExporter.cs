using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Tokenization;

namespace SharpMind.Model.Format;

/// <summary>
/// Exports a trained (F32) <see cref="TransformerWeights"/> into an .SMM
/// container using GGUF-compatible tensor names and layout, so the same
/// <see cref="SmmLoader"/> used for GGUF-converted files loads it.
///
/// Weights are mapped field-by-field from <see cref="TransformerWeights.BlockWeights"/>
/// (rather than relying on <c>Parameters()</c> names, which carry no block
/// index). FFN tensors are split into <c>ffn_gate</c>/<c>ffn_up</c> for the
/// fused gated layout, or emitted as <c>ffn_up</c>/<c>ffn_down</c> for dense.
/// </summary>
public static class SmmTrainingExporter
{
    /// <summary>
    /// Writes <paramref name="weights"/> to an .SMM file. The trained float
    /// weights are kept at F32 by default; pass
    /// <see cref="SmmWriteOptions.QuantizationLevel"/> to downcast (F16 is
    /// always safe; Q8_0/Q4_0 require dims divisible by 32).
    /// <paramref name="progress"/> reports 0..1 per tensor written.
    /// </summary>
    public static void Export(
        TransformerWeights weights,
        Tokenizer? tokenizer,
        string path,
        SmmWriteOptions? options = null,
        IProgress<float>? progress = null,
        string? chatTemplate = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        // Training presets (e.g. SharpMindConfig.Gpt) carry no architecture
        // name; tag the export so ForModel reproduces the same preset on load.
        var config = weights.Config with { Architecture = weights.Config.Architecture ?? "gpt2" };
        var tensorList = EnumerateTensors(weights).ToList();
        int total = tensorList.Count;
        int written = 0;
        IEnumerable<SmmTensorData> reporting = tensorList.Select(t =>
        {
            written++;
            progress?.Report((float)written / total);
            return t;
        });
        SmmWriter.Write(path, config, tokenizer, chatTemplate, reporting, options);
        progress?.Report(1f);
    }

    /// <summary>Enumerates the GGUF-layout tensors of a trained weight set.</summary>
    public static IEnumerable<SmmTensorData> EnumerateTensors(TransformerWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        // Global tensors
        yield return Tensor2D("token_embd.weight", weights.EmbeddingWeight);
        if (weights.LmHeadWeight is not null)
            yield return Tensor2D("output.weight", weights.LmHeadWeight);
        yield return Tensor1D("output_norm.weight", weights.FinalNormWeight);
        if (weights.FinalNormBias is not null)
            yield return Tensor1D("output_norm.bias", weights.FinalNormBias);

        int ffnDim = weights.Config.FfnDim;
        for (int i = 0; i < weights.Blocks.Length; i++)
        {
            var b = weights.Blocks[i];
            string blk = $"blk.{i}.";

            // Norms
            if (b.Norm1W is not null) yield return Tensor1D($"{blk}attn_norm.weight", b.Norm1W);
            if (b.Norm1B is not null) yield return Tensor1D($"{blk}attn_norm.bias", b.Norm1B);
            if (b.Norm2W is not null) yield return Tensor1D($"{blk}ffn_norm.weight", b.Norm2W);
            if (b.Norm2B is not null) yield return Tensor1D($"{blk}ffn_norm.bias", b.Norm2B);

            // Post-attention / post-FFN norms (Gemma-style)
            if (b.PostNorm1W is not null) yield return Tensor1D($"{blk}post_attention_norm.weight", b.PostNorm1W);
            if (b.PostNorm2W is not null) yield return Tensor1D($"{blk}post_ffw_norm.weight", b.PostNorm2W);

            // Attention
            if (b.Wq is not null) yield return Tensor2D($"{blk}attn_q.weight", b.Wq);
            if (b.Wk is not null) yield return Tensor2D($"{blk}attn_k.weight", b.Wk);
            if (b.Wv is not null) yield return Tensor2D($"{blk}attn_v.weight", b.Wv);
            if (b.Wo is not null) yield return Tensor2D($"{blk}attn_output.weight", b.Wo);
            if (b.WqBias is not null) yield return Tensor1D($"{blk}attn_q.bias", b.WqBias);
            if (b.WkBias is not null) yield return Tensor1D($"{blk}attn_k.bias", b.WkBias);
            if (b.WvBias is not null) yield return Tensor1D($"{blk}attn_v.bias", b.WvBias);
            if (b.WoBias is not null) yield return Tensor1D($"{blk}attn_output.bias", b.WoBias);
            if (b.QNormW is not null) yield return Tensor1D($"{blk}attn_q_norm.weight", b.QNormW);
            if (b.KNormW is not null) yield return Tensor1D($"{blk}attn_k_norm.weight", b.KNormW);

            // FFN — gated (fused gate+up) vs dense
            bool gated = b.Wf1 is { Shape.Length: 2 } && b.Wf1.Shape[1] == 2 * ffnDim && b.Wf2 is not null;
            if (b.Wf1 is not null && b.Wf2 is not null)
            {
                if (gated)
                {
                    yield return Tensor2D($"{blk}ffn_gate.weight", b.Wf1, 0, ffnDim);
                    yield return Tensor2D($"{blk}ffn_up.weight", b.Wf1, ffnDim, ffnDim);
                    yield return Tensor2D($"{blk}ffn_down.weight", b.Wf2);
                    if (b.Wf1Bias is not null)
                    {
                        yield return Tensor1D($"{blk}ffn_gate.bias", b.Wf1Bias, 0, ffnDim);
                        yield return Tensor1D($"{blk}ffn_up.bias", b.Wf1Bias, ffnDim, ffnDim);
                    }
                    if (b.Wf2Bias is not null)
                        yield return Tensor1D($"{blk}ffn_down.bias", b.Wf2Bias);
                }
                else
                {
                    yield return Tensor2D($"{blk}ffn_up.weight", b.Wf1);
                    yield return Tensor2D($"{blk}ffn_down.weight", b.Wf2);
                    if (b.Wf1Bias is not null)
                        yield return Tensor1D($"{blk}ffn_up.bias", b.Wf1Bias);
                    if (b.Wf2Bias is not null)
                        yield return Tensor1D($"{blk}ffn_down.bias", b.Wf2Bias);
                }
            }
        }
    }

    /// <summary>
    /// Emits a 2D float tensor in GGUF layout: stored shape = SharpMind shape
    /// (<c>[in, out]</c>), stored data = its row-major transpose (matching the
    /// buffer order <see cref="SmmLoader"/> / <see cref="GgufLoader"/> expect).
    /// A contiguous column slice (used to split a fused gated FFN weight into
    /// gate and up) is written with the same rule.
    /// </summary>
    private static SmmTensorData Tensor2D(string name, Tensor<float> source, int? colStart = null, int? colCount = null)
    {
        int rows = source.Shape.Rows;
        int cols = source.Shape.Cols;
        int start = colStart ?? 0;
        int outCols = colCount ?? cols;
        if (start + outCols > cols)
            throw new ArgumentException($"Column slice {start}+{outCols} exceeds tensor width {cols}.");

        var buffer = new float[checked(rows * outCols)];
        var data = source.Data;
        for (int i = 0; i < rows; i++)
            for (int o = 0; o < outCols; o++)
                buffer[o * rows + i] = data[i * cols + (start + o)];

        var bytes = new byte[buffer.Length * 4];
        Buffer.BlockCopy(buffer, 0, bytes, 0, bytes.Length);
        return new SmmTensorData
        {
            Name = name,
            Shape = [rows, outCols],
            Dtype = QuantDType.F32,
            GetBytes = () => bytes,
        };
    }

    private static SmmTensorData Tensor1D(string name, Tensor<float> source, int? start = null, int? count = null)
    {
        int offset = start ?? 0;
        int n = count ?? source.ElementCount;
        if (offset + n > source.ElementCount)
            throw new ArgumentException($"Slice {offset}+{n} exceeds tensor element count {source.ElementCount}.");

        float[] slice = source.Data.Slice(offset, n).ToArray();
        var bytes = new byte[n * 4];
        Buffer.BlockCopy(slice, 0, bytes, 0, bytes.Length);
        return new SmmTensorData
        {
            Name = name,
            Shape = [n],
            Dtype = QuantDType.F32,
            GetBytes = () => bytes,
        };
    }
}
