using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Ffn;
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
///
/// MoE router and per-expert tensors are not present in
/// <see cref="TransformerWeights.BlockWeights"/> — they live on the assembled
/// <see cref="FfnLayer"/> (see <see cref="FfnLayer.RouterLayer"/>) — so an
/// optional <paramref name="model"/> is required to export them as
/// <c>blk.{i}.ffn_gate.*</c> (router) and <c>blk.{i}.exps.{e}.*</c>.
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
        string? chatTemplate = null,
        Transformer? model = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        // Training presets (e.g. SharpMindConfig.Gpt) carry no architecture
        // name; tag the export so ForModel reproduces the same preset on load.
        var config = weights.Config with { Architecture = weights.Config.Architecture ?? "gpt2" };
        var tensorList = EnumerateTensors(weights, model).ToList();
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
    public static IEnumerable<SmmTensorData> EnumerateTensors(TransformerWeights weights, Transformer? model = null)
    {
        ArgumentNullException.ThrowIfNull(weights);

        // Global tensors
        yield return Tensor2D("token_embd.weight", weights.EmbeddingWeight, transpose: false);
        if (weights.PositionEmbedding is not null)
            yield return Tensor2D("position_embd.weight", weights.PositionEmbedding, transpose: false);
        if (weights.LmHeadWeight is not null)
            yield return Tensor2D("output.weight", weights.LmHeadWeight, transpose: false);
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

            // MoE — router + per-expert tensors. They live on the assembled
            // FfnLayer during training; after a load they may also be present
            // as float fields on BlockWeights. Emit from whichever is available.
            var ffn = model?.GetBlock(i)?.Ffn;
            bool isMoE = ffn?.RouterLayer is not null
                         || b.WRouter is not null
                         || b.WgateExp is { Count: > 0 }
                         || b.WupExp is { Count: > 0 }
                         || b.WdownExp is { Count: > 0 };
            if (isMoE)
            {
                int numExperts = weights.Config.NumExperts;
                if (numExperts > 0 && ffn?.RouterLayer is not null)
                {
                    yield return Tensor2D($"{blk}ffn_gate.weight", ffn.RouterLayer.Weight);
                    if (ffn.RouterLayer.Bias is not null) yield return Tensor1D($"{blk}ffn_gate.bias", ffn.RouterLayer.Bias);
                }
                else if (b.WRouter is not null)
                {
                    yield return Tensor2D($"{blk}ffn_gate.weight", b.WRouter);
                    if (b.WRouterBias is not null) yield return Tensor1D($"{blk}ffn_gate.bias", b.WRouterBias);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"Block {i} is a MoE FFN: a {nameof(Transformer)} model is required to export router/expert tensors.");
                }

                for (int e = 0; e < numExperts; e++)
                {
                    string exps = $"{blk}exps.{e}.";
                    var mFfn = ffn as MoEFfnLayer;
                    if (mFfn is not null)
                    {
                        var gate = mFfn.ExpertGateLayers![e];
                        var up = mFfn.ExpertUpLayers![e];
                        var down = mFfn.ExpertDownLayers![e];
                        yield return Tensor2D($"{exps}ffn_gate.weight", gate.Weight);
                        if (gate.Bias is not null) yield return Tensor1D($"{exps}ffn_gate.bias", gate.Bias);
                        yield return Tensor2D($"{exps}ffn_up.weight", up.Weight);
                        if (up.Bias is not null) yield return Tensor1D($"{exps}ffn_up.bias", up.Bias);
                        yield return Tensor2D($"{exps}ffn_down.weight", down.Weight);
                        if (down.Bias is not null) yield return Tensor1D($"{exps}ffn_down.bias", down.Bias);
                    }
                    else
                    {
                        if (b.WgateExp is not null && b.WgateExp.TryGetValue(e, out var gateW)) yield return Tensor2D($"{exps}ffn_gate.weight", gateW);
                        if (b.WgateExpBias is not null && b.WgateExpBias.TryGetValue(e, out var gateB)) yield return Tensor1D($"{exps}ffn_gate.bias", gateB);
                        if (b.WupExp is not null && b.WupExp.TryGetValue(e, out var upW)) yield return Tensor2D($"{exps}ffn_up.weight", upW);
                        if (b.WupExpBias is not null && b.WupExpBias.TryGetValue(e, out var upB)) yield return Tensor1D($"{exps}ffn_up.bias", upB);
                        if (b.WdownExp is not null && b.WdownExp.TryGetValue(e, out var downW)) yield return Tensor2D($"{exps}ffn_down.weight", downW);
                        if (b.WdownExpBias is not null && b.WdownExpBias.TryGetValue(e, out var downB)) yield return Tensor1D($"{exps}ffn_down.bias", downB);
                    }
                }
                continue;
            }

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
    /// gate and up) is written with the same rule. Pass <paramref name="transpose"/>
    /// = false for tensors that are stored row-major in GGUF (e.g. the embedding,
    /// which SmmLoader copies verbatim into <c>[vocab, hidden]</c>).
    /// </summary>
    private static SmmTensorData Tensor2D(string name, Tensor<float> source, int? colStart = null, int? colCount = null, bool transpose = true)
    {
        int rows = source.Shape.Rows;
        int cols = source.Shape.Cols;
        int start = colStart ?? 0;
        int outCols = colCount ?? cols;
        if (start + outCols > cols)
            throw new ArgumentException($"Column slice {start}+{outCols} exceeds tensor width {cols}.");

        var buffer = new float[checked(rows * outCols)];
        var data = source.Data;
        if (transpose)
        {
            for (int i = 0; i < rows; i++)
                for (int o = 0; o < outCols; o++)
                    buffer[o * rows + i] = data[i * cols + (start + o)];
        }
        else
        {
            for (int i = 0; i < rows; i++)
                for (int o = 0; o < outCols; o++)
                    buffer[i * outCols + o] = data[i * cols + (start + o)];
        }

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
