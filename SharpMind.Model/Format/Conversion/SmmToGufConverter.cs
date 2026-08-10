using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Text.Json.Nodes;

namespace SharpMind.Model.Format.Conversion;

/// <summary>
/// Converts a SharpMind Model (.SMM) container back into a GGUF file — the
/// inverse of <see cref="GgufToSmmConverter"/>.
///
/// <summary>
/// Converts a SharpMind Model (.SMM) container back into a GGUF file — the
/// inverse of <see cref="GgufToSmmConverter"/>.
///
/// Everything the source .SMM embedded is restored: the <see cref="ModelConfig"/>
/// (rebuilt as GGUF <c>{arch}.*</c> metadata keys), the tokenizer (as
/// <c>tokenizer.ggml.*</c> arrays), the chat template, and every tensor's raw
/// (already-quantized) bytes, streamed verbatim — no re-quantization, no
/// full-model buffer — so GGUF→SMM→GGUF is lossless at the byte level.
/// </summary>
public static class SmmToGufConverter
{
    /// <summary>
    /// Converts <paramref name="smmPath"/> to <paramref name="ggufPath"/> using
    /// GGUF v3 and 32-byte alignment. <paramref name="progress"/> reports 0..1
    /// as tensors are written; <paramref name="ct"/> cancels the copy — the
    /// partial temp file is then deleted and <paramref name="ggufPath"/> stays untouched.
    /// </summary>
    public static void Convert(
        string smmPath,
        string ggufPath,
        CancellationToken ct = default,
        IProgress<float>? progress = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(smmPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(ggufPath);
        if (!File.Exists(smmPath)) throw new FileNotFoundException(smmPath);

        ct.ThrowIfCancellationRequested();

        var meta = SmmLoader.LoadMeta(smmPath);
        var config = SmmLoader.LoadConfig(meta)
            ?? throw new InvalidDataException("SMM is missing its model config (smm.config_json).");
        var tokenizer = SmmLoader.LoadTokenizerFromMeta(meta);

        var kv = BuildKvPairs(meta, config, tokenizer);
        var tensors = BuildTensors(smmPath, meta, ct, progress);

        string tmpPath = ggufPath + ".tmp";
        try
        {
            GgufWriter.Write(tmpPath, kv, tensors);
            File.Move(tmpPath, ggufPath, overwrite: true);
        }
        catch (OperationCanceledException)
        {
            TryDelete(tmpPath);
            throw;
        }
        catch
        {
            TryDelete(tmpPath);
            throw;
        }
        finally
        {
            progress?.Report(1f);
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { /* best effort */ }
    }

    private static List<GgufKvPair> BuildKvPairs(ModelMetaData meta, ModelConfig config, Tokenizer? tokenizer)
    {
        var kv = new List<GgufKvPair>();

        string arch = config.Architecture ?? "gpt2";
        kv.Add(new GgufKvPair { Key = "general.architecture", Value = arch });
        kv.Add(new GgufKvPair { Key = $"{arch}.embedding_length", Value = (uint)config.HiddenDim });
        kv.Add(new GgufKvPair { Key = $"{arch}.block_count", Value = (uint)config.NumLayers });
        kv.Add(new GgufKvPair { Key = $"{arch}.feed_forward_length", Value = (uint)config.FfnDim });
        kv.Add(new GgufKvPair { Key = $"{arch}.context_length", Value = (uint)config.MaxSeqLen });
        kv.Add(new GgufKvPair { Key = $"{arch}.attention.head_count", Value = (uint)config.NumHeads });
        kv.Add(new GgufKvPair { Key = $"{arch}.attention.head_count_kv", Value = (uint)config.NumKvHeads });
        kv.Add(new GgufKvPair { Key = $"{arch}.attention.layer_norm_rms_epsilon", Value = config.NormEps });
        kv.Add(new GgufKvPair { Key = $"{arch}.rope.freq_base", Value = config.RopeTheta });
        kv.Add(new GgufKvPair { Key = $"{arch}.vocab_size", Value = (uint)config.VocabSize });

        if (config.HeadDimOverride is { } headDim)
            kv.Add(new GgufKvPair { Key = $"{arch}.head_dim", Value = (uint)headDim });
        if (config.KeyLength is { } keyLen)
            kv.Add(new GgufKvPair { Key = $"{arch}.attention.key_length", Value = (uint)keyLen });
        if (config.ValueLength is { } valLen)
            kv.Add(new GgufKvPair { Key = $"{arch}.attention.value_length", Value = (uint)valLen });
        if (config.RopeDim is { } ropeDim)
            kv.Add(new GgufKvPair { Key = $"{arch}.rope.dimension_count", Value = (uint)ropeDim });
        if (config.RopeScalingType is { } scaling)
        {
            kv.Add(new GgufKvPair { Key = $"{arch}.rope.scaling.type", Value = scaling });
            if (config.RopeScalingFactor is { } factor)
                kv.Add(new GgufKvPair { Key = $"{arch}.rope.scaling.factor", Value = factor });
            if (config.RopeOriginalContextLength is { } origCtx)
                kv.Add(new GgufKvPair { Key = $"{arch}.rope.scaling.original_context_length", Value = (uint)origCtx });
            if (config.RopeLowFreqFactor is { } lowFreq)
                kv.Add(new GgufKvPair { Key = $"{arch}.rope.scaling.low_freq_factor", Value = lowFreq });
            if (config.RopeHighFreqFactor is { } highFreq)
                kv.Add(new GgufKvPair { Key = $"{arch}.rope.scaling.high_freq_factor", Value = highFreq });
        }
        if (config.NormTypeOverride is { } normType)
            kv.Add(new GgufKvPair { Key = $"{arch}.norm_type", Value = (uint)normType });
        if (config.TieWordEmbeddings is { } tie)
            kv.Add(new GgufKvPair { Key = $"{arch}.tie_word_embeddings", Value = tie });
        if (config.NumExperts > 0)
            kv.Add(new GgufKvPair { Key = $"{arch}.expert_count", Value = (uint)config.NumExperts });
        if (config.TopKExperts > 0)
            kv.Add(new GgufKvPair { Key = $"{arch}.expert_used_count", Value = (uint)config.TopKExperts });

        AddTokenizerKvPairs(kv, meta, tokenizer);

        return kv;
    }

    private static void AddTokenizerKvPairs(List<GgufKvPair> kv, ModelMetaData meta, Tokenizer? tokenizer)
    {
        string? template = meta.GetChatTemplate();
        if (!string.IsNullOrWhiteSpace(template))
            kv.Add(new GgufKvPair { Key = "tokenizer.chat_template", Value = template });

        if (tokenizer is null) return;

        var tokens = tokenizer.Vocab.AllTokens;
        kv.Add(new GgufKvPair { Key = "tokenizer.ggml.tokens", Value = tokens.ToArray() });
        kv.Add(new GgufKvPair { Key = "tokenizer.ggml.bos_token_id", Value = (uint)tokenizer.BosId });
        kv.Add(new GgufKvPair { Key = "tokenizer.ggml.eos_token_id", Value = (uint)tokenizer.EosId });

        // token_type: GGUF 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED,
        // 5=UNUSED, 6=BYTE. Rebuild from the native tokenizer JSON so specials
        // (unk/bos/eos/pad/additional) round-trip as CONTROL.
        string? tokenizerJson = meta.GetString(SmmConstants.TokenizerKey);
        var types = BuildTokenTypes(tokens.Count, tokenizerJson, tokenizer);
        kv.Add(new GgufKvPair { Key = "tokenizer.ggml.token_type", Value = types });

        // merges — from the embedded native tokenizer JSON ("left right" in rank order).
        string[]? merges = ParseMerges(tokenizerJson);
        if (merges is { Length: > 0 })
            kv.Add(new GgufKvPair { Key = "tokenizer.ggml.merges", Value = merges });
    }

    private static int[] BuildTokenTypes(int count, string? tokenizerJson, Tokenizer tokenizer)
    {
        var types = new int[count];
        var specials = new HashSet<string>(tokenizer.Specials.All, StringComparer.Ordinal);

        // GGUF marks the vocab's [unk, bos, eos, pad, additional] as CONTROL (3)
        // except the unknown slot, which GgufConverter resolves by name when it
        // is type 2. Rebuild faithfully so the round-trip tokenizer is equivalent.
        if (!string.IsNullOrEmpty(tokenizerJson))
        {
            try
            {
                var node = JsonNode.Parse(tokenizerJson)!;
                var st = node?["special_tokens"];
                if (st is not null)
                {
                    string? unk = st["unk"]?.GetValue<string>();
                    string? bos = st["bos"]?.GetValue<string>();
                    string? eos = st["eos"]?.GetValue<string>();
                    string? pad = st["pad"]?.GetValue<string>();
                    for (int i = 0; i < count; i++)
                    {
                        string tok = tokenizer.Vocab.AllTokens[i];
                        if (tok == unk)
                        {
                            types[i] = 2;
                            continue;
                        }
                        if (specials.Contains(tok))
                        {
                            types[i] = 3;
                            continue;
                        }
                        types[i] = 1;
                    }
                }
                return types;
            }
            catch
            {
                // fall through to the specials-only heuristic below
            }
        }

        for (int i = 0; i < count; i++)
            types[i] = specials.Contains(tokenizer.IdToToken(i)) ? 3 : 1;
        return types;
    }

    private static string[]? ParseMerges(string? tokenizerJson)
    {
        if (string.IsNullOrEmpty(tokenizerJson)) return null;
        try
        {
            var node = JsonNode.Parse(tokenizerJson)!;
            if (node["merges"] is not JsonArray arr || arr.Count == 0) return null;
            var list = new string[arr.Count];
            for (int i = 0; i < arr.Count; i++)
                list[i] = arr[i]!.GetValue<string>();
            return list;
        }
        catch
        {
            return null;
        }
    }

    private static List<GgufTensor> BuildTensors(string smmPath, ModelMetaData meta, CancellationToken ct, IProgress<float>? progress)
    {
        var entries = SmmLoader.ReadTensorIndex(smmPath);
        int totalTensors = 0;
        var tensors = new List<GgufTensor>(entries.Count);
        foreach (var entry in entries)
        {
            long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
            if (rawSize <= 0) continue;
            int ordinal = totalTensors++;
            tensors.Add(new GgufTensor
            {
                Name = entry.Name,
                Shape = entry.Shape,
                Dtype = entry.Dtype,
                GetBytes = () =>
                {
                    ct.ThrowIfCancellationRequested();
                    var bytes = SmmLoader.ReadTensorBytes(smmPath, entry, rawSize);
                    if (progress is not null && totalTensors > 0)
                        progress.Report((ordinal + 1f) / totalTensors);
                    return bytes;
                },
            });
        }
        return tensors;
    }
}