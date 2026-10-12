using System.Text;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// BERT-family encoders (bge, roberta, xlm-roberta) plot differently from the
/// decoder architectures the loader grew up with: learned absolute position
/// embeddings, LayerNorm with biases, a *post-residual* norm order
/// (x += attn(x); x = attn_output_norm(x); x += ffn(x); x = layer_output_norm(x))
/// and per-token segment embeddings. These tests pin that a synthetic
/// BERT GGUF loads with the right config, that the per-layer post-norms and
/// token-type table reach their targets, and that a streaming load reproduces
/// the full load's hidden states bit-for-bit.
/// </summary>
public sealed class BertEncoderTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    private const int Hidden = 32;
    private const int Vocab = 64;
    private const int Layers = 2;
    private const int Heads = 4;
    private const int Ffn = 64;
    private const int MaxSeq = 16;
    private const int NumTokenTypes = 2;

    [Fact]
    public void Meta_ReadsBertPostNormTokenTypesAndPooling()
    {
        string path = WriteBertGguf();
        var meta = GgufLoader.LoadMeta(path);
        var config = GgufLoader.LoadConfig(meta)!;

        Assert.Equal("bert", config.Architecture);
        Assert.Equal(Hidden, config.HiddenDim);
        Assert.Equal(Vocab, config.VocabSize);
        Assert.Equal(Layers, config.NumLayers);
        Assert.Equal(Ffn, config.FfnDim);
        Assert.Equal(PositionalEncoding.Learned, config.PositionalEncoding);
        Assert.True(config.PostNormResidual);
        Assert.Equal(NumTokenTypes, config.NumTokenTypes);
        Assert.True(config.HasTokenTypes);
        Assert.Equal(PoolingType.Cls, config.Pooling);

        var sharp = config.ForModel(HardwareTier.Scalar);
        Assert.Equal(NormKind.LayerNorm, sharp.Norm);
        Assert.Equal(ArchKind.Encoder, sharp.Arch);
        Assert.Equal(FfnKind.Dense, sharp.Ffn);
    }

    [Fact]
    public void StreamingLoad_EncodeHidden_MatchesFullLoad()
    {
        string path = WriteBertGguf();
        var config = GgufLoader.LoadConfig(GgufLoader.LoadMeta(path))!;
        var sharp = config.ForModel(HardwareTier.Scalar);
        var qOps = QuantizationFactory.Create(sharp.ResolvedHardware);

        using var fullWeights = ModelFactory.CreateWeights(config, sharp, qOps, path, LoadMode.Full);
        fullWeights.InitializeWeights();
        using var full = ModelFactory.CreateTransformer(fullWeights, sharp);

        using var streamWeights = ModelFactory.CreateWeights(config, sharp, qOps, path, LoadMode.Streaming);
        streamWeights.InitializeWeights();
        using var streaming = ModelFactory.CreateTransformer(streamWeights, sharp); // owns the weights

        // The BERT per-layer post-norms land on Norm1/Norm2 with biases, and the
        // segment table lands on the token-type embedding, for both load modes.
        Assert.All(fullWeights.Blocks, b =>
        {
            Assert.NotNull(b.Norm1W);
            Assert.NotNull(b.Norm1B);
            Assert.NotNull(b.Norm2W);
            Assert.NotNull(b.Norm2B);
        });
        Assert.NotNull(fullWeights.TokenTypeEmbedding);
        Assert.NotNull(streamWeights.TokenTypeEmbedding);

        using var tokens = new Tensor<int>(1, 6);
        int[] ids = [2, 5, 11, 23, 40, 63];
        ids.CopyTo(tokens.Data);

        using var a = streaming.EncodeHidden(tokens);
        using var b = full.EncodeHidden(tokens);

        // A streaming load fills each layer lazily as the forward reaches it, so the
        // weight comparison only makes sense after a full pass.
        Compare("position", fullWeights.PositionEmbedding, streamWeights.PositionEmbedding);
        Compare("token_types", fullWeights.TokenTypeEmbedding, streamWeights.TokenTypeEmbedding);
        Compare("finalW", fullWeights.FinalNormWeight, streamWeights.FinalNormWeight);
        Compare("finalB", fullWeights.FinalNormBias, streamWeights.FinalNormBias);
        for (int i = 0; i < Layers; i++)
        {
            Compare($"n1w{i}", fullWeights.Blocks[i].Norm1W, streamWeights.Blocks[i].Norm1W);
            Compare($"n1b{i}", fullWeights.Blocks[i].Norm1B, streamWeights.Blocks[i].Norm1B);
            Compare($"n2w{i}", fullWeights.Blocks[i].Norm2W, streamWeights.Blocks[i].Norm2W);
            Compare($"n2b{i}", fullWeights.Blocks[i].Norm2B, streamWeights.Blocks[i].Norm2B);
        }

        Assert.Equal(b.Shape.Dims.ToArray(), a.Shape.Dims.ToArray());
        Assert.Equal(b.ElementCount, a.ElementCount);

        float maxAbs = 0f;
        for (int i = 0; i < a.ElementCount; i++)
        {
            Assert.True(float.IsFinite(a.Data[i]), $"streaming hidden {i} not finite: {a.Data[i]}");
            Assert.True(BitConverter.SingleToInt32Bits(b.Data[i]) == BitConverter.SingleToInt32Bits(a.Data[i]),
                $"hidden {i}: full={b.Data[i]} streaming={a.Data[i]}");
            maxAbs = MathF.Max(maxAbs, MathF.Abs(a.Data[i]));
        }

        // Magnitude sanity: a degenerately-zero or exploding encode would pass the
        // bit-equality check above (both modes share the bug), so guard separately.
        Assert.True(maxAbs > 1e-3f, "encode produced an all-but-zero hidden state");
        Assert.True(maxAbs < 1e4f, $"encode magnitude exploded: {maxAbs}");

        // CLS pooling must return the first row of the hidden state, L2-normalised.
        using var pooled = streaming.PooledEmbedding(tokens, null, normalize: true);
        Assert.Equal(Hidden, pooled.ElementCount);
        var normed = MathF.Sqrt(pooled.Data.ToArray().Sum(v => v * v));
        Assert.InRange(normed, 0.999f, 1.001f);
    }

    private static void Compare(string label, Tensor<float>? a, Tensor<float>? b)
    {
        if (a is null || b is null)
        {
            Assert.True(a is null && b is null, $"{label}: full={a is not null} stream={b is not null}");
            return;
        }
        Assert.Equal(a.ElementCount, b.ElementCount);
        for (int i = 0; i < a.ElementCount; i++)
            Assert.True(a.Data[i] == b.Data[i], $"{label}[{i}]: full={a.Data[i]} stream={b.Data[i]}");
    }

    private string WriteBertGguf()
    {
        string path = Path.Combine(_temp.Path, $"bert-{Guid.NewGuid():N}.gguf");

        // GGUF stores the innermost dimension first (ne0). The loader flat-copies
        // 1D/2D global tensors, matching row-major [rows, cols] to ne = [cols, rows].
        var layerTensors = new List<(string Name, int[] Shape, byte[] Data)>();
        for (int i = 0; i < Layers; i++)
        {
            string b = $"blk.{i}.";
            layerTensors.Add(($"{b}attn_q.weight", [Hidden, Hidden], Floats(Hidden * Hidden, 100 + i)));
            layerTensors.Add(($"{b}attn_k.weight", [Hidden, Hidden], Floats(Hidden * Hidden, 200 + i)));
            layerTensors.Add(($"{b}attn_v.weight", [Hidden, Hidden], Floats(Hidden * Hidden, 300 + i)));
            layerTensors.Add(($"{b}attn_output.weight", [Hidden, Hidden], Floats(Hidden * Hidden, 400 + i)));
            layerTensors.Add(($"{b}attn_q.bias", [Hidden], Floats(Hidden, 500 + i)));
            layerTensors.Add(($"{b}attn_k.bias", [Hidden], Floats(Hidden, 510 + i)));
            layerTensors.Add(($"{b}attn_v.bias", [Hidden], Floats(Hidden, 520 + i)));
            layerTensors.Add(($"{b}attn_output.bias", [Hidden], Floats(Hidden, 530 + i)));
            layerTensors.Add(($"{b}attn_output_norm.weight", [Hidden], Positive(Hidden, 600 + i)));
            layerTensors.Add(($"{b}attn_output_norm.bias", [Hidden], Floats(Hidden, 610 + i)));
            layerTensors.Add(($"{b}ffn_up.weight", [Hidden, Ffn], Floats(Hidden * Ffn, 700 + i)));
            layerTensors.Add(($"{b}ffn_up.bias", [Ffn], Floats(Ffn, 800 + i)));
            layerTensors.Add(($"{b}ffn_down.weight", [Ffn, Hidden], Floats(Ffn * Hidden, 900 + i)));
            layerTensors.Add(($"{b}ffn_down.bias", [Hidden], Floats(Hidden, 950 + i)));
            layerTensors.Add(($"{b}layer_output_norm.weight", [Hidden], Positive(Hidden, 960 + i)));
            layerTensors.Add(($"{b}layer_output_norm.bias", [Hidden], Floats(Hidden, 970 + i)));
        }

        var tensors = new List<(string Name, int[] Shape, byte[] Data)>
        {
            ("token_embd.weight", [Hidden, Vocab], Floats(Hidden * Vocab, 1)),
            ("position_embd.weight", [Hidden, MaxSeq], Floats(Hidden * MaxSeq, 2)),
            ("token_embd_norm.weight", [Hidden], Positive(Hidden, 3)),
            ("token_embd_norm.bias", [Hidden], Floats(Hidden, 4)),
            ("token_types.weight", [Hidden, NumTokenTypes], Floats(Hidden * NumTokenTypes, 5)),
        };
        tensors.AddRange(layerTensors);

        using var fs = new FileStream(path, FileMode.Create);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(0x46554747u); // "GGUF"
        w.Write(3u);          // version
        w.Write((ulong)tensors.Count);
        w.Write(12ul);        // kv_count

        void KvString(string key, string value) { WriteGgufString(w, key); w.Write(8u); WriteGgufString(w, value); }
        void KvU32(string key, uint value) { WriteGgufString(w, key); w.Write(4u); w.Write(value); }
        void KvF32(string key, float value) { WriteGgufString(w, key); w.Write(6u); w.Write(value); }
        void KvStringArray(string key, string[] values)
        {
            WriteGgufString(w, key);
            w.Write(9u);
            w.Write(8u);
            w.Write((ulong)values.Length);
            foreach (string v in values) WriteGgufString(w, v);
        }

        KvString("general.architecture", "bert");
        KvU32("bert.embedding_length", Hidden);
        KvU32("bert.block_count", Layers);
        KvU32("bert.feed_forward_length", Ffn);
        KvU32("bert.context_length", MaxSeq);
        KvU32("bert.attention.head_count", Heads);
        KvU32("bert.attention.head_count_kv", Heads);
        KvF32("bert.attention.layer_norm_epsilon", 1e-5f);
        KvU32("bert.pooling_type", (uint)PoolingType.Cls);
        KvString("tokenizer.ggml.model", "bert");
        KvU32("tokenizer.ggml.token_type_count", NumTokenTypes);
        KvStringArray("tokenizer.ggml.tokens", Enumerable.Range(0, Vocab).Select(i => $"<t{i}>").ToArray());

        var offsets = new long[tensors.Count];
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            offsets[i] = offset;
            offset += tensors[i].Data.Length;
        }

        for (int i = 0; i < tensors.Count; i++)
        {
            var (name, shape, _) = tensors[i];
            WriteGgufString(w, name);
            w.Write((uint)shape.Length);
            foreach (int d in shape) w.Write((ulong)d);
            w.Write(0u); // dtype: F32
            w.Write((ulong)offsets[i]);
        }

        long end = fs.Position;
        long aligned = (end + 31) & ~31L;
        if (aligned > end) fs.Write(new byte[aligned - end]);

        foreach (var (_, _, data) in tensors) fs.Write(data, 0, data.Length);
        return path;
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static byte[] Floats(int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = (float)(rng.NextDouble() * 2 - 1);
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] Positive(int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = 0.5f + (float)rng.NextDouble();
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
