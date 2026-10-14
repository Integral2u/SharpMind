using System.Text;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Gemma-4 carries a global matmul weight, <c>per_layer_model_proj</c>
/// (<c>[n_embd, n_embd_per_layer * n_layer]</c>), that the per-layer-embedding
/// builder feeds through a <see cref="SharpMind.Model.Layers.TrainingLinearLayer"/>
/// as <c>[inFeatures, outFeatures]</c>. Unlike the position/token-type tables
/// (declared with their SharpMind dims reversed relative to GGUF so a verbatim
/// copy lands correctly), this global is declared in GGUF order and so needs the
/// same [in, out] transpose the block F32 path applies. When it was copied
/// verbatim the projection was applied as Wᵀ and every block's per-layer input
/// was corrupted, so the whole model emitted garbage. This pins the loaded
/// orientation element-by-element.
/// </summary>
public sealed class Gemma4WeightLoadTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    private const int Hidden = 16;
    private const int Vocab = 32;
    private const int Layers = 2;
    private const int Heads = 4;
    private const int KvHeads = 2;
    private const int Ffn = 32;
    private const int PerLayerDim = 8;
    private const int TotalDim = PerLayerDim * Layers;

    [Fact]
    public void PerLayerModelProj_LoadsTransposedIntoInOutOrder()
    {
        string path = WriteGemma4Gguf();
        var config = GgufLoader.LoadConfig(GgufLoader.LoadMeta(path))!;
        var sharp = config.ForModel(HardwareTier.Scalar);
        Assert.Equal(ArchKind.Decoder, sharp.Arch);

        var qOps = QuantizationFactory.Create(sharp.ResolvedHardware);
        using var weights = ModelFactory.CreateWeights(config, sharp, qOps, path, LoadMode.Full);
        weights.InitializeWeights();

        var proj = weights.PerLayerModelProj;
        Assert.NotNull(proj);
        Assert.Equal(Hidden, proj!.Shape[0]);
        Assert.Equal(TotalDim, proj.Shape[1]);

        // The file stores ggml [ne0 = Hidden, ne1 = Total], i.e. element (i, o) at
        // file[o * Hidden + i]. The layer wants [in, out] row-major, element
        // (i, o) at Data[i * Total + o]. Assert the transpose landed.
        for (int i = 0; i < Hidden; i++)
            for (int o = 0; o < TotalDim; o++)
                Assert.Equal(Value(i, o), proj.Data[i * TotalDim + o]);
    }

    private static float Value(int i, int o) => (i + 1) + (o + 1) * 10_000f;

    private string WriteGemma4Gguf()
    {
        string path = Path.Combine(_temp.Path, $"gemma4-{Guid.NewGuid():N}.gguf");

        // per_layer_model_proj is written in ggml order: file[o * Hidden + i] = Value(i, o).
        var projData = new float[Hidden * TotalDim];
        for (int o = 0; o < TotalDim; o++)
            for (int i = 0; i < Hidden; i++)
                projData[o * Hidden + i] = Value(i, o);

        var tensors = new List<(string Name, int[] Shape, byte[] Data)>
        {
            ("token_embd.weight", [Hidden, Vocab], Floats(Hidden * Vocab, 1)),
            ("output_norm.weight", [Hidden], Positive(Hidden, 2)),
            ("per_layer_model_proj.weight", [Hidden, TotalDim], ToBytes(projData)),
            ("per_layer_proj_norm.weight", [PerLayerDim], Positive(PerLayerDim, 3)),
        };

        using var fs = new FileStream(path, FileMode.Create);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(0x46554747u); // "GGUF"
        w.Write(3u);          // version
        w.Write((ulong)tensors.Count);
        w.Write(14ul);        // kv_count

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

        KvString("general.architecture", "gemma4");
        KvU32("gemma4.embedding_length", Hidden);
        KvU32("gemma4.block_count", Layers);
        KvU32("gemma4.feed_forward_length", Ffn);
        KvU32("gemma4.context_length", 64);
        KvU32("gemma4.attention.head_count", Heads);
        KvU32("gemma4.attention.head_count_kv", KvHeads);
        KvU32("gemma4.attention.key_length", Hidden);
        KvU32("gemma4.attention.value_length", Hidden);
        KvF32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
        KvU32("gemma4.embedding_length_per_layer_input", PerLayerDim);
        KvF32("gemma4.rope.freq_base", 1_000_000f);
        KvString("tokenizer.ggml.model", "llama");
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

    private static byte[] ToBytes(float[] floats)
    {
        var bytes = new byte[floats.Length * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] Floats(int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = (float)(rng.NextDouble() * 2 - 1);
        return ToBytes(floats);
    }

    private static byte[] Positive(int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = 0.5f + (float)rng.NextDouble();
        return ToBytes(floats);
    }
}
