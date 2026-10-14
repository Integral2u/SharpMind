using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Gemma 3n/4 metadata: dual attention geometry (full-attention layers use a
/// larger head dim and rotary span than windowed ones), per-layer input
/// embeddings, KV sharing across trailing layers, no attention pre-scaling and
/// final logit soft-capping. These tests pin the parsing of every new key and
/// the per-layer derived helpers, built from an in-memory metadata table (no
/// file I/O, no weights).
/// </summary>
public sealed class Gemma4ConfigTests
{
    private static ModelMetaData Gemma4Meta() => new()
    {
        KvPairs =
        [
            Kv("general.architecture", "gemma4"),
            Kv("gemma4.block_count", (uint)42),
            Kv("gemma4.context_length", (uint)131072),
            Kv("gemma4.embedding_length", (uint)2560),
            Kv("gemma4.feed_forward_length", (uint)10240),
            Kv("gemma4.embedding_length_per_layer_input", (uint)256),
            Kv("gemma4.attention.head_count", (uint)8),
            Kv("gemma4.attention.head_count_kv", (uint)2),
            Kv("gemma4.attention.key_length", (uint)512),
            Kv("gemma4.attention.value_length", (uint)512),
            Kv("gemma4.attention.key_length_swa", (uint)256),
            Kv("gemma4.attention.value_length_swa", (uint)256),
            Kv("gemma4.attention.layer_norm_rms_epsilon", 1e-6f),
            Kv("gemma4.attention.shared_kv_layers", (uint)18),
            Kv("gemma4.attention.sliding_window", (uint)512),
            Kv("gemma4.rope.dimension_count", (uint)512),
            Kv("gemma4.rope.dimension_count_swa", (uint)256),
            Kv("gemma4.rope.freq_base", 1_000_000f),
            Kv("gemma4.rope.freq_base_swa", 10_000f),
            Kv("gemma4.final_logit_softcapping", 30f),
        ],
        Tensors =
        [
            new TensorInfo { Name = "token_embd.weight", Dtype = QuantDType.Q4_K, Shape = [262144, 2560], Offset = 0 },
        ],
    };

    private static KvPair Kv(string key, object value) => new() { Key = key, Value = value };

    [Fact]
    public void Config_ParsesGemma4ScalarsAndGeometry()
    {
        var config = GgufLoader.LoadConfig(Gemma4Meta())!;

        Assert.Equal("gemma4", config.Architecture);
        Assert.Equal(262144, config.VocabSize);
        Assert.Equal(2560, config.HiddenDim);
        Assert.Equal(42, config.NumLayers);
        Assert.Equal(8, config.NumHeads);
        Assert.Equal(2, config.NumKvHeads);
        Assert.Equal(10240, config.FfnDim);

        Assert.Equal(512, config.KeyLength);
        Assert.Equal(512, config.ValueLength);
        Assert.Equal(512, config.HeadDim);
        Assert.Equal(256, config.KeyLengthSwa);
        Assert.Equal(256, config.ValueLengthSwa);
        Assert.Equal(512, config.RopeDim);
        Assert.Equal(256, config.RopeDimSwa);

        Assert.Equal(1_000_000f, config.RopeTheta);
        Assert.Equal(10_000f, config.RopeThetaSwa);
        Assert.Equal(512, config.SlidingWindowSize);
        Assert.Equal(6, config.SlidingWindowPattern);
        Assert.True(config.IsHybridSlidingWindow);

        Assert.Equal(256, config.PerLayerEmbdDim);
        Assert.True(config.HasPerLayerEmbeddings);
        Assert.Equal(18, config.SharedKvLayers);
        Assert.True(config.HasSharedKv);
        Assert.Equal(30f, config.FinalLogitSoftcap);
        Assert.True(config.HasFinalLogitSoftcap);
        Assert.Equal(1.0f, config.AttentionScale);
    }

    [Fact]
    public void Config_DerivesPerLayerGeometry()
    {
        var config = GgufLoader.LoadConfig(Gemma4Meta())!;

        // Full-attention layers are il % 6 == 5 (5, 11, ..., 41); the rest are windowed.
        Assert.True(config.IsSwaLayer(0));
        Assert.False(config.IsSwaLayer(5));
        Assert.False(config.IsSwaLayer(41));

        Assert.Equal(256, config.HeadDimForLayer(0));
        Assert.Equal(512, config.HeadDimForLayer(5));
        Assert.Equal(256, config.RopeDimForLayer(0));
        Assert.Equal(512, config.RopeDimForLayer(5));

        // Trailing 18 layers (24..41) reuse earlier K/V; layers 0..23 own theirs.
        Assert.Equal(24, config.FirstKvSharedLayer);
        Assert.True(config.HasOwnKv(0));
        Assert.True(config.HasOwnKv(23));
        Assert.False(config.HasOwnKv(24));
        Assert.False(config.HasOwnKv(41));

        // Every shared layer reads the last owning layer of its own attention
        // type: windowed -> 22, full -> 23. This reproduces llama.cpp's reuse map
        // `n_layer_kv_from_start - (is_swa(il) ? 2 : 1)` exactly.
        Assert.Equal(22, config.KvSourceLayer(22));
        Assert.Equal(23, config.KvSourceLayer(23));
        Assert.Equal(22, config.KvSourceLayer(24));
        Assert.Equal(22, config.KvSourceLayer(28));
        Assert.Equal(23, config.KvSourceLayer(29));
        Assert.Equal(22, config.KvSourceLayer(40));
        Assert.Equal(23, config.KvSourceLayer(41));
    }
}
