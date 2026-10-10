using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Pins the two load-path behaviors ported from the GGUF loader into the shared
/// <see cref="ModelLoaderBase"/> so the .SMM container gets them too:
///
/// 1. A fused <c>attn_qkv</c> tensor is split into RawWq/RawWk/RawWv byte thirds.
///    SmmLoader previously routed the name to <c>SetRawField("RawWqkv", ...)</c>,
///    which has no case for that field — the Q/K/V bytes were silently dropped.
///
/// 2. A short-conv kernel tensor is transposed from the ggml channel-major
///    [l_cache, hidden] layout into the [tap][channel] row-major layout WScConv
///    and ApplyConv expect. The generic 2D branch mapped it into the wrong
///    positions, so the .SMM load produced a bogus kernel.
///
/// Both are asserted on the .SMM load directly and cross-checked against the
/// GGUF-converted twin so the two containers cannot drift.
/// </summary>
public class SmmPortParityTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    [Fact]
    public void FusedQkvInSmm_SplitsIntoRawQkWkAndV()
    {
        var config = ModelConfig.Learnable;
        int hidden = config.HiddenDim;           // 32
        int per = hidden * hidden;                // floats per Q/K/V part
        int partBytes = per * sizeof(float);

        byte[] q = RandomBytes(101, per);
        byte[] k = RandomBytes(202, per);
        byte[] v = RandomBytes(303, per);
        byte[] qkv = [.. q, .. k, .. v];

        string smmPath = WriteSmm("fused-qkv.smm", config, new SmmTensorData
        {
            Name = "blk.0.attn_qkv.weight",
            Shape = [hidden, 3 * hidden],
            Dtype = QuantDType.F32,
            GetBytes = () => qkv,
        });
        string ggufPath = ConvertToGguf(smmPath);

        using var smm = Load(smmPath, config);
        using var gguf = Load(ggufPath, config);

        foreach (var (tag, weights) in new[] { ("smm", smm), ("gguf", gguf) })
        {
            var block = weights.Blocks[0];

            // The fused bytes landed in the three split fields...
            Assert.Equal(q, block.RawWq);
            Assert.Equal(k, block.RawWk);
            Assert.Equal(v, block.RawWv);
            Assert.Equal(QuantDType.F32, block.QuantDtypeWq);

            // ...with per-part metadata the streaming forward path relies on
            // (its absence was the pre-port SMM bug: SetRawField("RawWqkv") stored
            // nothing, so RawWq/Wk/Wv stayed null and had no meta at all).
            Assert.True(block.TensorMeta.TryGetValue("RawWq", out var meta), $"{tag} lost RawWq meta");
            Assert.Equal(partBytes, meta.Size);
            Assert.Equal(QuantDType.F32, meta.Dtype);
        }

        // The two containers must agree byte-for-byte on the split and on the
        // per-part geometry (three contiguous thirds of partBytes each).
        Assert.Equal(smm.Blocks[0].RawWq, gguf.Blocks[0].RawWq);
        Assert.Equal(smm.Blocks[0].RawWk, gguf.Blocks[0].RawWk);
        Assert.Equal(smm.Blocks[0].RawWv, gguf.Blocks[0].RawWv);
        foreach (var weights in new[] { smm, gguf })
        {
            var meta = weights.Blocks[0].TensorMeta;
            Assert.Equal(partBytes, meta["RawWq"].Size);
            Assert.Equal(partBytes, meta["RawWk"].Size);
            Assert.Equal(partBytes, meta["RawWv"].Size);
            Assert.Equal(QuantDType.F32, meta["RawWq"].Dtype);
            Assert.Equal(partBytes, meta["RawWk"].Offset - meta["RawWq"].Offset);
            Assert.Equal(partBytes, meta["RawWv"].Offset - meta["RawWk"].Offset);
        }
    }

    [Theory]
    [InlineData(".smm")]
    [InlineData(".gguf")]
    public void ShortConvKernel_IsTransposedToTapChannelLayout(string format)
    {
        var config = ModelConfig.Learnable;
        int taps = config.ShortConvCacheLength;   // 3
        int chan = config.HiddenDim;              // 32

        // ggml stores the kernel channel-major: index = channel * taps + tap.
        float[] raw = new float[taps * chan];
        for (int i = 0; i < raw.Length; i++) raw[i] = i * 0.25f - 1f;
        byte[] rawBytes = new byte[raw.Length * sizeof(float)];
        Buffer.BlockCopy(raw, 0, rawBytes, 0, rawBytes.Length);

        // The conv needs it as [tap][channel]: target[tap * chan + channel].
        float[] expected = new float[taps * chan];
        for (int c = 0; c < chan; c++)
            for (int t = 0; t < taps; t++)
                expected[t * chan + c] = raw[c * taps + t];

        string smmPath = WriteSmm($"shortconv{format}.smm", config, new SmmTensorData
        {
            Name = "blk.0.shortconv.conv.weight",
            Shape = [taps, chan],
            Dtype = QuantDType.F32,
            GetBytes = () => rawBytes,
        });
        string path = format == ".gguf" ? ConvertToGguf(smmPath) : smmPath;

        using var weights = Load(path, config);

        var conv = weights.Blocks[0].WScConv;
        Assert.NotNull(conv);
        Assert.Equal(new TensorShape(taps, chan), conv.Shape);
        Assert.Equal(expected, conv.Data);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static byte[] RandomBytes(int seed, int floatCount)
    {
        var rng = new Random(seed);
        var floats = new float[floatCount];
        for (int i = 0; i < floats.Length; i++) floats[i] = (float)rng.NextDouble() - 0.5f;
        var bytes = new byte[floatCount * sizeof(float)];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private string WriteSmm(string file, ModelConfig config, params SmmTensorData[] tensors)
    {
        string path = Path.Combine(_temp.Path, file);
        SmmWriter.Write(path, config, null, null, tensors, new SmmWriteOptions { Source = "training" });
        return path;
    }

    private string ConvertToGguf(string smmPath)
    {
        string gguf = Path.ChangeExtension(smmPath, ".gguf");
        SmmToGufConverter.Convert(smmPath, gguf);
        return gguf;
    }

    private static TransformerWeights Load(string path, ModelConfig config)
    {
        var sharp = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var qOps = QuantizationFactory.Create(sharp.ResolvedHardware);
        var weights = ModelFactory.CreateWeights(config, sharp, qOps, path, LoadMode.Full);
        weights.InitializeWeights();
        return weights;
    }
}
