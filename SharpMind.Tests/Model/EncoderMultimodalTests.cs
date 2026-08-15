using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Encoders;

namespace SharpMind.Tests.Model;

/// <summary>
/// Tests for the B3 multimodal encoders (<see cref="VisionEncoder"/>,
/// <see cref="AudioEncoder"/>) and the fused <see cref="Transformer.ForwardMultimodal"/>
/// forward pass.
///
/// The encoders are verified against a plain-C# reference (patchify →
/// projection → positional embedding → RMS norm) and the transformer fuse path is
/// checked for: shape expansion ([vision | audio | text] prefix layouts), text-only
/// equivalence with the single-modality forward, deterministic behaviour for a
/// seeded RNG, and correct guarding when a modality is passed to a model that has
/// no encoder for it.
/// </summary>
public sealed class EncoderMultimodalTests
{
    private const int Patch = 4;
    private const int ImageSize = 8;          // 2×2 patches
    private const int Channels = 1;
    private const int NumPatches = 4;
    private const int PatchDim = Patch * Patch * Channels; // 16
    private const int MelBins = 5;
    private const int MaxFrames = 4;
    private const int HiddenDim = 8;

    private static ModelConfig EncoderConfig() => new()
    {
        VocabSize = 32,
        HiddenDim = HiddenDim,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 32,
        VisionPatchSize = Patch,
        VisionImageSize = ImageSize,
        VisionChannels = Channels,
        AudioMelBins = MelBins,
        AudioMaxFrames = MaxFrames,
    };

    private static Tensor<float> MakeImage(int batch, float seed)
    {
        var t = new Tensor<float>(batch, Channels, ImageSize, ImageSize);
        var d = t.Data;
        for (int i = 0; i < d.Length; i++)
            d[i] = seed + i * 0.01f;
        return t;
    }

    private static Tensor<float> MakeMel(int batch, int frames)
    {
        var t = new Tensor<float>(batch, frames, MelBins);
        var d = t.Data;
        for (int i = 0; i < d.Length; i++)
            d[i] = (i % 7) * 0.25f - 0.75f;
        return t;
    }

    // ---------- Vision -----------------

    [Fact]
    public void VisionEncoder_OutputShape_IsBatchByPatchesByHidden()
    {
        using var enc = new VisionEncoder(EncoderConfig(), new Random(1));
        using var image = MakeImage(2, 1.0f);
        using var output = enc.Forward(image);

        Assert.Equal(new[] { 2, NumPatches, HiddenDim }, output.Shape.Dims);
        Assert.Equal(NumPatches, enc.NumTokens);
        Assert.Equal(PatchDim, enc.InFeatures);
        Assert.Equal(HiddenDim, enc.HiddenDim);
    }

    [Fact]
    public void VisionEncoder_Forward_MatchesReferenceMath()
    {
        using var enc = new VisionEncoder(EncoderConfig(), new Random(7));
        var ps = enc.Parameters().ToArray();
        var proj = ps[0].Data;   // [patchDim, hidden]
        var pos = ps[1].Data;    // [patches, hidden]
        var normW = ps[2].Data;  // [hidden]

        // Patch indices for a B=1, 2×2 grid (i = patch row, j = patch col).
        using var image = MakeImage(1, 0.5f);
        using var output = enc.Forward(image);

        float eps = EncoderConfig().NormEps;
        int offset = 0;
        for (int p = 0; p < NumPatches; p++)
        {
            int i = p / 2, j = p % 2;
            var patch = new float[PatchDim];
            int k = 0;
            for (int c = 0; c < Channels; c++)
            for (int yy = 0; yy < Patch; yy++)
            for (int xx = 0; xx < Patch; xx++)
                patch[k++] = image[0, c, i * Patch + yy, j * Patch + xx];

            // projection: y[h] = sum_f p[f] * W[f, h]
            var unnormed = new float[HiddenDim];
            for (int h = 0; h < HiddenDim; h++)
            {
                float sum = 0;
                for (int f = 0; f < PatchDim; f++) sum += patch[f] * proj[f * HiddenDim + h];
                unnormed[h] = sum + pos[p * HiddenDim + h];
            }

            // RMS norm
            double sq = 0;
            for (int h = 0; h < HiddenDim; h++) sq += (double)unnormed[h] * unnormed[h];
            float inv = (float)(1.0 / Math.Sqrt(sq / HiddenDim + eps));
            for (int h = 0; h < HiddenDim; h++)
            {
                float exp = unnormed[h] * inv * normW[h];
                Assert.Equal(exp, output[0, p, h], precision: 4);
            }
        }
    }

    [Fact]
    public void VisionEncoder_RejectsWrongShapes()
    {
        using var enc = new VisionEncoder(EncoderConfig(), new Random(1));
        using var bad = new Tensor<float>(1, Channels, ImageSize, ImageSize / 2);
        Assert.Throws<ArgumentException>(() => enc.Forward(bad));
    }

    // ---------- Audio -----------------

    [Fact]
    public void AudioEncoder_OutputShape_IsBatchByFramesByHidden()
    {
        using var enc = new AudioEncoder(EncoderConfig(), new Random(2));
        using var mel = MakeMel(2, 3);
        using var output = enc.Forward(mel);

        Assert.Equal(new[] { 2, 3, HiddenDim }, output.Shape.Dims);
        Assert.Equal(MelBins, enc.InFeatures);
        Assert.Equal(MaxFrames, enc.MaxFrames);
        Assert.Equal(HiddenDim, enc.HiddenDim);
    }

    [Fact]
    public void AudioEncoder_Forward_MatchesReferenceMath()
    {
        using var enc = new AudioEncoder(EncoderConfig(), new Random(9));
        var ps = enc.Parameters().ToArray();
        var proj = ps[0].Data;  // [melBins, hidden]
        var pos = ps[1].Data;   // [maxFrames, hidden]

        using var mel = MakeMel(1, 2);
        using var output = enc.Forward(mel);

        float eps = EncoderConfig().NormEps;
        // Norm weights are all ones at construction.
        for (int f = 0; f < 2; f++)
        {
            var unnormed = new float[HiddenDim];
            for (int h = 0; h < HiddenDim; h++)
            {
                float sum = 0;
                for (int m = 0; m < MelBins; m++) sum += mel[0, f, m] * proj[m * HiddenDim + h];
                unnormed[h] = sum + pos[f * HiddenDim + h];
            }
            double sq = 0;
            for (int h = 0; h < HiddenDim; h++) sq += (double)unnormed[h] * unnormed[h];
            float inv = (float)(1.0 / Math.Sqrt(sq / HiddenDim + eps));
            for (int h = 0; h < HiddenDim; h++)
                Assert.Equal(unnormed[h] * inv, output[0, f, h], precision: 4);
        }
    }

    // ---------- Fused forward ----------

    private static (Transformer model, SharpMindConfig sc) BuildMultimodalModel(bool vision, bool audio)
    {
        var cfg = EncoderConfig();
        if (!vision) cfg = cfg with { VisionPatchSize = null, VisionImageSize = 0, VisionChannels = 0 };
        if (!audio) cfg = cfg with { AudioMelBins = null, AudioMaxFrames = 0 };
        var sc = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sc);
        var model = ModelFactory.CreateTrainingTransformer(weights, sc);
        return (model, sc);
    }

    [Fact]
    public void ForwardMultimodal_FusedShape_IsModalitiesPlusText()
    {
        var (model, _) = BuildMultimodalModel(vision: true, audio: true);
        using (model)
        using (var text = Tensor<int>.From([1, 2, 3, 4], 1, 4))
        using (var image = MakeImage(1, 1.0f))
        using (var mel = MakeMel(1, 2))
        {
            using var logits = model.ForwardMultimodal(text, image, mel);
            // vision(4) + audio(2) + text(4) = 10
            Assert.Equal(new[] { 1, 10, 32 }, logits.Shape.Dims);
        }
    }

    [Fact]
    public void ForwardMultimodal_WithNoModalities_MatchesTextOnlyForward()
    {
        var (model, _) = BuildMultimodalModel(vision: true, audio: true);
        using (model)
        using (var text = Tensor<int>.From([1, 2, 3], 1, 3))
        {
            using var fused = model.ForwardMultimodal(text, null, null);
            using var plain = model.Forward(text);
            Assert.Equal(plain.Shape.Dims, fused.Shape.Dims);
            for (int i = 0; i < fused.Data.Length; i++)
                Assert.Equal(plain.Data[i], fused.Data[i], precision: 5);
        }
    }

    [Fact]
    public void ForwardMultimodal_IsDeterministic_ForSeededRng()
    {
        var cfg = EncoderConfig(); // unevaluated; encoders constructed with same seed each model
        _ = cfg;
        var sc = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        float[] Run()
        {
            var weights = ModelFactory.CreateForTraining(EncoderConfig(), sc);
            using var model = ModelFactory.CreateTrainingTransformer(weights, sc);
            using var text = Tensor<int>.From([1, 2, 3, 4], 1, 4);
            using var image = MakeImage(1, 1.0f);
            using var mel = MakeMel(1, 2);
            using var logits = model.ForwardMultimodal(text, image, mel);
            return [.. logits.Data];
        }

        var a = Run();
        var b = Run();
        Assert.Equal(a, b);
    }

    [Fact]
    public void ForwardMultimodal_RejectsModalityWithoutEncoder()
    {
        var (textOnly, _) = BuildMultimodalModel(vision: false, audio: false);
        using (textOnly)
        using (var text = Tensor<int>.From([1, 2, 3], 1, 3))
        using (var image = MakeImage(1, 1.0f))
        {
            Assert.Throws<InvalidOperationException>(() => textOnly.ForwardMultimodal(text, image, null));
        }
    }

    [Fact]
    public void Transformer_Parameters_IncludesEncoderParameters()
    {
        var (model, _) = BuildMultimodalModel(vision: true, audio: true);
        using (model)
        {
            var names = model.Parameters().Select(p => p.Name).ToArray();
            Assert.Contains("vision.patch_proj.weight", names);
            Assert.Contains("vision.pos_embed", names);
            Assert.Contains("audio.frame_proj.weight", names);
            Assert.Contains("audio.pos_embed", names);
        }
    }
}