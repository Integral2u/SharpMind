namespace SharpMind.Tests.ModelFormat;

using System;
using System.IO;
using SharpMind.Model.Config;
using ModelConfig = SharpMind.Model.Config.ModelConfig;
using SharpMindModelConfig = SharpMind.Model.Format.SharpMindModelConfig;

/// <summary>
/// Synthetic tests for per-layer sliding-window attention configuration
/// (Gemma-3-style SWA + SWA RoPE base). No real model files are referenced.
/// </summary>
public class ModelConfigSwaTests
{
    private static ModelConfig Gemma3LikeConfig(int numLayers = 18) => new()
    {
        VocabSize = 256,
        HiddenDim = 128,
        NumLayers = numLayers,
        NumHeads = 4,
        NumKvHeads = 1,
        KeyLength = 32,
        ValueLength = 32,
        FfnDim = 512,
        MaxSeqLen = 2048,
        RopeTheta = 1_000_000f,
        RopeThetaSwa = 10_000f,
        SlidingWindowSize = 512,
        SlidingWindowPattern = 6,
    };

    [Fact]
    public void IsSwaLayer_Pattern6_MarksFullAttentionAtPeriodBoundary()
    {
        var config = Gemma3LikeConfig();

        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            bool expectedDense = layer % 6 == 5; // layers 5, 11, 17
            Assert.Equal(expectedDense, !config.IsSwaLayer(layer));
        }
    }

    [Fact]
    public void RopeThetaForLayer_DenseUsesBase_SwaUsesSwaBase()
    {
        var config = Gemma3LikeConfig();

        Assert.Equal(1_000_000f, config.RopeThetaForLayer(5));
        Assert.Equal(1_000_000f, config.RopeThetaForLayer(11));
        Assert.Equal(1_000_000f, config.RopeThetaForLayer(17));
        Assert.Equal(10_000f, config.RopeThetaForLayer(0));
        Assert.Equal(10_000f, config.RopeThetaForLayer(4));
        Assert.Equal(10_000f, config.RopeThetaForLayer(16));
    }

    [Fact]
    public void WindowSizeForLayer_DenseIsFull_SwaIsWindowed()
    {
        var config = Gemma3LikeConfig();

        Assert.Equal(0, config.WindowSizeForLayer(5));
        Assert.Equal(0, config.WindowSizeForLayer(11));
        Assert.Equal(0, config.WindowSizeForLayer(17));
        Assert.Equal(512, config.WindowSizeForLayer(0));
        Assert.Equal(512, config.WindowSizeForLayer(16));
    }

    [Fact]
    public void NoPattern_LegacyAllLayersAreSwa()
    {
        var config = Gemma3LikeConfig();
        config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 4,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            SlidingWindowSize = 512,
        };

        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            Assert.True(config.IsSwaLayer(layer));
            Assert.Equal(512, config.WindowSizeForLayer(layer));
        }
    }

    [Fact]
    public void NoRopeThetaSwa_SwaLayersFallBackToBaseTheta()
    {
        var config = Gemma3LikeConfig();
        config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 6,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            RopeTheta = 4321f,
            SlidingWindowSize = 256,
            SlidingWindowPattern = 6,
        };

        Assert.True(config.IsSwaLayer(0));
        Assert.Equal(4321f, config.RopeThetaForLayer(0));
        Assert.Equal(4321f, config.RopeThetaForLayer(5)); // dense layer still uses base
    }

    [Fact]
    public void Pattern1_AllLayersAreDense()
    {
        var config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 8,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            SlidingWindowSize = 256,
            SlidingWindowPattern = 1,
        };

        for (int layer = 0; layer < config.NumLayers; layer++)
            Assert.False(config.IsSwaLayer(layer));
    }

    [Fact]
    public void NoWindow_NoLayersAreSwa()
    {
        var config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 6,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            SlidingWindowPattern = 6,
        };

        Assert.False(config.IsSwaLayer(0));
        Assert.False(config.IsSwaLayer(5));
    }

    [Fact]
    public void Validate_RejectsPatternWithoutWindow()
    {
        var config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 6,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            SlidingWindowPattern = 6,
        };

        Assert.Throws<InvalidOperationException>(() => config.Validate());
    }

    [Fact]
    public void Validate_RejectsNonPositivePattern()
    {
        var config = new ModelConfig
        {
            VocabSize = 256,
            HiddenDim = 128,
            NumLayers = 6,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 512,
            MaxSeqLen = 1024,
            SlidingWindowSize = 256,
            SlidingWindowPattern = -1,
        };

        Assert.Throws<InvalidOperationException>(() => config.Validate());
    }

    [Fact]
    public void SmmConfig_MapsNewFields()
    {
        var modelConfig = Gemma3LikeConfig();

        var sharp = SharpMindModelConfig.FromModelConfig(modelConfig, "gemma3-270m");
        Assert.Equal(10_000f, sharp.RopeThetaSwa);
        Assert.Equal(512, sharp.SlidingWindowSize);
        Assert.Equal(6, sharp.SlidingWindowPattern);

        var roundTripped = sharp.ToModelConfig();
        Assert.Equal(10_000f, roundTripped.RopeThetaSwa);
        Assert.Equal(512, roundTripped.SlidingWindowSize);
        Assert.Equal(6, roundTripped.SlidingWindowPattern);
        Assert.Equal(1_000_000f, roundTripped.RopeTheta);
        Assert.True(roundTripped.IsSwaLayer(0));
        Assert.False(roundTripped.IsSwaLayer(5));
    }

    [Fact]
    public void SmmConfig_FileRoundtrip_PreservesSwaFields()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"SharpMindSwaTests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            var sharp = SharpMindModelConfig.FromModelConfig(Gemma3LikeConfig(), "gemma3-270m");
            var path = Path.Combine(dir, "config.json");
            sharp.Save(path);

            var loaded = SharpMindModelConfig.Load(path);
            Assert.Equal(10_000f, loaded.RopeThetaSwa);
            Assert.Equal(512, loaded.SlidingWindowSize);
            Assert.Equal(6, loaded.SlidingWindowPattern);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }
}