namespace SharpMind.Tests.ModelFormat;

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using SharpMind.Model.Format;
using SharpMind.Model.Config;
using SharpMindConfig = SharpMind.Model.Format.SharpMindConfig;
using QuantConfig = SharpMind.Model.Format.QuantConfig;
using Dtype = SharpMind.Model.Format.Dtype;
using ModelConfig = SharpMind.Model.Config.ModelConfig;

public class SharpMindConfigTests
{
    private readonly string _testDir;

    public SharpMindConfigTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), $"SharpMindTests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDir);
    }

    [Fact]
    public void SaveAndLoad_Roundtrip()
    {
        var config = new SharpMindConfig
        {
            VocabSize = 32000,
            HiddenDim = 4096,
            NumLayers = 32,
            NumHeads = 32,
            NumKvHeads = 32,
            FfnDim = 11008,
            MaxSeqLen = 2048,
            RopeTheta = 10000f,
            Source = "llama-3-8b",
            Quantization = QuantConfig.Int8,
        };

        var configPath = Path.Combine(_testDir, "config.json");
        config.Save(configPath);

        Assert.True(File.Exists(configPath));

        var loaded = SharpMindConfig.Load(configPath);

        Assert.Equal(32000, loaded.VocabSize);
        Assert.Equal(4096, loaded.HiddenDim);
        Assert.Equal(32, loaded.NumLayers);
        Assert.Equal("llama-3-8b", loaded.Source);
        Assert.NotNull(loaded.Quantization);
        Assert.Equal(Dtype.INT8, loaded.Quantization.Dtype);
    }

    [Fact]
    public void FromModelConfig_Mapper_Configuration()
    {
        var modelConfig = new ModelConfig
        {
            VocabSize = 30000,
            HiddenDim = 2048,
            NumLayers = 24,
            NumHeads = 16,
            NumKvHeads = 8,
            FfnDim = 5632,
            MaxSeqLen = 4096,
            RopeTheta = 50000f,
        };

        var sharpConfig = SharpMindConfig.FromModelConfig(modelConfig, "custom-source");

        Assert.Equal(30000, sharpConfig.VocabSize);
        Assert.Equal(2048, sharpConfig.HiddenDim);
        Assert.Equal(24, sharpConfig.NumLayers);
        Assert.Equal(16, sharpConfig.NumHeads);
        Assert.Equal(8, sharpConfig.NumKvHeads);
        Assert.Equal(5632, sharpConfig.FfnDim);
        Assert.Equal(4096, sharpConfig.MaxSeqLen);
        Assert.Equal(50000f, sharpConfig.RopeTheta);
        Assert.Equal("custom-source", sharpConfig.Source);
    }

    [Fact]
    public void ToModelConfig_Roundtrip()
    {
        var sharpConfig = new SharpMindConfig
        {
            VocabSize = 20000,
            HiddenDim = 1024,
            NumLayers = 12,
            NumHeads = 8,
            NumKvHeads = 8,
            FfnDim = 2048,
            MaxSeqLen = 1024,
            RopeTheta = 8000f,
        };

        var modelConfig = sharpConfig.ToModelConfig();

        Assert.Equal(20000, modelConfig.VocabSize);
        Assert.Equal(1024, modelConfig.HiddenDim);
        Assert.Equal(12, modelConfig.NumLayers);
        Assert.Equal(8, modelConfig.NumHeads);
        Assert.Equal(8, modelConfig.NumKvHeads);
        Assert.Equal(2048, modelConfig.FfnDim);
        Assert.Equal(1024, modelConfig.MaxSeqLen);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Save_WithOptionalFields(bool includeQuantization)
    {
        var config = new SharpMindConfig
        {
            VocabSize = 1000,
            HiddenDim = 128,
            NumLayers = 2,
            NumHeads = 2,
            NumKvHeads = 2,
            FfnDim = 256,
            MaxSeqLen = 128,
        };

        if (includeQuantization)
        {
            config.Quantization = QuantConfig.Int4;
        }

        var configPath = Path.Combine(_testDir, "config2.json");
        config.Save(configPath);

        var loaded = SharpMindConfig.Load(configPath);
        Assert.Equal(1000, loaded.VocabSize);

        if (includeQuantization)
        {
            Assert.NotNull(loaded.Quantization);
            Assert.Equal(Dtype.INT4, loaded.Quantization.Dtype);
        }
        else
        {
            Assert.Null(loaded.Quantization);
        }
    }
}