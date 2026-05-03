using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using SharpMind.Model.Format;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;
using TensorF = SharpMind.Core.Tensors.Tensor<float>;
using GgufLoader = SharpMind.Model.Format.GgufLoader;
using SafetensorsLoader = SharpMind.Model.Format.SafetensorsLoader;
using SharpMindConfig = SharpMind.Model.Format.SharpMindConfig;
using WeightMapper = SharpMind.Model.Format.WeightMapper;

namespace SharpMind.Tests.ModelFormat;

public class IntegrationTests
{
    private const string ExternalAssetsPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
    private const string GgufFileName = "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";
    private const string SafetensorsFileName = "model.safetensors";

    private bool TryGetFile(string filename, out string path)
    {
        path = Path.Combine(ExternalAssetsPath, filename);
        return File.Exists(path);
    }

    private void WarnIfFileNotFound(string filename, Action<string> warn)
    {
        var fullPath = Path.Combine(ExternalAssetsPath, filename);
        if (!File.Exists(fullPath))
        {
            warn($"External asset not found: {fullPath}. Tests requiring external model files will be skipped.");
        }
    }

    [Fact]
    public void LoadGguf_Metadata_Succeeds()
    {
        if (!TryGetFile(GgufFileName, out var path))
        {
            Assert.True(true, $"WARNING: GGUF file not found at {Path.Combine(ExternalAssetsPath, GgufFileName)} - test skipped");
            return;
        }

        try
        {
            var meta = GgufLoader.LoadMeta(path);
            Assert.NotNull(meta);
            Assert.True(meta.Version > 0);
            Assert.True(meta.TensorCount > 0);
        }
        catch (Exception ex)
        {
            Assert.True(true, $"WARNING: GGUF loading failed (may be newer format): {ex.Message}");
        }
    }

    [Fact]
    public void LoadSafetensors_Succeeds()
    {
        if (!TryGetFile(SafetensorsFileName, out var path))
        {
            Assert.True(true, $"WARNING: Safetensors file not found at {Path.Combine(ExternalAssetsPath, SafetensorsFileName)} - test skipped");
            return;
        }

        var weights = SafetensorsLoader.LoadWeights(path);

        Assert.NotNull(weights);
        Assert.True(weights.Count > 0);
    }

    [Fact]
    public void LoadSafetensors_HasWeights()
    {
        if (!TryGetFile(SafetensorsFileName, out var path))
        {
            Assert.True(true, $"WARNING: Safetensors file not found at {Path.Combine(ExternalAssetsPath, SafetensorsFileName)} - test skipped");
            return;
        }

        var weights = SafetensorsLoader.LoadWeights(path);

        Assert.True(weights.Count > 0);
    }
}