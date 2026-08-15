using SharpMind.CUI.App;
using SharpMind.Core;
using SharpMind.Model.Config;

namespace SharpMind.Tests.Data;

/// <summary>
/// Verifies the B4 architecture/optimizer options: the preset catalog, preset
/// application onto a job, resolution into <see cref="SharpMindConfig"/> and
/// <see cref="ModelConfig"/>, and that jobs without the new fields keep the
/// historic GPT-2-style behaviour.
/// </summary>
public sealed class TrainingModelOptionsTests
{
    [Fact]
    public void PresetCatalog_HasExpectedFamilies()
    {
        var keys = TrainingModelOptions.Presets.Select(p => p.ArchitecturePresetKey).ToArray();
        Assert.Contains("gpt2", keys);
        Assert.Contains("llama", keys);
        Assert.Contains("bert", keys);
        Assert.Contains("qwen3", keys);
        Assert.Contains("mixtral", keys);
    }

    [Fact]
    public void Apply_Gpt2FillsDimsAndOptions()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets[0]);

        Assert.Equal("gpt2", job.ArchitecturePreset);
        Assert.Equal("GELU", job.Activation);
        Assert.Equal("None", job.Gate);
        Assert.Equal("Dense", job.Ffn);
        Assert.Equal("LayerNorm", job.Norm);
        Assert.Equal("Decoder", job.Arch);
        Assert.Equal("MHA", job.Attention);
        Assert.Equal("RoPE", job.PositionalEncoding);
        Assert.Equal(128, job.HiddenDim);
        Assert.Equal(8, job.NumHeads);
    }

    [Fact]
    public void Apply_MixtralFillsMoE()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets.Single(p => p.ArchitecturePresetKey == "mixtral"));

        Assert.Equal("MoE", job.Ffn);
        Assert.Equal(8, job.NumExperts);
        Assert.Equal(2, job.TopKExperts);
        Assert.Equal("RMSNorm", job.Norm);
    }

    [Fact]
    public void ResolveSharpConfig_NoPreset_MatchesHistoricGpt()
    {
        var job = new TrainJobSettings();
        var cfg = TrainingModelOptions.ResolveSharpConfig(job);

        Assert.Equal(ActivationKind.GELU, cfg.Activation);
        Assert.Equal(GateKind.None, cfg.Gate);
        Assert.Equal(FfnKind.Dense, cfg.Ffn);
        Assert.Equal(NormKind.LayerNorm, cfg.Norm);
        Assert.Equal(ArchKind.Decoder, cfg.Arch);
        Assert.Equal(HardwareTier.Auto, cfg.Hardware);
    }

    [Fact]
    public void ResolveSharpConfig_Preset_MatchesPresetFamily()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets.Single(p => p.ArchitecturePresetKey == "llama"));
        var cfg = TrainingModelOptions.ResolveSharpConfig(job);

        Assert.Equal(ActivationKind.SiLU, cfg.Activation);
        Assert.Equal(GateKind.SwiGLU, cfg.Gate);
        Assert.Equal(FfnKind.Gated, cfg.Ffn);
        Assert.Equal(NormKind.RMSNorm, cfg.Norm);
        Assert.Equal(ArchKind.Decoder, cfg.Arch);
    }

    [Fact]
    public void ResolveSharpConfig_Mixtral_IsMoE()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets.Single(p => p.ArchitecturePresetKey == "mixtral"));
        var cfg = TrainingModelOptions.ResolveSharpConfig(job);

        Assert.Equal(FfnKind.MoE, cfg.Ffn);
        Assert.Equal(ActivationKind.SiLU, cfg.Activation);
    }

    [Fact]
    public void ResolveSharpConfig_ExplicitOverridesWinOverPreset()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets[0]); // gpt2 → LayerNorm
        job.Norm = "RMSNorm";
        job.Activation = "ReLU";

        var cfg = TrainingModelOptions.ResolveSharpConfig(job);
        Assert.Equal(NormKind.RMSNorm, cfg.Norm);
        Assert.Equal(ActivationKind.ReLU, cfg.Activation);
        Assert.Equal(FfnKind.Dense, cfg.Ffn); // untouched preset field survives
    }

    [Fact]
    public void ResolveModelConfig_CarriesDimensionsAndMoE()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets.Single(p => p.ArchitecturePresetKey == "bert"));

        var cfg = TrainingModelOptions.ResolveModelConfig(job, vocabSize: 1024);

        Assert.Equal(1024, cfg.VocabSize);
        Assert.Equal(128, cfg.HiddenDim);
        Assert.Equal(4, cfg.NumLayers);
        Assert.Equal(8, cfg.NumHeads);
        Assert.Equal(8, cfg.NumKvHeads);
        Assert.Equal(512, cfg.FfnDim);
        Assert.Equal(256, cfg.MaxSeqLen);
        Assert.Equal(PositionalEncoding.NoPE, cfg.PositionalEncoding);
    }

    [Fact]
    public void UsesSgd_OnlyWhenExplicitlySgd()
    {
        Assert.True(TrainingModelOptions.UsesSgd(new TrainJobSettings { Optimizer = "SGD" }));
        Assert.True(TrainingModelOptions.UsesSgd(new TrainJobSettings { Optimizer = "sgd" }));
        Assert.False(TrainingModelOptions.UsesSgd(new TrainJobSettings()));
        Assert.False(TrainingModelOptions.UsesSgd(new TrainJobSettings { Optimizer = "AdamW" }));
    }
}