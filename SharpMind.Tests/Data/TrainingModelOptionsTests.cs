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

    [Fact]
    public void QatLabels_IncludeLegacyBlockFormats()
    {
        Assert.Contains("F16", TrainingModelOptions.QatLabels);
        Assert.Contains("Q8_0", TrainingModelOptions.QatLabels);
        Assert.Contains("Q4_0", TrainingModelOptions.QatLabels);
        Assert.Contains("Q8_K", TrainingModelOptions.QatLabels);
        Assert.Contains("Q2_K", TrainingModelOptions.QatLabels);
    }

    [Theory]
    [InlineData(0)]   // off
    [InlineData(1)]   // F16
    [InlineData(2)]   // Q8_0
    [InlineData(3)]   // Q4_0
    [InlineData(4)]   // Q8_K
    [InlineData(9)]   // Q2_K
    public void QatIndex_AndStoredName_RoundTrip(int index)
    {
        string? stored = TrainingModelOptions.QatStoredName(index);
        Assert.Equal(index, TrainingModelOptions.QatIndexFor(stored));
    }

    [Fact]
    public void QatIndexFor_LegacyNames_AreNotRemapped()
    {
        // The old mapping collapsed a stored Q8_0 job onto Q8_K (and Q4_0 onto Q4_K)
        // and rewrote it on save. Each legacy name must now keep its own entry.
        Assert.Equal("Q8_0", TrainingModelOptions.QatStoredName(TrainingModelOptions.QatIndexFor("Q8_0")));
        Assert.Equal("Q4_0", TrainingModelOptions.QatStoredName(TrainingModelOptions.QatIndexFor("Q4_0")));
        Assert.Equal("Q8_K", TrainingModelOptions.QatStoredName(TrainingModelOptions.QatIndexFor("Q8_K")));
        Assert.NotEqual(TrainingModelOptions.QatIndexFor("Q8_K"), TrainingModelOptions.QatIndexFor("Q8_0"));
    }

    [Fact]
    public void QatIndexFor_OffAndUnknown_MapToOff()
    {
        Assert.Equal(0, TrainingModelOptions.QatIndexFor(null));
        Assert.Equal(0, TrainingModelOptions.QatIndexFor(""));
        Assert.Equal(0, TrainingModelOptions.QatIndexFor("F32"));
        Assert.Equal(0, TrainingModelOptions.QatIndexFor("nonsense"));
        Assert.Null(TrainingModelOptions.QatStoredName(0));
    }

    [Fact]
    public void ValidateQat_F16AndOff_AlwaysPass()
    {
        Assert.Null(TrainingModelOptions.ValidateQat(null, hiddenDim: 1));
        Assert.Null(TrainingModelOptions.ValidateQat("F32", hiddenDim: 1));
        Assert.Null(TrainingModelOptions.ValidateQat("F16", hiddenDim: 1));
    }

    [Fact]
    public void ValidateQat_BlockFormats_NeedHiddenDimMultipleOf32()
    {
        Assert.Null(TrainingModelOptions.ValidateQat("Q8_0", hiddenDim: 64));
        Assert.Null(TrainingModelOptions.ValidateQat("Q4_0", hiddenDim: 128));
        Assert.NotNull(TrainingModelOptions.ValidateQat("Q8_0", hiddenDim: 48));
    }

    [Fact]
    public void ValidateQat_KQuants_NeedHiddenDimMultipleOf128()
    {
        Assert.Null(TrainingModelOptions.ValidateQat("Q6_K", hiddenDim: 128));
        Assert.NotNull(TrainingModelOptions.ValidateQat("Q6_K", hiddenDim: 64));
        Assert.NotNull(TrainingModelOptions.ValidateQat("Q2_K", hiddenDim: 96));
    }

    [Fact]
    public void ValidateQat_UnsupportedTarget_ReportsIt()
    {
        Assert.NotNull(TrainingModelOptions.ValidateQat("Q4_1", hiddenDim: 128));
        Assert.NotNull(TrainingModelOptions.ValidateQat("BF16", hiddenDim: 128));
    }

    [Theory]
    [InlineData(1000, 1024)]
    [InlineData(1024, 1024)]
    [InlineData(1, 32)]
    public void QatPaddedVocab_BlockFormats_RoundUpTo32(int input, int expected)
    {
        Assert.Equal(expected, TrainingModelOptions.QatPaddedVocab(
            TrainingModelOptions.ParseQat("Q8_0"), input, hiddenDim: 64));
    }

    [Fact]
    public void QatPaddedVocab_KQuants_RoundUpToBlockTightVocab()
    {
        // H = 128 -> gcd(256,128)=128 -> step 2; H = 384 -> step 2; H = 256 -> step 1.
        Assert.Equal(1002, TrainingModelOptions.QatPaddedVocab(TrainingModelOptions.ParseQat("Q6_K"), 1001, 128));
        Assert.Equal(1000, TrainingModelOptions.QatPaddedVocab(TrainingModelOptions.ParseQat("Q6_K"), 1000, 384));
        Assert.Equal(1001, TrainingModelOptions.QatPaddedVocab(TrainingModelOptions.ParseQat("Q6_K"), 1001, 256));
    }

    [Fact]
    public void QatPaddedVocab_OffAndF16_LeaveVocabUnchanged()
    {
        Assert.Equal(1001, TrainingModelOptions.QatPaddedVocab(null, 1001, 64));
        Assert.Equal(1001, TrainingModelOptions.QatPaddedVocab(TrainingModelOptions.ParseQat("F16"), 1001, 64));
    }

    [Fact]
    public void ResolveModelConfig_PadsVocabOnlyUnderQat()
    {
        var job = new TrainJobSettings();
        TrainingModelOptions.Apply(job, TrainingModelOptions.Presets.Single(p => p.ArchitecturePresetKey == "bert"));

        Assert.Equal(1000, TrainingModelOptions.ResolveModelConfig(job, 1000).VocabSize);

        job.QuantAwareTraining = "Q8_0";
        Assert.Equal(1024, TrainingModelOptions.ResolveModelConfig(job, 1000).VocabSize);
    }
}