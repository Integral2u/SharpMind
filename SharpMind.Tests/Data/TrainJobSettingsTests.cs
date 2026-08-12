using SharpMind.CUI.App;

namespace SharpMind.Tests.Data;

/// <summary>
/// Verifies <see cref="TrainJobSettings"/> JSON persistence: a full job with
/// multiple sources (each with per-source stages), global stages, QAT, keep
/// recent, export path, resume, and that the checkpoint directory is always
/// derived from the export path base plus "checkpoints-{job name}".
/// </summary>
public sealed class TrainJobSettingsTests
{
    [Fact]
    public void SaveLoad_RoundTripsFullJob()
    {
        using var dir = new TempDirectory();
        string path = dir.Write("job.smmt", "");
        File.Delete(path);

        var job = new TrainJobSettings
        {
            Name = "my-corpus",
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "Text File",
                        TypeName = "SharpMind.Data.Sources.TextFileSource, SharpMind.Data",
                        Args = new Dictionary<string, string> { ["path"] = "/data/a.txt", ["mode"] = "LinePerDoc" },
                    },
                    Stages =
                    {
                        new JobComponent
                        {
                            DisplayName = "Min Length Filter",
                            TypeName = "SharpMind.Data.Pipeline.Stages.MinLengthFilter, SharpMind.Data",
                            Args = new Dictionary<string, string> { ["minLength"] = "10" },
                        },
                    },
                },
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "JSONL",
                        TypeName = "SharpMind.Data.Sources.JsonlSource, SharpMind.Data",
                        Args = new Dictionary<string, string> { ["path"] = "/data/b.jsonl" },
                    },
                },
            },
            GlobalStages =
            {
                new JobComponent
                {
                    DisplayName = "Normalise Whitespace",
                    TypeName = "SharpMind.Data.Pipeline.Stages.NormaliseWhitespace, SharpMind.Data",
                    Args = new Dictionary<string, string>(),
                },
            },
            TokenizerVocabSize = 2048,
            HiddenDim = 256,
            NumLayers = 6,
            NumHeads = 8,
            NumKvHeads = 8,
            FfnDim = 1024,
            MaxSeqLen = 512,
            TotalSteps = 1000,
            SeqLen = 32,
            BatchSize = 4,
            GradAccumSteps = 2,
            LearningRate = 1e-3f,
            CheckpointInterval = 100,
            KeepRecent = 2,
            QuantAwareTraining = "Q8_0",
            ExportPath = "/tmp/model.smm",
            ResumeFrom = "/tmp/checkpoints-my-corpus/step-0000500",
        };

        Assert.True(job.Save(path, out var saveErr), saveErr);

        var loaded = TrainJobSettings.Load(path, out var loadErr);
        Assert.NotNull(loaded);
        Assert.Null(loadErr);

        Assert.Equal("my-corpus", loaded.Name);
        Assert.Equal(2, loaded.Sources.Count);
        Assert.Equal("Text File", loaded.Sources[0].Component.DisplayName);
        Assert.Equal("/data/a.txt", loaded.Sources[0].Component.Args["path"]);
        Assert.Single(loaded.Sources[0].Stages);
        Assert.Equal("Min Length Filter", loaded.Sources[0].Stages[0].DisplayName);
        Assert.Empty(loaded.Sources[1].Stages);
        Assert.Single(loaded.GlobalStages);
        Assert.Equal("Normalise Whitespace", loaded.GlobalStages[0].DisplayName);
        Assert.Equal(2048, loaded.TokenizerVocabSize);
        Assert.Equal(256, loaded.HiddenDim);
        Assert.Equal(6, loaded.NumLayers);
        Assert.Equal(1000, loaded.TotalSteps);
        Assert.Equal("Q8_0", loaded.QuantAwareTraining);
        Assert.Equal(2, loaded.KeepRecent);
        Assert.Equal("/tmp/model.smm", loaded.ExportPath);
        Assert.Equal(Path.Combine(Path.GetDirectoryName("/tmp/model.smm")!, "checkpoints-my-corpus"), loaded.CheckpointDir);
        Assert.Equal("/tmp/checkpoints-my-corpus/step-0000500", loaded.ResumeFrom);
    }

    [Fact]
    public void SaveLoad_NullListsAndDefaults()
    {
        using var dir = new TempDirectory();
        string path = Path.Combine(dir.Path, Guid.NewGuid().ToString("N") + TrainJobSettings.JobExtension);

        var job = new TrainJobSettings { Name = "empty-job" };
        Assert.True(job.Save(path, out var saveErr));

        var loaded = TrainJobSettings.Load(path, out var loadErr);
        Assert.NotNull(loaded);
        Assert.Null(loadErr);
        Assert.Empty(loaded.Sources);
        Assert.Empty(loaded.GlobalStages);
        Assert.Null(loaded.QuantAwareTraining);
        Assert.Equal(3, loaded.KeepRecent);
    }

    [Fact]
    public void CheckpointDir_IsDerivedFromExportPath()
    {
        var job = new TrainJobSettings { Name = "my-job", ExportPath = @"C:\models\my-job.smm" };
        Assert.Equal(@"C:\models\checkpoints-my-job", job.CheckpointDir);
    }

[Fact]
    public void CheckpointDir_FallsBackToDefaultFolder()
    {
        var job = new TrainJobSettings { Name = "no-export" };
        Assert.Equal(Path.Combine(TrainJobSettings.DefaultFolder, "checkpoints-no-export"), job.CheckpointDir);
    }

    [Fact]
    public void CheckpointDir_TreatsFolderExportPathAsBase()
    {
        var job = new TrainJobSettings { Name = "my-job", ExportPath = @"C:\temp" };
        Assert.Equal(@"C:\temp\checkpoints-my-job", job.CheckpointDir);
        Assert.Equal(@"C:\temp\my-job.smm", job.ExportFilePath);
    }

    [Fact]
    public void ExportFilePath_FileExportUsedVerbatim()
    {
        var job = new TrainJobSettings { Name = "my-job", ExportPath = @"C:\models\my-job.smm" };
        Assert.Equal(@"C:\models\my-job.smm", job.ExportFilePath);
    }

    [Fact]
    public void ListSaved_OnlyMatchesJobExtension()
    {
        using var dir = new TempDirectory();
        File.WriteAllText(Path.Combine(dir.Path, "job-a.smmt"), "{}");
        File.WriteAllText(Path.Combine(dir.Path, "job-a.tokenizer.json"), "{}");
        File.WriteAllText(Path.Combine(dir.Path, "job-b.smmt"), "{}");

        var saved = TrainJobSettings.ListSavedIn(dir.Path);

        Assert.Contains(saved, p => Path.GetFileName(p) == "job-a.smmt");
        Assert.Contains(saved, p => Path.GetFileName(p) == "job-b.smmt");
        Assert.DoesNotContain(saved, p => Path.GetFileName(p).EndsWith(".tokenizer.json"));
        Assert.DoesNotContain(saved, p => Path.GetFileName(p).EndsWith(".json"));
    }

    [Fact]
    public void ExportFilePath_BlankDefaultsUnderDefaultFolder()
    {
        var job = new TrainJobSettings { Name = "blank-export" };
        Assert.Equal(
            Path.Combine(TrainJobSettings.DefaultFolder, "blank-export", "blank-export.smm"),
            job.ExportFilePath);
    }

    [Fact]
    public void SaveLoad_ResumeFromCheckpointPreserved()
    {
        using var dir = new TempDirectory();
        string path = Path.Combine(dir.Path, "resume" + TrainJobSettings.JobExtension);

        var job = new TrainJobSettings
        {
            Name = "resume-job",
            ExportPath = Path.Combine(dir.Path, "resume-job.smm"),
            ResumeFrom = Path.Combine(dir.Path, "checkpoints-resume-job", "step-0000123"),
        };
        Assert.True(job.Save(path, out var err));

        var loaded = TrainJobSettings.Load(path, out var loadErr);
        Assert.NotNull(loaded);
        Assert.Equal(Path.Combine(dir.Path, "checkpoints-resume-job", "step-0000123"), loaded.ResumeFrom);
    }

    [Fact]
    public void SaveLoad_EmbedOptionsRoundTrip()
    {
        using var dir = new TempDirectory();
        string path = Path.Combine(dir.Path, "embed" + TrainJobSettings.JobExtension);

        var job = new TrainJobSettings
        {
            Name = "embed-job",
            SystemPromptPath = @"C:\prompts\assistant.md",
            SkillsFolder = @"C:\skills",
            PluginDllPaths = ["C:\\tools\\Weather.dll", "C:\\tools\\Calculator.dll"],
        };
        Assert.True(job.Save(path, out var saveErr), saveErr);

        var loaded = TrainJobSettings.Load(path, out var loadErr);
        Assert.NotNull(loaded);
        Assert.Equal(@"C:\prompts\assistant.md", loaded.SystemPromptPath);
        Assert.Equal(@"C:\skills", loaded.SkillsFolder);
        Assert.Equal(["C:\\tools\\Weather.dll", "C:\\tools\\Calculator.dll"], loaded.PluginDllPaths);
    }
}