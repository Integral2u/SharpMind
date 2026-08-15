using SharpMind.CUI.App;
using SharpMind.Data;
using SharpMind.Data.Metadata;
using SharpMind.Data.Sources;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Verifies incremental-training delta logic: per-file delta computation feeds
/// restricted sources, unchanged sources are skipped, remote/non-file sources
/// stay whole, "nothing to train" is signalled, and the on-disk hash store
/// survives a round trip.
/// </summary>
public sealed class IncrementalTests
{
    private static readonly IReadOnlyList<ComponentDescriptor> Registry =
        ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly);

    private static TrainJobSettings JobWithTextSources(string pattern)
        => new()
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "Text",
                        TypeName = typeof(TextFileSource).AssemblyQualifiedName!,
                        Args = new Dictionary<string, string> { ["path"] = pattern },
                    },
                },
            },
            IncrementalMode = true,
        };

    private static TrainJobSettings JobWithNoPathSource()
        => new()
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "Pseudo",
                        TypeName = typeof(SharpMind.Data.Sources.PseudoLanguage.PseudoLanguageSource).AssemblyQualifiedName!,
                        Args = new Dictionary<string, string> { ["vocabSize"] = "5000", ["rootMorphemes"] = "300", ["affixes"] = "20", ["sequenceCount"] = "10000" },
                    },
                },
            },
            IncrementalMode = true,
        };

    [Fact]
    public void NonIncremental_BuildsAllSources()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "one\n");
        string b = dir.Write("b.txt", "two\n");
        var job = new TrainJobSettings
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "Text",
                        TypeName = typeof(TextFileSource).AssemblyQualifiedName!,
                        Args = new Dictionary<string, string> { ["path"] = dir.Path + "\\*.txt" },
                    },
                },
            },
        };

        var plan = IncrementalPlanner.Build(job, Registry, _ => { });

        Assert.False(plan.NothingToTrain);
        Assert.Single(plan.Sources);
        Assert.DoesNotContain(true, plan.SkipSource);
        _ = (a, b);
    }

    [Fact]
    public void Incremental_FirstRun_TrainsEverything()
    {
        using var dir = new TempDirectory();
        dir.Write("a.txt", "one\n");
        dir.Write("b.txt", "two\n");
        var job = JobWithTextSources(dir.Path + "\\*.txt");

        var plan = IncrementalPlanner.Build(job, Registry, _ => { });

        Assert.False(plan.NothingToTrain);
        Assert.Single(plan.Sources);
        Assert.DoesNotContain(true, plan.SkipSource);
        Assert.IsType<TextFileSource>(plan.Sources[0]);
    }

    [Fact]
    public void Incremental_NoChanges_NothingToTrain()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "one\n");

        // Hash the file as it stands, then record it as already-trained.
        var job = JobWithTextSources(dir.Path + "\\*.txt");
        var current = SourceHasher.ComputeFileHashes(job.Sources);
        job.SourceFileHashes = current;

        var plan = IncrementalPlanner.Build(job, Registry, _ => { });

        Assert.True(plan.NothingToTrain);
        Assert.Empty(plan.Sources);
        Assert.Contains(true, plan.SkipSource);
    }

    [Fact]
    public async Task Incremental_ChangedFile_TrainsOnlyDelta()
    {
        using var dir = new TempDirectory();
        dir.Write("unchanged.txt", "untouched line\n");
        dir.Write("changed.txt", "original line\n");

        var job = JobWithTextSources(dir.Path + "\\*.txt");
        var current = SourceHasher.ComputeFileHashes(job.Sources);
        job.SourceFileHashes = current;

        // Change one file — only that file should feed the delta source.
        File.WriteAllText(Path.Combine(dir.Path, "changed.txt"), "BRAND NEW LINE\n");
        var plan = IncrementalPlanner.Build(job, Registry, _ => { });

        Assert.False(plan.NothingToTrain);
        var text = Assert.IsType<TextFileSource>(plan.Sources[0]);
        var docs = await text.ReadAsync().ToListAsync();
        Assert.Single(docs);
        Assert.Contains("BRAND NEW LINE", docs[0], StringComparison.Ordinal);
        Assert.DoesNotContain(docs, d => d.Contains("untouched line", StringComparison.Ordinal));
        Assert.DoesNotContain(true, plan.SkipSource);
    }

    [Fact]
    public void Incremental_RemoteSource_AlwaysWhole()
    {
        var job = JobWithNoPathSource();
        var logs = new List<string>();

        var plan = IncrementalPlanner.Build(job, Registry, logs.Add);

        Assert.False(plan.NothingToTrain);
        Assert.Single(plan.Sources);
        Assert.Contains("retraining it whole", string.Join(" ", logs), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Incremental_RestrictedSource_CarriesModeArgument()
    {
        using var dir = new TempDirectory();
        dir.Write("a.txt", "one\n");
        var job = new TrainJobSettings
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "Text",
                        TypeName = typeof(TextFileSource).AssemblyQualifiedName!,
                        Args = new Dictionary<string, string>
                        {
                            ["path"] = dir.Path + "\\*.txt",
                            ["mode"] = "FilePerDoc",
                        },
                    },
                },
            },
            IncrementalMode = true,
        };

        var plan = IncrementalPlanner.Build(job, Registry, _ => { });

        var text = Assert.IsType<TextFileSource>(plan.Sources[0]);
        Assert.Contains("FilePerDoc", text.Description);
    }

    [Fact]
    public void IncrementalStore_RoundTripsHashes()
    {
        using var dir = new TempDirectory();
        var job = new TrainJobSettings
        {
            ExportPath = dir.Path,
            Name = "roundtrip",
            SourceFileHashes = new Dictionary<string, Dictionary<string, string>>
            {
                ["Text"] = new() { ["a.txt"] = new string('1', 64), ["b.txt"] = new string('2', 64) },
            },
        };

        IncrementalStore.Save(job);
        var loaded = IncrementalStore.Load(new TrainJobSettings { ExportPath = dir.Path, Name = "roundtrip" });

        Assert.Equal(2, loaded["Text"].Count);
        Assert.Equal(new string('1', 64), loaded["Text"]["a.txt"]);
    }

    [Fact]
    public void IncrementalStore_Load_MissingFile_ReturnsEmpty()
    {
        using var dir = new TempDirectory();
        var job = new TrainJobSettings { ExportPath = dir.Path, Name = "never-saved" };

        Assert.Empty(IncrementalStore.Load(job));
    }
}