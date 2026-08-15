using SharpMind.CUI.App;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Verifies <see cref="SourceHasher"/>: per-source fingerprints are stable for
/// identical inputs, change when content changes, and skip non-file sources.
/// </summary>
public sealed class SourceHasherTests
{
    private static TrainJobSettings JobWithTextFile(string path, string displayName = "Text File")
        => new()
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = displayName,
                        TypeName = "SharpMind.Data.Sources.TextFileSource, SharpMind.Data",
                        Args = new Dictionary<string, string> { ["path"] = path },
                    },
                },
            },
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
                        DisplayName = "Remote Dataset",
                        TypeName = "SharpMind.Data.Sources.HuggingFaceSource, SharpMind.Data",
                        Args = new Dictionary<string, string> { ["dataset"] = "allenai/c4" },
                    },
                },
            },
        };

    [Fact]
    public void Compute_HashesFileContent()
    {
        using var dir = new TempDirectory();
        string path = dir.Write("corpus.txt", "the quick brown fox\njumps over the lazy dog\n");

        var job = JobWithTextFile(path);
        var hashes = SourceHasher.Compute(job.Sources);

        Assert.Single(hashes);
        Assert.Equal("Text File", hashes.Keys.Single());
        Assert.NotNull(hashes["Text File"]);
        Assert.Equal(64, hashes["Text File"]!.Length); // SHA-256 hex
    }

    [Fact]
    public void Compute_SameContent_SameHash()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "identical corpus\n");
        string b = dir.Write("b.txt", "identical corpus\n");

        var h1 = SourceHasher.Compute(JobWithTextFile(a, "A").Sources);
        var h2 = SourceHasher.Compute(JobWithTextFile(b, "B").Sources);

        Assert.Equal(h1["A"], h2["B"]);
    }

    [Fact]
    public void Compute_ContentChange_ChangesHash()
    {
        using var dir = new TempDirectory();
        string path = dir.Write("corpus.txt", "original text\n");

        var before = SourceHasher.Compute(JobWithTextFile(path).Sources);
        File.WriteAllText(path, "changed text\n");
        var after = SourceHasher.Compute(JobWithTextFile(path).Sources);

        Assert.NotEqual(before["Text File"], after["Text File"]);
    }

    [Fact]
    public void Compute_DifferentFiles_DifferentHash()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "content one\n");
        string b = dir.Write("b.txt", "content two\n");

        var ha = SourceHasher.Compute(JobWithTextFile(a).Sources);
        var hb = SourceHasher.Compute(JobWithTextFile(b).Sources);

        Assert.NotEqual(ha["Text File"], hb["Text File"]);
    }

    [Fact]
    public void Compute_SkipsSourcesWithoutResolvablePath()
    {
        var hashes = SourceHasher.Compute(JobWithNoPathSource().Sources);
        Assert.Empty(hashes);
    }

    [Fact]
    public void Combined_SameAcrossSourceSets_WhenContentsMatch()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "shared corpus\n");
        string b = dir.Write("b.txt", "shared corpus\n");

        var jobA = JobWithTextFile(a);
        var jobB = JobWithTextFile(b);

        Assert.Equal(SourceHasher.Combined(jobA.Sources), SourceHasher.Combined(jobB.Sources));
    }

    [Fact]
    public void Combined_IsNull_WhenNothingHashable()
    {
        Assert.Null(SourceHasher.Combined(JobWithNoPathSource().Sources));
    }

    [Fact]
    public void ComputeFileHashes_MapsEachFileToItsHash()
    {
        using var dir = new TempDirectory();
        string a = dir.Write("a.txt", "file one\n");
        string b = dir.Write("b.txt", "file two\n");

        var job = new TrainJobSettings
        {
            Sources =
            {
                new JobSource
                {
                    Component = new JobComponent
                    {
                        DisplayName = "multi",
                        TypeName = "SharpMind.Data.Sources.TextFileSource, SharpMind.Data",
                        Args = new Dictionary<string, string> { ["path"] = dir.Path + "\\*.txt" },
                    },
                },
            },
        };

        var map = SourceHasher.ComputeFileHashes(job.Sources);

        Assert.Single(map);
        Assert.Equal(2, map["multi"].Count);
        Assert.Contains(a, map["multi"].Keys);
        Assert.Contains(b, map["multi"].Keys);
        Assert.All(map["multi"].Values, v => Assert.Equal(64, v.Length));
        Assert.NotEqual(map["multi"][a], map["multi"][b]);
    }

    [Fact]
    public void ComputeFileHashes_SkipsSourcesWithoutResolvablePath()
    {
        var map = SourceHasher.ComputeFileHashes(JobWithNoPathSource().Sources);
        Assert.Empty(map);
    }

    [Fact]
    public void ComputeDeltas_IdentifiesNewChangedAndUnchanged()
    {
        var current = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new()
            {
                ["new.txt"] = "h-new",
                ["changed.txt"] = "h-changed-v2",
                ["same.txt"] = "h-same",
            },
        };
        var previous = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new()
            {
                ["changed.txt"] = "h-changed-v1",
                ["same.txt"] = "h-same",
            },
        };

        var deltas = SourceHasher.ComputeDeltas(current, previous);

        Assert.Single(deltas);
        Assert.Equal(new[] { "changed.txt", "new.txt" }, deltas["src"].OrderBy(p => p, StringComparer.Ordinal).ToArray());
        Assert.DoesNotContain("same.txt", deltas["src"]);
    }

    [Fact]
    public void ComputeDeltas_AllFilesNew_WhenNoPreviousMap()
    {
        var current = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new() { ["a.txt"] = "h-a", ["b.txt"] = "h-b" },
        };

        var deltas = SourceHasher.ComputeDeltas(current, []);

        Assert.Equal(new[] { "a.txt", "b.txt" }, deltas["src"].OrderBy(p => p, StringComparer.Ordinal).ToArray());
    }

    [Fact]
    public void ComputeDeltas_Empty_WhenNoChanges()
    {
        var current = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new() { ["a.txt"] = "h-a" },
        };

        var deltas = SourceHasher.ComputeDeltas(current, current);

        Assert.Empty(deltas);
    }

    [Fact]
    public void ComputeDeltas_IgnoresRemovedFiles()
    {
        var current = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new() { ["kept.txt"] = "h-kept" },
        };
        var previous = new Dictionary<string, Dictionary<string, string>>
        {
            ["src"] = new() { ["deleted.txt"] = "h-gone", ["kept.txt"] = "h-kept" },
        };

        var deltas = SourceHasher.ComputeDeltas(current, previous);

        Assert.Empty(deltas);
    }
}