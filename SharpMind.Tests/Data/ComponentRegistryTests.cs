using System.Reflection;
using SharpMind.Data.Metadata;
using SharpMind.Data.Sources;
using SharpMind.Data.Sources.Csv;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;

namespace SharpMind.Tests.Data;

/// <summary>
/// Verifies <see cref="ComponentRegistry"/>: built-in discovery, reflection-driven
/// parameter description, friendly-type conversion/build, unknown-value errors,
/// and tolerant scanning of a plugins folder (bad DLLs become warnings).
/// </summary>
public sealed class ComponentRegistryTests
{
    [Fact]
    public void Scan_FindsBuiltInSourcesAndStages()
    {
        var registry = ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly);

        Assert.Contains(registry, d => d.Kind == ComponentKind.Source && d.Name == "Text File");
        Assert.Contains(registry, d => d.Kind == ComponentKind.Source && d.Name == "CSV");
        Assert.Contains(registry, d => d.Kind == ComponentKind.Source && d.Name == "JSONL");
        Assert.Contains(registry, d => d.Kind == ComponentKind.Source && d.Name == "Pseudo Language");
        Assert.Contains(registry, d => d.Kind == ComponentKind.Stage && d.Name == "Min Length Filter");
    }

    [Fact]
    public void Scan_CoversEveryComponentKindAttribute()
    {
        var registry = ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly);
        var decorated = typeof(ComponentRegistry).Assembly.GetTypes()
            .Where(t => t.GetCustomAttribute<ComponentKindAttribute>(inherit: false) is not null);

        foreach (var t in decorated)
            Assert.Contains(registry, d => d.Type == t);
    }

    [Fact]
    public void Find_MatchByFullNameAndSimpleName()
    {
        var registry = ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly);
        var byFull = ComponentRegistry.Find(typeof(TextFileSource).FullName, registry);
        var bySimple = ComponentRegistry.Find(typeof(TextFileSource).Name, registry);

        Assert.NotNull(byFull);
        Assert.Same(byFull, bySimple);
        Assert.Null(ComponentRegistry.Find("No.Such.Type", registry));
    }

    [Fact]
    public void Build_CsvSource_ConvertsValuesFromStrings()
    {
        var csv = ComponentRegistry.Find(typeof(CsvDataSource).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(csv);

        var source = ComponentRegistry.Build<IDataSource>(csv!, new Dictionary<string, string>
        {
            ["path"] = "corpus.csv",
            ["textColumn"] = "body",
            ["hasHeader"] = "true",
            ["delimiter"] = ";",
        });

        var csvSource = Assert.IsType<CsvDataSource>(source);
        Assert.Contains("corpus.csv", csvSource.Description);
    }

    [Fact]
    public void Build_UsesDefaultValueAttributeForMissingOptional()
    {
        var csv = ComponentRegistry.Build<CsvDataSource>(
            ComponentRegistry.Find(typeof(CsvDataSource).FullName!, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly))!,
            new Dictionary<string, string> { ["path"] = "corpus.csv" });

        Assert.Contains("corpus.csv", csv.Description);
    }

    [Fact]
    public void Build_PseudoLanguageSource_ConvertsValuesFromStrings()
    {
        var pseudo = ComponentRegistry.Find(typeof(PseudoLanguageSource).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(pseudo);

        var source = ComponentRegistry.Build<IDataSource>(pseudo!, new Dictionary<string, string>
        {
            ["vocabSize"] = "2000",
            ["rootMorphemes"] = "200",
            ["affixes"] = "15",
            ["sequenceCount"] = "25",
            ["level"] = "Patterns",
        });

        var pseudoSource = Assert.IsType<PseudoLanguageSource>(source);
        Assert.Contains("Patterns", pseudoSource.Description);
    }

    [Fact]
    public void Build_RejectsNonNumericForIntParam()
    {
        var minLen = ComponentRegistry.Find(typeof(MinLengthFilter).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(minLen);

        Assert.Throws<InvalidOperationException>(() =>
            ComponentRegistry.Build<MinLengthFilter>(minLen!, new Dictionary<string, string> { ["minLength"] = "not-a-number" }));
    }

    [Fact]
    public void ScanFolder_TolerantOfBadDll_AndReportsWarning()
    {
        using var dir = new TempDirectory();
        dir.Write("broken.dll", "this is not a real dll");

        var (registry, warnings) = (ComponentRegistry.ScanFolder(dir.Path, out var ws), ws);

        Assert.NotEmpty(warnings);
        Assert.Contains("broken.dll", string.Join(" ", warnings));
        Assert.Contains(registry, d => d.Kind == ComponentKind.Source && d.Name == "Text File");
    }

    [Fact]
    public void Build_MissingRequiredThrows()
    {
        var minLen = ComponentRegistry.Find(typeof(MinLengthFilter).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(minLen);
        Assert.Throws<InvalidOperationException>(() =>
            ComponentRegistry.Build<MinLengthFilter>(minLen!, new Dictionary<string, string>()));
    }

    [Fact]
    public void Build_Stage_FromStrings()
    {
        var minLen = ComponentRegistry.Find(typeof(MinLengthFilter).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(minLen);
        var stage = ComponentRegistry.Build<ICleaningStage>(minLen!, new Dictionary<string, string> { ["minLength"] = "5" });
        Assert.NotNull(stage);
    }

    [Fact]
    public void Scan_EnumChoicesFromAttribute()
    {
        var pii = ComponentRegistry.Find(typeof(PiiMasker).FullName, ComponentRegistry.Scan(typeof(ComponentRegistry).Assembly));
        Assert.NotNull(pii);
        Assert.Equal(ComponentKind.Stage, pii.Kind);
    }

    [Fact]
    public void Scan_InterfaceFallback_RegistersUnattributedSourceAndStage()
    {
        // Fixture types live in the test assembly and carry no [ComponentKind].
        var warnings = new List<string>();
        var registry = ComponentRegistry.Scan(typeof(UnattributedFallbackSource).Assembly, warnings);

        Assert.Contains(registry, d => d.Type == typeof(UnattributedFallbackSource) && d.Kind == ComponentKind.Source);
        Assert.Contains(registry, d => d.Type == typeof(UnattributedFallbackStage) && d.Kind == ComponentKind.Stage);

        var src = registry.Single(d => d.Type == typeof(UnattributedFallbackSource));
        Assert.Equal(nameof(UnattributedFallbackSource), src.Name);
        Assert.Contains("data source", src.Description, StringComparison.OrdinalIgnoreCase);

        var stage = registry.Single(d => d.Type == typeof(UnattributedFallbackStage));
        Assert.Equal(nameof(UnattributedFallbackStage), stage.Name);
        Assert.Contains("pipeline stage", stage.Description, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(warnings);
    }

    [Fact]
    public void Scan_AttributeStillWins_OverInterfaceFallbackName()
    {
        var registry = ComponentRegistry.Scan(typeof(UnattributedFallbackSource).Assembly);
        var attr = registry.Single(d => d.Type == typeof(AttributedFallbackSource));
        Assert.Equal("Custom Attributed Source", attr.Name);
        Assert.Equal("Declared description.", attr.Description);
    }
}

/// <summary>Test fixture: an IDataSource without [ComponentKind], discovered by interface fallback.</summary>
public sealed class UnattributedFallbackSource : IDataSource
{
    public long? EstimatedCount => 0;
    public string Description => "Unattributed fallback source.";
    public IAsyncEnumerable<string> ReadAsync(CancellationToken cancellationToken = default) => AsyncEnumerable.Empty<string>();
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>Test fixture: an ICleaningStage without [ComponentKind], discovered by interface fallback.</summary>
public sealed class UnattributedFallbackStage : ICleaningStage
{
    public string Name => "Unattributed Fallback Stage";
    public string? Process(string document) => document;
}

/// <summary>Test fixture: carries [ComponentKind] so the attribute supplies the display name.</summary>
[ComponentKind("Custom Attributed Source", "Declared description.")]
public sealed class AttributedFallbackSource : IDataSource
{
    public long? EstimatedCount => 0;
    public string Description => "Attributed source.";
    public IAsyncEnumerable<string> ReadAsync(CancellationToken cancellationToken = default) => AsyncEnumerable.Empty<string>();
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}