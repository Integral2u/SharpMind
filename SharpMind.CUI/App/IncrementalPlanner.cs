using SharpMind.Data;
using SharpMind.Data.Metadata;
using SharpMind.Data.Sources;

namespace SharpMind.CUI.App;

/// <summary>
/// Builds the data-source list for an incremental training run. Instead of
/// resolving the job's configured sources to everything they match, it diffs
/// the current per-file hashes against the previous run's persisted map
/// (<see cref="TrainJobSettings.SourceFileHashes"/>) and rebuilds each
/// file-based source restricted to exactly the new + changed files. Sources the
/// run must still see the whole of (remote, generated, or bespoke plugin
/// sources without a multi-path constructor) are kept whole and noted loudly.
/// </summary>
public sealed class IncrementalPlan
{
    /// <summary>The sources that feed this run, aligned with non-skipped indices.</summary>
    public List<IDataSource> Sources { get; set; } = [];

    /// <summary>Per job.Sources index: true when the source contributes nothing.</summary>
    public bool[] SkipSource { get; init; } = [];

    /// <summary>True when nothing is left to train (all sources were skipped).</summary>
    public bool NothingToTrain { get; init; }

    /// <summary>
    /// Per-file hashes computed for this corpus as it currently sits on disk;
    /// the caller persists these onto the job after a successful run so the
    /// NEXT run can diff against exactly what was trained.
    /// </summary>
    public Dictionary<string, Dictionary<string, string>> CurrentFileHashes { get; init; } = [];
}

/// <summary>Computes and applies per-file corpus deltas for an incremental run.</summary>
public static class IncrementalPlanner
{
    /// <summary>
    /// Builds the delta plan. For each configured source:
    ///  <list type="bullet">
    ///   <item>No resolvable <c>path</c> arg (remote/generated) → kept whole; the
    ///        run cannot attribute documents to files, so it retrains it all.</item>
    ///   <item>File-based with a multi-path constructor (Text/JSONL/Fusechat) → a
    ///        new instance over exactly the new+changed files.</item>
    ///   <item>Any other file-based source → kept whole when at least one of its
    ///        files changed (per-file restriction unsupported), skipped otherwise.</item>
    ///  </list>
    /// Returns a plan with <see cref="IncrementalPlan.NothingToTrain"/> when
    /// incremental mode is on and every source contributed no deltas.
    /// </summary>
    public static IncrementalPlan Build(
        TrainJobSettings job,
        IReadOnlyList<ComponentDescriptor> registry,
        Action<string> log)
    {
        if (!job.IncrementalMode)
        {
            var all = job.Sources.Select(s => BuildSource(s.Component, registry)).ToList();
            return new IncrementalPlan
            {
                SkipSource = new bool[job.Sources.Count],
                Sources = all,
                CurrentFileHashes = SourceHasher.ComputeFileHashes(job.Sources),
            };
        }

        var current = SourceHasher.ComputeFileHashes(job.Sources);
        var deltas = SourceHasher.ComputeDeltas(current, job.SourceFileHashes);

        var sources = new List<IDataSource>();
        var skip = new bool[job.Sources.Count];
        bool anyDelta = false;

        for (int i = 0; i < job.Sources.Count; i++)
        {
            var js = job.Sources[i];
            string key = js.DisplayName ?? js.Component.TypeName;
            string? path = PathArg(js.Component.Args);

            if (string.IsNullOrWhiteSpace(path))
            {
                // Cannot attribute documents to files — keep the whole source.
                log($"Incremental: {key} is a non-file source; retraining it whole.");
                sources.Add(BuildSource(js.Component, registry));
                anyDelta = true;
                continue;
            }

            bool hasDeltas = deltas.TryGetValue(key, out var deltaFiles) && deltaFiles.Length > 0;
            if (!hasDeltas)
            {
                log($"Incremental: {key} — no new or changed files; skipping.");
                skip[i] = true;
                continue;
            }

            var rebuilt = BuildDeltaSource(js.Component, deltaFiles!, registry, out var restricted);
            if (restricted)
                log($"Incremental: {key} — training {deltaFiles!.Length} new/changed file(s) only.");
            else
                log($"Incremental: {key} — {deltaFiles!.Length} file(s) changed; source cannot be restricted per-file, retraining it whole.");
            sources.Add(rebuilt);
            anyDelta = true;
        }

        return new IncrementalPlan
        {
            Sources = sources,
            SkipSource = skip,
            NothingToTrain = !anyDelta,
            CurrentFileHashes = current,
        };
    }

    /// <summary>
    /// Rebuilds <paramref name="component"/> over exactly the delta files when
    /// its runtime type exposes a multi-path constructor (<c>(IEnumerable&lt;string&gt;, ...)</c>),
    /// passing the remaining constructor args from the component's saved values.
    /// Falls back to a whole-source build otherwise (<c>restricted=false</c>).
    /// </summary>
    private static IDataSource BuildDeltaSource(
        JobComponent component,
        string[] deltaFiles,
        IReadOnlyList<ComponentDescriptor> registry,
        out bool restricted)
    {
        var descriptor = ComponentRegistry.Find(component.TypeName, registry);
        if (descriptor is not null && descriptor.Type.Name.EndsWith(nameof(TextFileSource), StringComparison.Ordinal))
        {
            restricted = true;
            return new TextFileSource(deltaFiles, ModeArg(component.Args));
        }
        if (descriptor is not null && descriptor.Type.Name.EndsWith(nameof(JsonlSource), StringComparison.Ordinal))
        {
            restricted = true;
            return new JsonlSource(deltaFiles, TextFieldArg(component.Args));
        }
        if (descriptor is not null && descriptor.Type.Name.EndsWith(nameof(FusechatSource), StringComparison.Ordinal))
        {
            restricted = true;
            return new FusechatSource(deltaFiles);
        }

        restricted = false;
        return BuildSource(component, registry);
    }

    private static IDataSource BuildSource(JobComponent component, IReadOnlyList<ComponentDescriptor> registry)
    {
        if (component is null)
            throw new InvalidOperationException("No data source configured for this job.");
        var descriptor = ComponentRegistry.Find(component.TypeName, registry)
            ?? throw new InvalidOperationException($"Unknown data source '{component.TypeName}'.");
        return (IDataSource)ComponentRegistry.Build<IDataSource>(descriptor, component.Args);
    }

    private static TextFileSource.DocumentMode ModeArg(IReadOnlyDictionary<string, string> args)
        => args.TryGetValue("mode", out var raw) &&
           Enum.TryParse<TextFileSource.DocumentMode>(raw, ignoreCase: true, out var mode)
            ? mode : TextFileSource.DocumentMode.LinePerDoc;

    private static string TextFieldArg(IReadOnlyDictionary<string, string> args)
        => args.TryGetValue("textField", out var raw) && !string.IsNullOrWhiteSpace(raw)
            ? raw : "text";

    private static string? PathArg(IReadOnlyDictionary<string, string> args)
        => args.TryGetValue("path", out var raw) ? raw : null;
}