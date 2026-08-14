using System.Text.Json;
using SharpMind.Core.Quantization;

namespace SharpMind.CUI.App;

/// <summary>A configured training job — the persisted "what &amp; how to train" document.</summary>
public sealed class TrainJobSettings
{
    /// <summary>File extension for saved training jobs (<c>*.smmt</c>).</summary>
    public const string JobExtension = ".smmt";

    /// <summary>Display name; also the file name (<c>&lt;name&gt;.smmt</c>) under <see cref="DefaultFolder"/>.</summary>
    public string Name { get; set; } = "Untitled";

    public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

    // --- What: sources + stage chains --------------------------------

    /// <summary>Each configured data source with its own per-source stages.</summary>
    public List<JobSource> Sources { get; set; } = [];

    /// <summary>Global cleaning stages applied to the merged stream of all sources.</summary>
    public List<JobComponent> GlobalStages { get; set; } = [];

    /// <summary>
    /// Content fingerprints for each configured source (display name → SHA-256),
    /// recorded at the start of a run. Lets a later run detect that the corpus
    /// changed — invalidating the path-only tokenizer cache — and stamps an
    /// audit fingerprint onto exported models. Unhashable sources (remote) are
    /// absent. Empty for jobs that have never run.
    /// </summary>
    public Dictionary<string, string?> SourceHashes { get; set; } = [];

    // --- Tokenizer ---------------------------------------------------------

    /// <summary>Target BPE vocabulary size (not yet trained).</summary>
    public int TokenizerVocabSize { get; set; } = 1024;

    /// <summary>Where the trained tokenizer is cached on disk. Auto-derived from <see cref="Name"/> if null.</summary>
    public string? TokenizerCachePath { get; set; }

    // --- How: model + hyperparameters ------------------------------------

    /// <summary>True when sizing ran ModelSizer; the result is stored in the config fields below.</summary>
    public bool AutoSized { get; set; }

    public int VocabSize { get; set; }
    public int HiddenDim { get; set; } = 128;
    public int NumLayers { get; set; } = 4;
    public int NumHeads { get; set; } = 8;
    public int NumKvHeads { get; set; } = 8;
    public int FfnDim { get; set; } = 512;
    public int MaxSeqLen { get; set; } = 256;
    public float NormEps { get; set; } = 1e-3f;

    public int TotalSteps { get; set; } = 200;
    public int LogInterval { get; set; } = 25;
    public int BatchSize { get; set; } = 1;
    public int SeqLen { get; set; } = 16;
    public int GradAccumSteps { get; set; } = 1;
    public float LearningRate { get; set; } = 8e-4f;
    public float MinLr { get; set; } = 3e-5f;
    public int WarmupSteps { get; set; } = 20;
    public float WeightDecay { get; set; } = 0.1f;
    public float GradClipNorm { get; set; } = 1.0f;
    public float LabelSmoothing { get; set; } = 0.1f;

    /// <summary>QAT target as its enum name ("F32" disables QAT); null = F32. See <see cref="QuantDType"/>.</summary>
    public string? QuantAwareTraining { get; set; }

    // --- Where: checkpoints + export -------------------------------------

    public int CheckpointInterval { get; set; } = 50;

    /// <summary>Retained step checkpoints during a run: 0=ALL, N=keep newest N, negative=none (final only).</summary>
    public int KeepRecent { get; set; } = 3;

    /// <summary>Final .smm export path (browseable afterwards). May name a <c>.smm</c>
    /// file or an output folder; when blank the run defaults it under <see cref="DefaultFolder"/>.</summary>
    public string? ExportPath { get; set; }

    /// <summary>True when <see cref="ExportPath"/> points at a concrete <c>.smm</c> file rather than an output folder.</summary>
    public static bool IsExportFile(string? path)
    {
        if (string.IsNullOrWhiteSpace(path)) return false;
        if (Directory.Exists(path)) return false;
        return Path.GetExtension(path).Length > 0 || (!Directory.Exists(Path.GetDirectoryName(path) ?? "") && !path.EndsWith(Path.DirectorySeparatorChar.ToString()));
    }

    /// <summary>
    /// The output folder that receives the <c>checkpoints-{Name}</c> subfolder and,
    /// by default, the final <c>.smm</c>. When <see cref="ExportPath"/> names a file
    /// the folder is its parent; otherwise the path itself is the folder.
    /// Falls back to <see cref="DefaultFolder"/> when no export path is set.
    /// </summary>
    public string ExportFolder
    {
        get
        {
            if (string.IsNullOrWhiteSpace(ExportPath)) return DefaultFolder;
            if (IsExportFile(ExportPath)) return Path.GetDirectoryName(ExportPath) ?? DefaultFolder;
            return ExportPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        }
    }

    /// <summary>
    /// Checkpoint directory, always derived from the export path base plus the
    /// job name prefixed with "checkpoints": <c>&lt;ExportFolder&gt;/checkpoints-{Name}</c>.
    /// Falls back to <see cref="DefaultFolder"/> when no export path is set.
    /// </summary>
    public string CheckpointDir => Path.Combine(ExportFolder, $"checkpoints-{Sanitize(Name)}");

    /// <summary>
    /// Resolves <see cref="ExportPath"/> to a concrete writable <c>.smm</c> file:
    /// a folder is used as-is with <c>{Sanitize(Name)}.smm</c> appended, a file path
    /// is used verbatim, and a blank path defaults under <see cref="DefaultFolder"/>.
    /// </summary>
    public string ExportFilePath
    {
        get
        {
            if (string.IsNullOrWhiteSpace(ExportPath))
                return Path.Combine(DefaultFolder, Sanitize(Name), Sanitize(Name) + ".smm");
            return IsExportFile(ExportPath)
                ? ExportPath
                : Path.Combine(ExportFolder, Sanitize(Name) + ".smm");
        }
    }

    /// <summary>Resume from this checkpoint directory; null = auto (see <see cref="StartFresh"/>).</summary>
    public string? ResumeFrom { get; set; }

    /// <summary>
    /// True to train from random weights even when checkpoints exist, ignoring
    /// <see cref="ResumeFrom"/>. False (default) resumes the latest checkpoint
    /// under <see cref="CheckpointDir"/>, or starts fresh when none exist.
    /// </summary>
    public bool StartFresh { get; set; }

    // --- Embed in the exported .smm ----------------------------

    /// <summary>
    /// Path to a <c>.txt</c>/<c>.md</c> file whose contents become the embedded
    /// system prompt. The file is read at export time so edits don't force a
    /// wizard round-trip. Optional.
    /// </summary>
    public string? SystemPromptPath { get; set; }

    /// <summary>
    /// Folder containing <c>*.md</c> skill documents; each file is embedded as
    /// one skill. Read at export time. Optional.
    /// </summary>
    public string? SkillsFolder { get; set; }

    /// <summary>Tool DLL paths to embed in the exported .smm file. Optional.</summary>
    public List<string> PluginDllPaths { get; set; } = [];

    // --- Persistence -------------------------------------------------------

    public static string DefaultFolder
    {
        get
        {
            string root = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            if (string.IsNullOrEmpty(root)) root = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(root, "SharpMind", "train-jobs");
        }
    }

    public bool Save(string path, out string? error)
    {
        error = null;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            SavedAtUtc = DateTime.UtcNow;
            File.WriteAllText(path, JsonSerializer.Serialize(this, Options));
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static TrainJobSettings? Load(string path, out string? error)
    {
        error = null;
        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<TrainJobSettings>(json, Options);
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return null;
        }
    }

    public static List<string> ListSaved()
        => ListSavedIn(DefaultFolder);

    /// <summary>Lists saved job files in <paramref name="folder"/> matching <see cref="JobExtension"/>.</summary>
    public static List<string> ListSavedIn(string folder)
    {
        if (!Directory.Exists(folder)) return [];
        return Directory.GetFiles(folder, "*" + JobExtension)
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .ToList();
    }

    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    private static readonly QuantDType[] QatTargets =
    [
        QuantDType.F32, QuantDType.F16, QuantDType.Q8_0, QuantDType.Q4_0,
    ];

    private static string Sanitize(string name)
    {
        foreach (var c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return string.IsNullOrWhiteSpace(name) ? "job" : name.Trim();
    }
}