using System.Text.Json;
using SharpMind.Core.Quantization;

namespace SharpMind.CUI.App;

/// <summary>A configured training job — the persisted "what &amp; how to train" document.</summary>
public sealed class TrainJobSettings
{
    /// <summary>File extension for saved training jobs (<c>*.smmt</c>).</summary>
    public const string JobExtension = ".smmt";

    /// <summary>
    /// Current on-disk container version. Bumped only when the wrapped
    /// <see cref="TrainJobDocument"/> shape changes incompatibly; the job's own
    /// field additions remain additive and don't require a bump. Readers reject
    /// files written by a newer build and (via <see cref="Load"/>) transparently
    /// migrate unwrapped v1 files saved by older builds.
    /// </summary>
    public const int CurrentFormatVersion = 1;

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

    // --- Incremental training ------------------------------------------

    /// <summary>
    /// True to train only on the corpus deltas since the last run: per-file
    /// hashes (see <see cref="SourceFileHashes"/>) identify exactly which files
    /// are new or changed, and the pipeline is built over only those files.
    /// Runs always resume the latest checkpoint regardless of
    /// <see cref="StartFresh"/>/<see cref="ResumeFrom"/>. When no deltas exist
    /// the run is a clean no-op ("nothing to train").
    /// </summary>
    public bool IncrementalMode { get; set; }

    /// <summary>
    /// Per-source per-file content hashes (display name → absolute file path →
    /// SHA-256), persisted after a successful incremental run. The next run
    /// diffs the current corpus against this map to decide what still needs
    /// training. Persisted on-disk alongside the checkpoints so it survives
    /// app restarts even before the job file is manually saved.
    /// </summary>
    public Dictionary<string, Dictionary<string, string>> SourceFileHashes { get; set; } = [];

    // --- Tokenizer ---------------------------------------------------------

    /// <summary>Target BPE vocabulary size (not yet trained).</summary>
    public int TokenizerVocabSize { get; set; } = 1024;

    /// <summary>
    /// Tokenizer kind, an additive choice. "Char" trains a character-level
    /// tokenizer whose vocabulary is the sorted set of distinct characters in
    /// the corpus (a byte/character GPT); "Bpe" (or null/empty) trains the
    /// usual byte-pair-encoding tokenizer. Defaults to BPE so existing jobs
    /// behave exactly as before.
    /// </summary>
    public string? TokenizerKind { get; set; }

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

    // --- Architecture & kernel options ---
    // Null means "use the architecture preset's default" (see TrainingPresets);
    // explicit values override the preset. Without a preset the resolved config
    // matches SharpMindConfig.Gpt so existing jobs behave exactly as before.

    /// <summary>
    /// Architecture preset key ("gpt2", "llama", "bert", "qwen", "mixtral").
    /// Null = manual/custom (resolves to GPT-2 style unless options are set).
    /// Used to pick activation/gate/ffn/norm defaults and to stamp the exported
    /// model's architecture name so loading reproduces the same config.
    /// </summary>
    public string? ArchitecturePreset { get; set; }

    /// <summary>Activation enum name ("GELU", "SiLU", "ReLU"); null = preset default.</summary>
    public string? Activation { get; set; }

    /// <summary>Gate enum name ("None", "SwiGLU", "GeGLU"); null = preset default.</summary>
    public string? Gate { get; set; }

    /// <summary>FFN enum name ("Dense", "Gated", "MoE"); null = preset default.</summary>
    public string? Ffn { get; set; }

    /// <summary>Attention enum name ("MHA", "GQA", "MQA"); null = derived from heads vs KV heads.</summary>
    public string? Attention { get; set; }

    /// <summary>Norm enum name ("RMSNorm", "LayerNorm"); null = preset default.</summary>
    public string? Norm { get; set; }

    /// <summary>Architecture kind ("Decoder", "Encoder"); null = preset default.</summary>
    public string? Arch { get; set; }

    /// <summary>Positional encoding enum name ("NoPE", "RoPE", "ALiBi", "Learned"); null = RoPE.</summary>
    public string? PositionalEncoding { get; set; }

    /// <summary>
    /// Optimizer name ("AdamW", "SGD"); null/empty = AdamW. SGD uses
    /// <see cref="SgdMomentum"/>; AdamW keeps its internal beta defaults.
    /// </summary>
    public string? Optimizer { get; set; }

    /// <summary>SGD momentum coefficient (ignored by AdamW). Default 0.</summary>
    public float SgdMomentum { get; set; } = 0f;

    /// <summary>Total number of experts when <see cref="Ffn"/> is MoE.</summary>
    public int NumExperts { get; set; } = 8;

    /// <summary>Experts activated per token (top-k) when <see cref="Ffn"/> is MoE.</summary>
    public int TopKExperts { get; set; } = 2;

    public int TotalSteps { get; set; } = 200;
    public int LogInterval { get; set; } = 25;
    public int BatchSize { get; set; } = 1;
    public int SeqLen { get; set; } = 16;
    public int GradAccumSteps { get; set; } = 1;

    /// <summary>
    /// Batch strategy, an additive choice. "RandomWindow" samples each batch of
    /// <see cref="SeqLen"/>-token windows at random offsets from one flat corpus
    /// stream (nanoGPT-style); "Packing" (or null/empty) uses the default
    /// document-packing batcher. Defaults to packing so existing jobs behave
    /// exactly as before.
    /// </summary>
    public string? BatchingKind { get; set; }

    /// <summary>True when the job selects the character-level tokenizer.</summary>
    public bool UsesCharacterTokenizer
        => !string.IsNullOrWhiteSpace(TokenizerKind)
           && TokenizerKind.Equals("Char", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the job selects random-window batch sampling.</summary>
    public bool UsesRandomWindowBatching
        => !string.IsNullOrWhiteSpace(BatchingKind)
           && BatchingKind.Equals("RandomWindow", StringComparison.OrdinalIgnoreCase);

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

    public static string DefaultFolder => SharpMindPaths.Training;

    public bool Save(string path, out string? error)
    {
        error = null;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            SavedAtUtc = DateTime.UtcNow;
            var doc = new TrainJobDocument
            {
                Version = CurrentFormatVersion,
                SavedAtUtc = SavedAtUtc,
                Job = this,
            };
            File.WriteAllText(path, JsonSerializer.Serialize(doc, Options));
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
            using var root = JsonDocument.Parse(json);
            var rootElement = root.RootElement;

            // Versioned container produced by this build. Older (v1, unwrapped)
            // files are plain TrainJobSettings JSON and load straight through.
            if (rootElement.TryGetProperty("version", out var versionProp))
            {
                int version = versionProp.GetInt32();
                if (version > CurrentFormatVersion)
                {
                    error = $"Job file format v{version} is newer than this build supports (v{CurrentFormatVersion}).";
                    return null;
                }

                if (version < CurrentFormatVersion)
                {
                    // Older wrapped container — the inner job shape is the same
                    // payload type, just deserialize it. (Future migrations hook
                    // in here.)
                }

                if (rootElement.TryGetProperty("job", out var jobProp) && jobProp.ValueKind == JsonValueKind.Object)
                    return jobProp.Deserialize<TrainJobSettings>(Options);
                error = "Job file is missing its 'job' payload.";
                return null;
            }

            return rootElement.Deserialize<TrainJobSettings>(Options);
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
        return [.. Directory.GetFiles(folder, "*" + JobExtension).OrderByDescending(File.GetLastWriteTimeUtc)];
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

/// <summary>
/// The on-disk wrapper around <see cref="TrainJobSettings"/>: a stable version
/// stamp plus the saved-at time, with the job's JSON kept whole inside
/// <c>job</c>. The version lets a future format change be detected (and
/// migrated) instead of failing to parse; older builds saved the bare job
/// object, which <see cref="TrainJobSettings.Load"/> still accepts so legacy
/// <c>*.smmt</c> files (e.g. <c>shakespeare-job.smmt</c>) survive.
/// </summary>
public sealed class TrainJobDocument
{
    /// <summary>On-disk format version written by this build.</summary>
    [System.Text.Json.Serialization.JsonPropertyName("version")]
    public int Version { get; set; } = TrainJobSettings.CurrentFormatVersion;

    [System.Text.Json.Serialization.JsonPropertyName("savedAtUtc")]
    public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

    [System.Text.Json.Serialization.JsonPropertyName("job")]
    public TrainJobSettings Job { get; set; } = new();
}