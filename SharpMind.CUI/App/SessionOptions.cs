using SharpMind;
using SharpMind.Inference;
using SharpMind.Inference.Agent;

namespace SharpMind.CUI.App;

/// <summary>Which IGenerator strategy to build the session with.</summary>
public enum GeneratorStrategy
{
    /// <summary>One token at a time. Simplest, always correct, baseline speed.</summary>
    Standard,
    /// <summary>Draft-and-verify against a smaller model. Faster, needs a draft model path.</summary>
    Speculative,
    /// <summary>Multiple candidate heads predicted per step. Faster, needs a Medusa-trained model.</summary>
    Medusa,
    /// <summary>
    /// No real model at all — SessionLauncher skips loading any GGUF entirely
    /// and the session is driven by DebugChatBridge's scripted responses.
    /// Exists purely for testing the CUI's own plumbing (rendering, dialogs,
    /// sub-agent name display) without needing a model file or paying any
    /// inference cost. Type <c>TestOptions</c> or <c>TestAgent</c> into chat
    /// once running to exercise specific scripted scenarios.
    /// </summary>
    UIDebug
}

/// <summary>Which IKVCacheBuilder strategy backs the session's attention cache.</summary>
public enum CacheStrategy
{
    /// <summary>Flat pre-allocated buffer. Simplest, most memory-hungry per sequence.</summary>
    Standard,
    /// <summary>Block/page-based allocation. Better for many concurrent or growing sequences.</summary>
    Paged,
    /// <summary>Cache entries stored quantized. Less memory, small accuracy cost.</summary>
    Quantized
}

/// <summary>
/// Everything needed to stand up a ChatSession, gathered in one place so the
/// UI can present it as a single options screen and the launcher can resolve
/// it to a concrete generic ChatSession&lt;T,K&gt; instantiation.
/// </summary>
public sealed class SessionOptions
{
    // Model + project
    public string? ModelPath { get; set; }

    /// <summary>
    /// Working directory for file-related tool calls and example code. Worth
    /// being precise about what this actually does: the engine's
    /// InterceptingFileSystem gates *whether* a tool call may touch a given
    /// path, but it does not root relative paths against this folder or
    /// confine absolute ones to it — there's no sandbox enforcement at the
    /// engine level. What this field actually does is get surfaced to the
    /// agent as context (see SessionLauncher) so a model has a sensible
    /// default location to read/write relative to, rather than guessing.
    /// </summary>
    public string? ProjectPath { get; set; }
    public List<string> SkillFolders { get; set; } = [];

    /// <summary>Explicit, individually-chosen tool DLL paths (the ;-separated field on the Options screen).</summary>
    public List<string> ToolAssemblyPaths { get; set; } = [];

    /// <summary>
    /// A folder scanned for *.dll at launch time, in addition to the explicit
    /// paths above. Kept separate rather than pre-expanded into
    /// ToolAssemblyPaths so that dropping a new tool DLL into this folder
    /// takes effect on the next launch without needing the Options screen's
    /// path list to be re-typed or re-synced — the folder is the source of
    /// truth, re-read every time a session actually starts, not snapshotted
    /// once when the options were first built.
    /// </summary>
    public string? ToolsFolder { get; set; }

    // Strategy selection
    public GeneratorStrategy Generator { get; set; } = GeneratorStrategy.Standard;
    public CacheStrategy Cache { get; set; } = CacheStrategy.Standard;

    /// <summary>
    /// CPU code-path selection for JigSaw's mapping. Auto (the engine's own
    /// default) genuinely detects FMA/AVX2/SSE3 support at runtime via
    /// System.Runtime.Intrinsics.X86 checks — it isn't a placeholder, it's a
    /// real, working choice. The explicit tiers exist for forcing a specific
    /// path regardless of what the CPU actually supports (testing, or
    /// deliberately stepping down from a tier that's misbehaving on a given
    /// machine).
    /// </summary>
    public HardwareTier HardwareTier { get; set; } = HardwareTier.Auto;

    /// <summary>
    /// Whether to chain MappingBuilder.WithGpu() into the launch mapping.
    /// Requires SharpMind.GPU to be referenced by whatever project starts
    /// the session — JigSaw discovers the GPU kernel entries by scanning the
    /// loaded AppDomain, so the reference is what actually makes WithGpu()'s
    /// overrides resolve to real kernels rather than no-ops.
    /// </summary>
    public bool UseGpu { get; set; }

    /// <summary>
    /// Governs tool calls that touch the file system, via the engine's own
    /// ToolPermission enum (Ask/Always/Never) — used directly as the answer
    /// SessionLauncher's permission callback gives for ToolCategory.File
    /// requests, no translation layer in between. Defaults to Ask: silently
    /// allowing file IO by default would be a worse surprise than a prompt
    /// the first time a tool actually wants it.
    /// </summary>
    public ToolPermission FileAccess { get; set; } = ToolPermission.Ask;

    /// <summary>Same as <see cref="FileAccess"/> but for ToolCategory.Network requests.</summary>
    public ToolPermission NetworkAccess { get; set; } = ToolPermission.Ask;

    // Sampling / generation
    public SamplingConfig Sampling { get; set; } = SamplingConfig.Llama3Chat;
    public GenerationConfig Generation { get; set; } = GenerationConfig.Default;

    // Agent
    public string AgentName { get; set; } = "Delta";
    public bool AgentsEnabled { get; set; }
    public int MaxAgentDepth { get; set; } = 2;
    public int MaxToolCallsPerTurn { get; set; } = 10;

    public static SessionOptions Default => new();
}
