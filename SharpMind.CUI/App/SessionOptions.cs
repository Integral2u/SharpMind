using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.CUI.App;

/// <summary>Which IChatPromptFormatter strategy to build the session with.</summary>
public enum FormatterStrategy
{
    /// <summary>null, will default to <see cref="SharpMind.Inference.Chat.PromptFormatters.ChatPromptFormatterFactory"/>.</summary>
    Auto,
    /// <summary><see cref="SharpMind.Inference.Chat.PromptFormatters.JinjaTemplateFormatter"/>.</summary>
    Jinja,
    /// <summary><see cref="SharpMind.Inference.Chat.PromptFormatters.ChatMLFormatter"/>.</summary>
    ChatML,
    /// <summary><see cref="SharpMind.Inference.Chat.PromptFormatters.SimpleFormatter"/>.</summary>
    Simple,
    /// <summary>"Q:{prompt} A:" format <see cref="SharpMind.Inference.Chat.PromptFormatters.QuestionAnswerFormatter"/>.</summary>
    QuestionAnswer,
    /// <summary>Direct Prompt <see cref="SharpMind.Inference.Chat.PromptFormatters.RawTemplateFormatter"/>.</summary>
    Raw
}
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

/// <summary>Which IContextCompactor strategy to use for context management.</summary>
public enum CompactorStrategy
{
    /// <summary>No compaction. History is never compacted.</summary>
    None,
    /// <summary>Mark oldest non-pinned messages as ignored when token budget is exceeded.</summary>
    Truncate,
    /// <summary>Summarize older messages using the session's own model.</summary>
    Summarize
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

    public FormatterStrategy Formatter { get; set; } = FormatterStrategy.Auto;
    /// <summary>How model weights are loaded — Full (all at once, shared) or Streaming (per-layer, isolated).</summary>
    public LoadMode LoadMode { get; set; } = LoadMode.Full;

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

    /// <summary>Override non-quantized operations (pointwise, gate, softmax, RMSNorm) on GPU.</summary>
    public bool GpuNonQuant { get; set; } = true;

    /// <summary>Override quantized vector dot operations on GPU.</summary>
    public bool GpuVecDot { get; set; }

    /// <summary>Override quantized matrix multiplication operations on GPU.</summary>
    public bool GpuMatMul { get; set; }
    public bool UseParallelKernels { get; set; } = true;

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
    public string UserName { get; set; } = "User";
    public CompactorStrategy Compactor { get; set; } = CompactorStrategy.None;

    /// <summary>When non-null, overrides <see cref="Compactor"/> with a plugin-loaded compactor identified by name.</summary>
    public string? PluginCompactorName { get; set; }

    public bool AgentsEnabled { get; set; }
    public int MaxAgentDepth { get; set; } = 2;
    public int MaxToolCallsPerTurn { get; set; } = 10;

    /// <summary>Whether to show the model's internal thinking process in the UI.</summary>
    public bool ShowThinking { get; set; } = true;

    /// <summary>
    /// Whether to set <c>enable_thinking</c> in the chat template (Qwen3-style).
    /// Default <see langword="false"/> so the model emits an empty reasoning block
    /// and answers directly instead of streaming chain-of-thought.
    /// </summary>
    public bool EnableThinking { get; set; }

    /// <summary>Set of tool names that should be disabled for this session.</summary>
    public HashSet<string> DisabledTools { get; set; } = [];

    /// <summary>Set of pre-processor names that should be disabled for this session.</summary>
    public HashSet<string> DisabledPreProcessors { get; set; } = [];

    /// <summary>Set of post-processor names that should be disabled for this session.</summary>
    public HashSet<string> DisabledPostProcessors { get; set; } = [];

    public static SessionOptions Default => new();
}
