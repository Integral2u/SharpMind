using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

/// <summary>Well-known constants for the SharpMind Model (.SMM) container.</summary>
public static class SmmConstants
{
    /// <summary>File magic — "SMM1" as a little-endian uint32.</summary>
    public const uint Magic = 0x314D4D53;

    /// <summary>Current container version.</summary>
    public const uint Version = 1;

    /// <summary>Byte alignment for the tensor data region.</summary>
    public const int DefaultAlignment = 64;

    /// <summary>Fixed header size in bytes (2 u32 + 7 u64).</summary>
    public const int HeaderSize = 64;

    /// <summary>KvPair key holding the full serialised <see cref="SharpMind.Model.Config.ModelConfig"/>.</summary>
    public const string ConfigKey = "smm.config_json";

    /// <summary>KvPair key holding the embedded SharpMind tokenizer JSON.</summary>
    public const string TokenizerKey = "smm.tokenizer_json";

    /// <summary>KvPair key holding the embedded default system prompt.</summary>
    public const string SystemPromptKey = "smm.system_prompt";

    /// <summary>KvPair key holding the embedded skills (JSON array of markdown strings).</summary>
    public const string SkillsKey = "smm.skills";
}

/// <summary>
/// Which optional sections to emit in an .SMM container. The model metadata /
/// config JSON is always written — the loader needs it to reconstruct the
/// <see cref="SharpMind.Model.Config.ModelConfig"/>.
/// </summary>
[Flags]
public enum SmmOutputs
{
    None = 0,

    /// <summary>Embed the tokenizer JSON so the file is self-contained.</summary>
    Tokenizer = 1,

    /// <summary>Embed the chat template string (used for Jinja/ChatML formatting).</summary>
    ChatTemplate = 2,

    /// <summary>Embed plugin assemblies.</summary>
    Plugins = 4,

    Default = Tokenizer | ChatTemplate,
    All = Tokenizer | ChatTemplate | Plugins,
}

/// <summary>A plugin assembly packaged inside an .SMM container.</summary>
public sealed class SmmPluginEntry
{
    /// <summary>Assembly file name (e.g. "SharpMind.Plugins.Weather.dll").</summary>
    public required string Name { get; init; }

    /// <summary>Raw assembly bytes, loaded via <c>Assembly.Load(byte[])</c> on activation.</summary>
    public required byte[] AssemblyBytes { get; init; }

    /// <summary>True when the plugin's tools are marked as recommended in the CUI.</summary>
    public bool Recommended { get; init; }
}

/// <summary>Write-time options for <see cref="SmmWriter"/>.</summary>
public sealed class SmmWriteOptions
{
    /// <summary>Byte alignment for the tensor data region.</summary>
    public int Alignment { get; init; } = SmmConstants.DefaultAlignment;

    /// <summary>Plugin assemblies to embed in the container. Optional.</summary>
    public IReadOnlyList<SmmPluginEntry>? Plugins { get; init; }

    /// <summary>
    /// Which optional sections to emit. Defaults to <see cref="SmmOutputs.Default"/>.
    /// </summary>
    public SmmOutputs Outputs { get; init; } = SmmOutputs.Default;

    /// <summary>
    /// Target weight dtype for a training (F32) export. <see langword="null"/> or
    /// <see cref="QuantDType.F32"/> keeps the trained weights at full float —
    /// the safe default. <see cref="QuantDType.F16"/> halves size with no block
    /// layout concerns. <see cref="QuantDType.Q8_0"/> / <see cref="QuantDType.Q4_0"/>
    /// are supported only when every tensor dimension is a multiple of 32 (the
    /// block layouts the quantized kernels expect); otherwise a clear error is
    /// thrown and F16/F32 is recommended. Tensors that are already quantized
    /// (e.g. GGUF conversion) are never re-quantized.
    /// </summary>
    public QuantDType? QuantizationLevel { get; init; }

    /// <summary>Diagnostic source label written into the metadata ("training", "gguf", ...).</summary>
    public string? Source { get; init; }

    /// <summary>
    /// Embedded skills. Each entry is one full skill document (markdown), emitted
    /// as the <c>skills</c> array in the meta JSON and silently auto-applied to
    /// the agent when this .SMM is opened. Optional.
    /// </summary>
    public IReadOnlyList<string>? Skills { get; init; }

    /// <summary>
    /// Embedded default system prompt. Emitted as <c>system_prompt</c> in the
    /// meta JSON and injected as an additional system message at the top of the
    /// chat when this .SMM is opened. Optional.
    /// </summary>
    public string? SystemPrompt { get; init; }
}

/// <summary>
/// One tensor to be written to an .SMM container. The raw bytes are fetched
/// lazily via <see cref="GetBytes"/> so a GGUF→SMM converter can stream
/// quantized data straight from the source file instead of holding it all in
/// memory.
/// </summary>
public sealed class SmmTensorData
{
    /// <summary>GGUF-compatible tensor name (e.g. "blk.0.attn_q.weight").</summary>
    public required string Name { get; init; }

    /// <summary>Tensor shape, GGUF layout ([in, out] for 2D weights).</summary>
    public required int[] Shape { get; init; }

    public required QuantDType Dtype { get; init; }

    /// <summary>Returns the raw tensor bytes — exactly what GGUF stores on disk.</summary>
    public required Func<byte[]> GetBytes { get; init; }
}
