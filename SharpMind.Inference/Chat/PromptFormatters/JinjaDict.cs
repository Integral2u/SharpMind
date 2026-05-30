namespace SharpMind.Inference.Chat.PromptFormatters;

// ── Supporting types ──────────────────────────────────────────────────────────

/// <summary>Dictionary representing a Jinja template object (message, loop, etc.).</summary>
public sealed class JinjaDict : Dictionary<string, object?> { }
