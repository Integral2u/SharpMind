# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Planned next release: **1.0.1.0** — contains a few slightly breaking changes
that are already on `master`.

### Added

- `SessionOptions.MaxTokens` override — `null`/`0` uses the model's full context window (`MaxSeqLen`); a positive value clamps to `[1, MaxSeqLen]` to opt back into a truncated window.
- `SessionOptions.SkipAgentPrompt` — drops the whole agent layer (no synthesized agent prompt, no sub-agents, no tool loop).
- `SessionOptions.DisableTools` — keeps the agent prompt but registers no tools; the tool-call loop is additionally guarded on `RegisteredToolNames.Count > 0`.
- Options view: "Max context tokens (0 = full)" field plus "Skip agent prompt" and "Disable tools" toggles.
- Chunked prompt prefill with UI progress surfaced as "Prefilling NN.NN%" (`IGenerator<T>.PrefillProgress`, drained via `ChatSession`); optional timing trace with `SHARPMIND_PREFILL_TRACE=1` → `%TEMP%\prefill_trace.log`.
- `SessionOptions.Clone()` / `CopyTo()` — a single deep-copy path shared by every clone/preset/resume path.
- CUI error surfacing for session-launch failures.

### Changed

- `Session.MaxTokens` is now the **context-window budget** and defaults to the model's full `MaxSeqLen` instead of being capped at `MaxNewTokens`; long conversations are kept intact rather than trimmed into a token budget that silently evicted the agent/tool system prompt.
- Agent system prompt reworked to be more compact.

### Breaking

- `IGenerator<T>` gained a `PrefillProgress` member — custom/plugin generator implementations must add it (build break).
- `Prefill.ForwardLastLogitsChunked` now takes a prefill-progress callback parameter — direct callers are affected (build break).
- `ChatSession` no longer disposes a model it was handed by the caller (ownership is now explicit) — callers that relied on the session disposing the model must dispose it themselves.
- KV caches now throw `ArgumentOutOfRangeException` where a buffer/stride would overflow `int` instead of silently truncating/overflowing.

### Fixed

- CUI option cloning silently dropped fields (`UserName`, and the new knobs) on every session launch/resume — launched sessions now honor all options.
- Broken solution restore; `Transformer.DisposeCache` properly wired into disposal.
- KV cache `Snapshot` used 32-bit arithmetic that could overflow at full context windows (`KVCache`, `PagedKVCache`, `QuantizedKVCache`).
- A hallucinated `<tool_call>` no longer enters the tool loop when tools are disabled.
- Removed redundant/unused implementation code.

## [1.0.0.0] - 2026-08-16

Initial release of SharpMind.

### Added

- **Console User Interface** — full terminal chat client (model browser, session manager, settings, file picker, plugin loading, permission gate UI).
- **Model loading** — GGUF and SMM loading, in full (`LoadMode.Full`) and streaming (`LoadMode.Streaming`, memory-mapped, layer-at-a-time) variants.
- **Inference** — standard, quantized, and Medusa decoding, plus speculative decoding, behind a common `IGenerator<T>` / `IGeneratorBuilder` interface.
- **GPU kernels** for common quant types via `SharpMind.GPU` (ILGPU-backed, opt-in).
- **Terminal chat app with agent tooling** — tool-calling, permission gating (`Never` / `Ask` / `Always`), and sub-agent orchestration.
- **Training** — autograd engine, optimizers & schedulers, LoRA fine-tuning, `ModelSizer`, synthetic data, and composable data sources (CSV, JSONL, text, HuggingFace datasets-server streaming).
- **Quantization** — GGUF-style coverage Q2_K through Q8_0/Q8_1/Q8_K, classic block types (Q4_0/Q4_1/Q5_0/Q5_1), and 1-bit/ternary formats (IQ1_S, IQ1_M, TQ1_0, TQ2_0), with scalar/SSE/AVX2/FMA and GPU kernel variants.
- **Conversion** — model conversion tooling.
- **Documentation and getting-started guides** — quick-start, compatibility matrix, and deep dives.
- **JigSaw kernel dispatch** — pluggable hardware-specific kernel variants selected at runtime without `if`/`switch` ladders.

[Unreleased]: https://github.com/Integral2u/SharpMind
[1.0.0.0]: https://github.com/Integral2u/SharpMind/releases/tag/v1.0.0.0