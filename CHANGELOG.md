# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Planned next release: **1.0.3** — contains a few slightly breaking changes
that are already on `master`.

### Added

- `SessionOptions.MaxTokens` override — `null`/`0` uses the model's full context window (`MaxSeqLen`); a positive value clamps to `[1, MaxSeqLen]` to opt back into a truncated window.
- `SessionOptions.SkipAgentPrompt` — drops the whole agent layer (no synthesized agent prompt, no sub-agents, no tool loop).
- `SessionOptions.DisableTools` — keeps the agent prompt but registers no tools; the tool-call loop is additionally guarded on `RegisteredToolNames.Count > 0`.
- Options view: "Max context tokens (0 = full)" field plus "Skip agent prompt" and "Disable tools" toggles.
- Chunked prompt prefill with UI progress surfaced as "Prefilling NN.NN%" (`IGenerator<T>.PrefillProgress`, drained via `ChatSession`);
- `SessionOptions.Clone()` / `CopyTo()` — a single deep-copy path shared by every clone/preset/resume path.
- CUI error surfacing for session-launch failures.
- Quantized-resident loading — chat/inference loads keep only the raw quantized bytes and skip the per-layer dequantized F32 copies, roughly halving resident memory for a load.
- Load-time validation: weight shapes are checked against the config by name and unsupported architectures rejected by name, so a bad model fails with a clear message instead of a byte-count mismatch at the first matmul.
- Vectorised decode/prefill kernels — SIMD-widened fp16 weights (exact for normals and denormals), vectorised LM head, work-based attention parallelism, work-chunked decode, F16 matmul row blocking, cache-line-aligned row tiling, and vectorised Q8 block tails.
- Deterministic test reference data — `TinyReferenceModel` builds a seed-fixed reference `.SMM` in milliseconds so the session/CUI tests exercise the full load → chat path without loading a real model file.
- KV-cache persistence: sessions now save and restore the pre-filled KV cache alongside chat history. On resume, if the prompt (system prompt + tools + history) hasn't changed, the expensive prefill is skipped entirely — the cache is restored from disk and the first user turn extends it incrementally.
- Quantized-resident loading no longer allocates a dead full-F32 weight per inference layer — `InferenceLinearLayer` passes `allocateFullWeight: false` to the base constructor, cutting peak memory by ~28 GB on a 7B model (PR #8).
- **`SharpMind.Extensions.AI`** — new library bridging SharpMind into the `Microsoft.Extensions.AI` ecosystem:
  - `SharpMindChatClient` — `IChatClient` adapter wrapping any `IChatSession`; supports both single-shot (`GetResponseAsync`) and streaming (`GetStreamingResponseAsync`) via a channel-based bridge.
  - `ChatMessageConverter` — bidirectional `ChatMessage` mapping between MEAI and SharpMind types, plus `ChatOptions` → `IChatSession` option forwarding.
  - `AiFunctionToolAdapter` — routes MEAI `AIFunction` instances into SharpMind's `IAgentBuilder.WithTool` delegate path, so MEAI-defined tools work inside SharpMind's agent loop without a separate execution path.
- `IAgentBuilder.WithTool(name, description, schema, execute)` — delegate-based tool registration overload, enabling external callers (e.g. the MEAI adapter) to register tools with an explicit JSON Schema and async executor without reflection or `[ToolDesc]` attributes.
- **`SharpMind.Extensions.Tools`** — optional common tools package, auto-discovered from the CUI plugins folder:
  - `GrepTool` — regex/literal file-content search across a directory tree, returns matching lines with file paths and line numbers.
  - `GitTool` — read-only git command execution (status, log, diff, show, blame, branch, remote); write commands (push, commit, merge, etc.) are blocked.
  - `DateTimeTool` — current UTC/local time, time zone conversion, time zone listing.
  - CUI build automatically copies `SharpMind.Tools.dll` to the `plugins/` output folder via a post-build target.

### Changed

- `Session.MaxTokens` is now the **context-window budget** and defaults to the model's full `MaxSeqLen` instead of being capped at `MaxNewTokens`; long conversations are kept intact rather than trimmed into a token budget that silently evicted the agent/tool system prompt.
- Agent system prompt reworked to be more compact.
- BPE merge encoding rewritten; embeddings routed per-layer; NeoX rotary convention applied for architectures that need it.
- The RoPE table cache is keyed by config instead of a hash of it.
- Native buffer pooling: the pooled marker is now a CompareExchange transition (`0 → -1`), so a view racing a return-to-pool always wins and the buffer stays alive rather than being freed or re-rented out from under it.
- CUI/formatter warnings surfaced when a chosen formatter's turn markers are absent from the model vocabulary.
- Chat turns now extend the KV cache incrementally via `FeedForPrompt`: a comparison-based prefix detector finds the longest common prefix between the current prompt and the generator's cached tokens, truncates the cache, and feeds only the tail — no manual bookkeeping, no formatter-specific code paths. Any mismatch (history edit, thinking strip, tokenization drift) falls back to a full prefill automatically.
- Chat status sidebar now shows the resolved formatter name (e.g. "Llama3Formatter", "ChatMLFormatter") instead of the strategy enum value ("Auto").
- Saved sessions now include chat history — loading a saved session restores the previous conversation in the chat view instead of starting empty.

### Fixed

- CUI option cloning silently dropped fields (`UserName`, and the new knobs) on every session launch/resume — launched sessions now honor all options.
- Broken solution restore; `Transformer.DisposeCache` properly wired into disposal.
- KV cache `Snapshot` used 32-bit arithmetic that could overflow at full context windows (`KVCache`, `PagedKVCache`, `QuantizedKVCache`).
- A hallucinated `<tool_call>` no longer enters the tool loop when tools are disabled.
- Removed redundant/unused implementation code.
- Native buffer pool contamination under parallel load — a concurrent `AddRef` racing a buffer's return to the pool could free (or re-rent) a buffer a live view still held, surfacing as `ObjectDisposedException` in MoE backprop (`FfnOut.Reshape`) and foreign-bucket pops in the pool probe.
- Pooled buffers were silently re-allocated on every rent (the Rent CAS compared against the old pooled marker) — pooling now actually reuses instances past its configured capacity.
- Training linear layer data race on gradient writes under parallel backprop.
- Session loading now shows chunked "Prefilling X%" progress while encoding the system prompt, tools, and agent configuration into the KV cache — the first user turn then extends the already-warm cache instead of re-prefilling everything from scratch.
- Transcript now correctly populates when loading a saved session — `RebuildTranscript()` runs after the view is added to the layout instead of during construction, so `SetNeedsDisplay` actually takes effect; `SourceFilePath` is now carried through all load/resume paths including the welcome screen.
- Progress during session loading now renders in real time via main-loop polling (not `MainLoop.Invoke` which never drains while `await`-ing), displays two decimal places, and labels reflect whether the KV cache is being built or rebuilt.
- Chat sidebar widened by 6 characters to accommodate longer formatter names.
- Save session now offers a Save As file picker for first-time saves and asks before overwriting an existing file instead of silently replacing it.
- Q4_0 KV-cache quantization packed nibbles in half-split layout but attention kernels dequantized interleaved — swapped to interleaved packing so the two are consistent.
- `SessionOptions.CopyTo` was copying the transient `PendingSnapshot` field, causing a stale snapshot to silently resurrect old history into unrelated sessions launched later.
- Interrupting a turn that only produced `<think>` tokens left stale thinking content and a stale `_liveResponseStartOffset`, corrupting the next turn's transcript rendering.
- Swapping away from a ChatView (e.g. launching a new session while one is generating) left its 16ms poll timer running on the orphaned view; `SwapContent` now disposes removed child views.

### Removed

- Real-model diagnostic probes (`ModelSpeedProbeTests`, `RealModelPrefillDiagnosticsTests`) — the suite no longer loads a real GGUF, cutting the full run from ~22 minutes to ~1.5 minutes (≈15× faster); the chunked-prefill regression coverage lives on in reference-model-driven tests.

### Breaking

- `IGenerator<T>` gained a `PrefillProgress` member — custom/plugin generator implementations must add it (build break).
- `IGenerator<T>` gained a `Caches`, `CacheTokens`, `TruncateCache`, and `SetCacheTokens` members — custom/plugin generator implementations must expose their KV-cache array and cache-token tracking (build break).
- `ChatSession` no longer disposes a model it was handed by the caller (ownership is now explicit) — callers that relied on the session disposing the model must dispose it themselves.
- KV caches now throw `ArgumentOutOfRangeException` where a buffer/stride would overflow `int` instead of silently truncating/overflowing.

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