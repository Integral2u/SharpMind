# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.5.0] - unreleased

### Added

- **Accelerator plugin seam** — `IAcceleratorPlugin` in `SharpMind.Core` (`Name`, `Description`, `Capabilities` picked by type) with `AcceleratorLoader` scanning the plugins folder recursively; `IMappingOverrides` for kernel-level (JigSaw) substitution; `ITrainingEngine` / `ITrainingEngineFactory` / `TrainingEngineContext` in `SharpMind.Training` for a whole-step, device-resident training engine. `TrainLoop` accepts an `engine:`; the default `CpuTrainingEngine` is the previous step verbatim. Training jobs (`.smmt`) gain `Accelerator` and the wizard an "Accelerator:" selector listing discovered plugins; a name that cannot be honoured fails the run instead of silently training on the CPU. No accelerator code ships in the main line.
- **SharpMind.Server** — OpenAI-compatible HTTP server (`SharpMind.Server` class library + `SharpMind.Server.CLI` executable). Exposes `/v1/chat/completions` (streaming and non-streaming), `/v1/models`, `/v1/models/loaded`, `/v1/models/{model}`, `/v1/models/{model}/load`, `DELETE /v1/models/{model}`, `/v1/health`, and `/v1/shutdown`. Models load lazily on first request, cache with ref-counting, and stay resident until explicitly unloaded. CLI spawns the service as an orphaned process and provides an interactive REPL with `/model`, `/models`, `/loaded`, `/unload`, `/unloadall`, `/clear`, `/restart`, `/stop`, `/exit` commands. Flags: `--models`, `--host`, `--port`, `--model` (comma-separated), `--stop`, `--nocli`, `--no-files`, `--no-network`.
- **Ministral-3-3B-Instruct support** — full sliding-window attention (SWA) for ministral/mistral3 architectures. All 12 SDPA kernels updated with per-layer window masking; KV-cache capped to the model's sliding window; RoPE tables sized accordingly; GgufLoader fallback defaults to 4096 when the GGUF omits the key.
- **Tool-dispatch interception seam** (`IChatSession.ProcessToolRequest`) — a single host-facing point where a caller decides, per tool call, what happens to it. The seam is a `Func<string, JsonObject, CancellationToken, Task<ToolRequestResult>>` returning one of three outcomes per call: `Handled(result)` (the host ran the tool; the session feeds the result back and continues the agentic loop), `Defer` (the session dispatches natively through its agent loop with File/Network gating), or `ReturnToCaller` (the session hands the whole call back to the caller and ends the turn, surfacing it as a new `ChatStatus.ToolCall` + `ChatStreamEntry.ToolCall` entry without dispatching or feeding a result back). When the seam is left null the session's built-in native dispatch is unchanged. `SharpMindChatClient` wires the seam to the MEAI tool adapter (MEAI-registered functions are host-dispatched; native SharpMind tools defer), matching the old behavior with no regression.
- `IChatSession.StopStrings` property — allows callers to set text-based stop sequences on a session. `ChatSession` merges formatter defaults with session-level stop strings into `GenerationConfig.StopStrings` at generation time.
- `IChatPromptFormatter.MergeStopStrings` helper in `ChatSession` — deduplicates formatter default stop strings with caller-supplied stop strings.
- **Fused QKV tensor support** — `GgufLoader.LoadSingleTensor` now detects a single `attn_qkv` weight and splits it into separate Q/K/V projections, enabling models that export attention as a fused matrix.
- **Q1_0 quantization (1-bit)** — new `QuantDType.Q1_0` (dtype 41) with sign-only quantization: `value = (2*bit - 1) * d`. VecDot kernels (scalar, FMA, AVX2, AVX512) and QuantizedMatMul dispatch added.
- **`ComputeMaxCacheLength`** — static helper on `ModelConfig` that caps KV-cache size to 40% of available system memory, with optional user override. Prevents OOM on large models with high head counts.
- **`--max-cache-len` CLI option** — allows manual KV-cache cap override, threaded through generator builders to `ChatSession`.
- **Streaming layer sentinel** — `EnsureLayerLoadedSync`, `PreloadLayerAsync`, `FreeLayer`, and `CompleteForward` now check `Wq == null && RawWq == null` instead of just `Wq == null`, correctly detecting layer-loaded state for models with fused QKV where the float tensor is never allocated.
- **Chat sidebar model name** — the chat screen's status sidebar pins the loaded model's file name (extension stripped) to its bottom row so it stays visible while chatting; debug sessions show `(none — UIDebug mode)`. Same name source as the session manager's "Continue with" button.
- **Phi-3-mini-4k-instruct support** — validated end to end (load → fused-QKV prefill → decode, Q4_K, `Phi-3-mini-4k-instruct-q4.gguf`) and added to the README compatibility matrix. Phi-3's GGUF lays out a fused `attn_qkv.weight` and a fused gate+up FFN stored under `ffn_up` with no `ffn_gate` tensor; together the two fused-tensor corrections above and below are what make it load and generate.
- **`GpuStepProfiler`** — per-phase wall-clock breakdown of a GPU training step, reached as `GpuBackpropEngine.Profiler` and enabled by `SM_PROF=1` or by setting `Enabled`. Every mark synchronises the device, because kernel launches are asynchronous and an unsynchronised phase's cost lands on whichever later phase happens to block; a profiled step is therefore slower than a real one, and throughput must be read from an unprofiled run. Disabled — the default — a mark is a single bool test, so the calls stay in the shipped engine and a breakdown never needs a rebuild of a plugin that is loaded from a path.
- **`tools/GpuTrainBench`** — measures the wall-clock cost of a GPU LoRA training step at a configurable shape (`--batch`, `--seq`, `--steps`, `--warmup`, `--layers`, `--flash`, `--prof`). Times and profiles in separate passes over the same warmed engine so the reported ms/step is never a profiled one.
- **Accelerator inference engines** — `IInferenceEngine` / `IInferenceEngineFactory` / `InferenceEngineContext` in `SharpMind.Inference` model GPU accelerate inference the same way `ITrainingEngine` does training: an engine owns the numerics and KV cache while a generator (`EngineGenerator`) handles sampling, streaming and stop strings. `IAcceleratorPlugin.Capabilities` now also advertises `IInferenceEngineFactory`; the CUDA plugin ships `GpuInferenceEngineFactory` with a hybrid hybrid first-turn GPU prefill → host-CPU decode. `InferenceEngineResolver` resolves an engine by plugin name from the plugins folder, honouring the choice or failing the launch — never silently falling back to the CPU. `SessionOptions.InferenceAccelerator` (persisted with the session) wires a session onto an accelerator engine; `ChatSession` special-cases engine-backed sessions for snapshot/restore via `ExportCache`/`ImportCache`. Continued-turn prefill and all decode run on the CPU against a host KV cache materialized from the GPU prefill; true GPU decode is (deliberately) a follow-up.
- **Plugin renamed `cuda` → `ilgpu`** — `CudaAcceleratorPlugin` is now `IlgpuAcceleratorPlugin` (`Name == "ilgpu"`, plugin folder `Plugins/ilgpu/`); it is the cross-vendor ILGPU backend (OpenCL devices work too), not NVIDIA-only. The legacy `cuda` name still resolves as an alias in both `TrainingEngineResolver` and `InferenceEngineResolver`, so stored `.smmt` jobs and session presets from before the rename load unchanged.
- **Consent-based device fallback** — when an accelerator that can't actually run here is chosen (e.g. `ilgpu` selected but no CUDA/OpenCL device present), the CUI now asks the user how to proceed instead of failing the run or launching silently: the dialog lists the CPU (guaranteed) plus every discovered plugin that offers the matching factory capability, and the choice is stored under the plugin's canonical name.
- **GPU inference kernels for K-quants (Q2_K..Q6_K)** — the ILGPU inference path now dequantizes the K-quant families on the device (`Q2_K`..`Q6_K`), extending GPU inference past Q8_0 to the K-quant block types. `Q8_K` deliberately has no device kernel: its F32 super-block scale cannot be reinterpreted in ILGPU, making it the canonical host-routed quant instead.
- **Per-tensor CPU fallback for GPU-incompatible content** — an accelerator can now run a mixed GPU+CPU model instead of refusing the whole thing. Quants without an on-device kernel (notably `Q8_K`, and any deferred K-quant) route their `blk.*` linears through the host: `GpuLinear` downloads the input, runs the layer's own CPU quantized `Forward` on the raw bytes (the exact call the CPU transformer makes, bias included), and uploads the result — while every other tensor keeps its GPU kernel. `GpuInferenceEngine.CheckSupported` no longer refuses any quant; only genuine architecture limits (MoE, LayerNorm-final, dense FFN, non-RoPE, zero blocks) still refuse. `IInferenceEngineFactory.DescribeCpuFallback` (a defaulted, backward-compatible member) reports which dtypes would fall back and to what backend, and the GPU factory delegates to a static that filters to dtypes actually sitting on `blk.` linears.
- **Pre-load CPU-fallback consent gate** — before the weight read (fail-fast, so a model can't half-load and then stop), the CUI session launcher asks how to proceed when the chosen accelerator needs host fallback: the dialog's first option is **Allow CPU fallback** (keep the accelerator, run the fallen-back tensors on the CPU), with the existing CPU and plugin rows after it. Consent sets a transient `SessionOptions.AllowCpuFallback` flag (deliberately excluded from `Clone`/`CopyTo` and settings persistence, so every fresh launch asks again) and the load retries past the metadata gate without re-prompting. The pre-load gate distinguishes a real architecture refusal (`ModelLoadResult.AcceleratorRefusal`) from a mere quant fallback it can proceed with once consented (`CpuFallbackWarning`) — neither is a hard error like a file/parse/tokenizer failure. Non-CUI callers default silently to the mixed path, and boundary tensors (embedding, LM head) never fall back because the loader always keeps F32 copies for them.
- **Standalone SharpMind CLI Setup installer** — `SharpMind CLI Setup/` packages the `SharpMind.Server.CLI` executable (via its `PublishItems` output group) into its own MSI with a distinct product identity, so the OpenAI-protocol server CLI installs separately from the interactive console app. The console setup's `ProductVersion` is likewise kept in sync with the assembly/NuGet version prefix; Windows Installer only supports `major.minor.build` (the 4th revision lives in the assembly metadata, not the MSI).
- **Engine-usage display** — `ITrainingEngine.Description` and `IInferenceEngine.Description` report the actual backend in use (e.g. `[Cuda] GTX 1060, cuBLAS 12.8`, `OpenCL`, or `CPU`). Training shows it on the training-progress screen and chat shows it in the status sidebar, so OpenCL vs. ILGPU-CUDA vs. cuBLAS vs. CPU is always visible.
- **Backend hint in accelerator selectors** — the Options and training-wizard accelerator rows now show which backend the ILGPU path would actually run on this machine (e.g. `ilgpu (OpenCL)`, `ilgpu (CUDA · cuBLAS)` or `ilgpu (CUDA)`), not just a bare plugin name, via a new `IBackendHintProvider` plugin capability probed once per process by `GpuDevice`. The hint is presentation only — the stored value, job/preset matching and legacy `cuda` alias all keep the bare canonical plugin name.

### Changed

- **Attention runs on cuBLAS** — `GpuKernels.AttnFwd`/`AttnBwd` now issue scores, output, dP, dQ, dK and dV as GEMMs instead of the hand-written thread-per-row kernels, which were 85% of a training step at the SmolLM2-135M shape (`AttentionKernels.BwdKV` alone 59%, occupancy-bound at `batch·numKv·seqLen` threads). `AttentionKernels` keeps only the two genuine row reductions, `SoftmaxRow` and `SoftmaxRowBwd`; both write the masked upper triangle as zero because the GEMMs fill and read the whole S×S block, which also removes the forward's pre-zero of the probabilities. `FlashAttentionKernels` is unchanged and remains the slower path. Measured on a GTX 1060 with cuBLAS 12.8 at 134M params, LoRA r=8, seq 256: attention 2339 → 75 ms/step (31×) and the step 2760 → 444 ms at batch 2, 182 → 1153 tok/s.
- `GpuDevice.Gemm` takes an `ldc` row stride for C (0 = dense = `n`), so one attention head's `[seqLen, headDim]` result can be written directly into a `[batch·seqLen, heads·headDim]` tensor whose rows sit a head-stride apart. Threaded through `Cublas.GemmRowMajor` and the `tiled16` ILGPU fallback.
- `GpuKernels` holds its `GpuDevice` so the attention entry points can issue GEMMs through it.

### Removed

- **`SharpMind.Benchmarks` project** — removed from the solution. It was the sole consumer of BenchmarkDotNet but was unreferenced by any other project: its active `KernelsBenchmarks` was trivial, the VecDot/HSum256 benchmarks were commented out and used a stale net9 job moniker, `BenchToConfig` produced a `qconfig.json` with no consumer in the codebase, and the `Evaluation/` metrics (perplexity/BLEU/F1) were referenced nowhere and untested. Contrast with the retained `tools/GpuTrainBench`, which has no equivalent parity tool.
- **`SharpMind.GPU.Tests` project** — folded into the main `SharpMind.Tests` suite as the `SharpMind.Tests.GPU` namespace (files under `SharpMind.Tests/GPU/`). The standalone project was removed from the solution and `SharpMind.Tests` now references `SharpMind.GPU` (transitively ILGPU), so the main suite is no longer accelerator-free — the GPU collection runs via `GpuDevice.Shared`, falling back to ILGPU's CPU accelerator when no GPU is present. Its tests were updated for the merge: namespaces renamed, files that relied on enclosing-namespace resolution of `SharpMind.GPU` types gained explicit `using SharpMind.GPU;`, `SharpMind.GPU`'s `InternalsVisibleTo` now points at the `SharpMind.Tests` assembly, and `AcceleratorPluginTests.Plugin_IsDiscoverableThroughTheHostsOwnLoader` scans an isolated temp folder (copied from the output minus `SharpMind.Tests.dll`) so the host's own `AcceleratorLoader` no longer picks up the main suite's plugin fakes.

### Fixed

- **`NativeBufferPoolTests` false failure under the merged GPU suite** — `Rent_Reuses_SameInstance_WellBeyond_BucketCapacity` asserts the double bucket keeps reusing the same instance, but `NativeBufferPoolConfig.MaxTotalMemoryMB` is a global across every pooled type. With the GPU tests now in-process their cumulative native-float allocations drive `TotalMemoryUsed` toward/over the 512 MB cap, so `Return()` freed the ~8 MB probe buffer instead of pooling it — a spurious "double-increment bug" failure at cycle 0. The test now pins the cap high for its duration (restoring it after, via `finally`) so reuse is decided purely by `bucket.Count`.
- **`SharpMindChatClient` leaked tool-call markup as assistant text and dropped returned tool calls.** The session streams every generated fragment as it arrives and only parses the completed buffer for a `<tool_call>` afterwards, so the raw markup reached the adapter as ordinary `Responding` text and was forwarded to the MEAI caller verbatim — visible in the default wiring, where the native loop dispatches the tool and carries on, as `<tool_call>{"tool":…}</tool_call>` sitting in front of the reply. The adapter now suppresses a completed tool-call block, holds a partial opening tag until the next fragment resolves it, and releases trailing prose unchanged. Separately, a call handed back by `ToolRequestOutcome.ReturnToCaller` arrived as a `ChatStatus.ToolCall` entry whose `Token` is the tool *name*: that name was appended to the assistant text and the call itself discarded, so the turn finished `stop` with no `FunctionCallContent` and no tool middleware could ever see it. It is now mapped to `FunctionCallContent` with `ChatFinishReason.ToolCalls`. `GetResponseAsync` aggregates the streaming path rather than duplicating it, so both forms behave identically.
- **Multi-part message content passed as raw JSON** — `OpenAiMapper.ExtractTextContent` now parses JSON array content (`[{"type":"text","text":"..."}]`) sent by modern OpenAI client libraries, extracting and concatenating text parts instead of passing the raw JSON string to the model as if it were the user's message.
- **Stop sequences silently ignored** — `OpenAiMapper.ApplyToSession` no longer attempts `int.TryParse` on stop strings (which silently dropped all non-numeric values). Stop strings from the OpenAI request are now mapped to `IChatSession.StopStrings` and merged into `GenerationConfig.StopStrings` at generation time.
- **Progress tick display** — service-side `CreateProgress` now calls `Output.Flush()` after writing `\r`-prefixed tick text, ensuring data hits the pipe immediately. CLI-side `PipeStreamAsync` now flushes accumulated text at the end of each read batch, so the final progress tick (e.g. `100.00%`) is always visible instead of being stuck in an unflushed buffer.
- **Progress line overlap** — status messages following a tick (e.g. "Creating transformer...") now advance past the tick line with `\n` before writing, preventing them from appearing on the same line as the progress percentage.
- **Prefill progress leaked to client** — the streaming handler in `SharpMindService` now skips `ChatStatus.Updating` entries (prefill progress like "Prefilling 92.75%") so they are not sent as SSE content to the CLI client.
- **Streaming KV weight typo** — `TransformerWeights` streaming path wrote `RawVv` instead of `RawWv` for the V projection, causing the attention layer to default to F32 for V weights on every forward pass.
- **Non-block dequant memory** — `GgufLoader.LoadSingleTensor` now reads non-block tensors (embedding, lm_head, norms) directly into the target tensor's backing array instead of allocating a temporary buffer, eliminating a large transient allocation for models with very large embedding tables.
- **Inner exception chain in error dialog** — the CUI error dialog now walks `InnerException` showing each exception type, message, and stack trace, instead of displaying only the top-level message.
- **Fused gate+up FFN support** — `FfnLayer` now handles GGUFs that store the gated FFN as a single fused `[HiddenDim, 2*FfnDim]` tensor under only `ffn_up` (no separate `ffn_gate`, e.g. Phi-3): `SetWeights` uses `RawWup` directly when `RawWgate` is absent, and the gated projection's quant dtype comes from the nullable `QuantDtypeWgate`/`QuantDtypeWup` properties that `SetRawField` fills. Previously the dtype was read with `TensorMeta.GetValueOrDefault(...)`; `TensorMeta` is a record struct, so a missing key returned the default (`F32`, never null) and the `RawWup` fallback was dead code — `SetRawWeight` then validated the fused Q4_K tensor's 28,311,552 bytes against an F32 shape expecting 201,326,592 and threw `NotSupportedException`. `ResolveFloatTarget` lazily initializes `Wf1` so the float fallback path has valid data.

## [1.0.4.0] - 2026-08-23

### Added

- **Limit Breaker** — removed the int.MaxValue element-count ceiling from the inference and loading paths:
  - `BigArray<T>` — paged array backed by `ArrayPool<T>.Shared` segments, supporting more than int.MaxValue elements with long-indexed access and int-sized block iteration for SIMD kernels.
  - `IWorkspace` interface extracted from `Workspace`, enabling pluggable workspace implementations.
  - `BigWorkspace` — workspace variant for oversized contexts (throws for >int.MaxValue deferred to a future release).
  - `MemoryHelpers` — central routing layer for all memory allocations: `RentArray<T>`, `ReturnArray<T>`, `Rent<T>` (auto-selects array vs BigArray based on count), `CreateWorkspace`, `RentBuffer<T>`, `ReturnBuffer<T>`. All ArrayPool and NativeBufferPool calls in production code now route through this single entry point.
  - `TensorLoadHelper` — shared helper for computing element counts (`ComputeElementCount`, `ComputeElementCountChecked`) and guarding long→int casts (`CheckedInt`) during model loading.
  - 20 unit tests covering BigArray, BigWorkspace, and MemoryHelpers.
- SmmLoader int-overflow bug fixed — the element-count computation `int count = 1; foreach (int d in shape) count *= d;` silently wrapped negative for large tensors, causing `ArrayPool.Rent(-...)` crashes downstream. Now uses `TensorLoadHelper.ComputeElementCountChecked` with an explicit guard.
- Oversized tensor guards added to both loaders — LmHead creation, rawSize casts, and colBytes casts in GgufLoader and SmmLoader now throw `NotSupportedException` with a clear message instead of silently overflowing.
- Oversized tensors in both loaders now degrade gracefully instead of crashing — when element count exceeds int.MaxValue, raw quantized bytes are loaded but dequantization is skipped, matching the streaming-mode precedent. The streaming forward pass reads the raw bytes directly.
- **Training sample with actual data** — `SharpMind.Samples/Training/Acutal/` now ships a complete, reproducible training run: the Shakespeare corpus (`shakespeare.txt`, 40K lines), the training job config (`shakespeare-job.smmt`), and the resulting trained checkpoint (`shakespeare.smm`). Run `SmmRealTextExample` to reproduce the end-to-end pipeline from raw text to a chat-capable .SMM model.
- **Actual training checkpoint** — `shakespeare.smm` is a pre-trained GPT-2-style model (HiddenDim=384, 6 layers, 6 heads, MaxSeqLen=256) trained on the Shakespeare corpus for 600 steps. Load it directly in the CUI or via `SmmTrainingPipeline.LoadForInference` to chat with a model that was genuinely trained (not just loaded from a GGUF).

### Changed

- All layer/model method signatures changed from `Workspace?` to `IWorkspace?` across 36 method signatures in 20 files (IArchitecture, Transformer, TransformerBlock, AttentionLayer, FfnLayer, LinearLayer, NormLayer, LogitOps, ActivationOps, EmbeddingTable, and their implementations).
- Generator workspace creation (`StandardGenerator`, `SpeculativeGenerator`, `MedusaGenerator`) now uses `MemoryHelpers.CreateWorkspace` instead of `new Workspace(...)`, routing through the central allocation layer.
- `Prefill.ForwardLastLogitsChunked` internal parameter type changed from `Workspace` to `IWorkspace`.
- `Tensor` constructor routes `NativeBufferPool<T>.Rent` through `MemoryHelpers.RentBuffer<T>`.
- `Sampler` all ArrayPool calls (6 Rent + 6 Return) routed through `MemoryHelpers.RentArray<T>` / `MemoryHelpers.ReturnArray`.
- `GgufLoader` and `SmmLoader` ArrayPool calls routed through `MemoryHelpers.RentArray<float>` / `MemoryHelpers.ReturnArray`.
- GgufLoader manual long guard for element count replaced with `TensorLoadHelper.ComputeElementCountChecked`.

### Fixed

- `WarmupPrefillAsync` now trims the encoded prompt to `MaxTokens` (= `MaxSeqLen`) before prefill, preventing RoPE overflow on small-context trained SMM checkpoints (e.g. MaxSeqLen=256) where the agent/system prompt exceeds the model's context window.
- `TrimToFitContext` negative-capacity crash fixed — the importance-scored message-level eviction loop computed `new List<ChatMessage>(_history.Count - removed.Count)` where `removed` tracked original indices but `_history` shrank each iteration, eventually producing a negative capacity. Removed the capacity hint and added a no-progress break so the loop falls through to Phase 2 token-level truncation when message eviction can't shrink enough.
- Sliding-window decode position bug fixed in all three generators (`StandardGenerator`, `SpeculativeGenerator`, `MedusaGenerator`) — after `TrimToLast` rewound the KV cache, the decode position was still computed from pre-trim bookkeeping (`posOffset + promptLen + step` / `currentPos`), causing RoPE to receive an offset past `MaxSeqLen`. All three generators now derive position from `_caches[0].Length` (the live cache length) after any trim operation.

## [1.0.3.0] - 2026-08-22
contains a few slightly breaking changes.

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
[1.0.5.0]: https://github.com/Integral2u/SharpMind/releases/tag/v1.0.5.0
[1.0.4.0]: https://github.com/Integral2u/SharpMind/releases/tag/v1.0.4.0
[1.0.3.0]: https://github.com/Integral2u/SharpMind/releases/tag/v1.0.3.0
[1.0.0.0]: https://github.com/Integral2u/SharpMind/releases/tag/v1.0.0.0