<p align="center">
  <img src="SharpMind/sharpmind_logo.svg" alt="SharpMind logo" width="160"/>
</p>

<h1 align="center">SharpMind</h1>
<p align="center"><b>A pure C# / .NET LLM engine — inference, training, and agent tooling in one solution.</b></p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-pre--release-orange">
  <img alt="lang" src="https://img.shields.io/badge/language-C%23%20(.NET%2010)-239120">
  <img alt="deps" src="https://img.shields.io/badge/dependencies-near--zero-blue">
</p>

> **⚠️ Pre-release.** SharpMind is under active development. APIs, config formats, and project layout are still moving. Inference is the most mature and best-tested part of the stack today; training is functional but earlier-stage and evolving fastest. Expect breaking changes between commits until a tagged `v0` release lands.

---

## What is SharpMind?

SharpMind is an end-to-end LLM stack written entirely in C#, with no dependency on llama.cpp, PyTorch, or any native runtime for its core path. It loads GGUF models, runs quantized CPU/GPU inference with modern decoding acceleration (speculative + Medusa-style drafting), and — unusually for a C# inference engine — also includes its own autograd engine so you can fine-tune (LoRA), distill, and prune models in the same process that serves them.

It ships as a set of composable libraries plus a terminal chat application (`SharpMind.CUI`) built on top of them.

| | |
|---|---|
| **Chat / conversation view** | **Model & session welcome view** |
| ![Chat view](<SharpMind/CUI ChatView.PNG>) | ![Welcome screen](<SharpMind/CUI WelcomeScreen.PNG>) |
| **Runtime options (hardware tier, load mode, sampling)** | |
| ![Options view](<SharpMind/CUI OptionsView.PNG>) | |

---

## Why SharpMind

- **No native runtime required.** Tensor math, quantization kernels, and the training loop are all managed C#. GPU acceleration (via ILGPU) is opt-in and lives in its own assembly — the CPU path never needs it.
- **Inference *and* training in one codebase.** Most C# "LLM" libraries are thin bindings around llama.cpp and only run models. SharpMind can load a GGUF checkpoint, chat with it, LoRA-fine-tune it, distill it into a smaller student, or train a model from scratch — all with the same tensor/quantization primitives.
- **Runs models bigger than your RAM.** A disk-streaming load mode pages transformer layers in and out during the forward pass instead of holding the whole model resident (details below).
- **Modern decoding, not just greedy/top-p.** Both classic speculative decoding and Medusa-style multi-head speculative decoding are implemented from scratch, with careful KV-cache rollback on rejection.
- **A genuinely pluggable kernel system.** Hardware-specific kernel variants (scalar/SSE/AVX2/FMA/GPU) are wired up at runtime through a small internal dispatch layer ("JigSaw") rather than hand-written `if`/`switch` ladders — adding a new backend means adding an assembly, not editing the core.
- **Agent tooling included.** Tool-calling, permission gating (`Never` / `Ask` / `Always`), and sub-agent orchestration ship in `SharpMind.Inference.Agent`.

---

## Inference deep dive

### Decoding strategies

SharpMind ships three interchangeable generators behind a common `IGenerator<T>` / `IGeneratorBuilder` interface, so switching strategy is a builder call, not a rewrite:

| Generator | Idea | Where |
|---|---|---|
| `StandardGenerator` | Classic one-token-per-forward-pass autoregressive decoding. | `SharpMind.Inference/StandardGenerator.cs` |
| `SpeculativeGenerator<T>` | A small draft model proposes several tokens ahead; the target model verifies them in a single batched forward pass. | `SharpMind.Inference/SpeculativeGenerator.cs` |
| `MedusaGenerator<T>` | K extra "draft heads" attached to one hidden state each predict a token at a future offset; all K+1 candidates are verified in one forward pass — no separate draft model needed. | `SharpMind.Inference/MedusaGenerator.cs` |

**Medusa in more detail**, since it's the more novel of the two: each decoding round, the LM head's own greedy pick becomes `token₀`, and K trained head projections from the *same* hidden state produce `token₁ … token_K`. That draft of length K+1 is run through the model as one batch. Verification then walks the draft left to right — `token₀` is always accepted (it's the model's own choice), and each subsequent token is accepted only if the model's forward pass agrees with the head's guess; the walk stops at the first disagreement. If every token in the draft is accepted, a bonus token is generated for free before the next round starts. On partial acceptance, the KV cache is trimmed back to the last accepted position so generation is bit-for-bit identical to plain greedy decoding — Medusa can only change throughput, never correctness. In the ideal case, with K=3 well-calibrated heads, this gives up to a ~2.5× reduction in forward passes per token; today the heads are randomly initialized and need `MedusaHeads.Calibrate` to be run before that speedup materializes.

Speculative decoding follows the more familiar draft-and-verify pattern with an independent draft model, defaulting to 4 draft tokens per round, and shares the same accept/rollback discipline over the KV cache.

### Streaming model loading (`LoadMode.Streaming`)

Normally (`LoadMode.Full`) every transformer layer's weights are allocated and loaded up front. `LoadMode.Streaming` instead:

1. Memory-maps the GGUF file and reads only tensor **metadata** (offsets, shapes, dtypes) at load time — no weight bytes are touched yet.
2. Before each layer runs in the forward pass, `EnsureLayerLoadedSync` blocks (if needed) on that layer's data being read from disk.
3. While the current layer computes, the **next** layer is prefetched on a background task (`PreloadLayerAsync`), overlapping I/O with compute.
4. Once a layer has been consumed, its weights are released (`FreeLayer`) — only the current and next layer's weights are ever resident at once.
5. `CompleteForward` sweeps any remaining resident layers at the end of a pass so memory doesn't creep up across tokens.

The net effect: a model whose full weights don't fit in RAM can still run, trading some throughput for a small, roughly constant memory footprint (current + next layer, rather than all layers). The quantized LM head is also read directly from its raw on-disk bytes in streaming mode rather than materialized as a float tensor, cutting one of the largest single allocations for typical vocab sizes.

### Quantization

Full GGUF-style quant coverage — Q2_K through Q8_0/Q8_1/Q8_K, plus the classic block types (Q4_0/Q4_1/Q5_0/Q5_1) and several 1-bit/ternary formats (IQ1_S, IQ1_M, TQ1_0, TQ2_0) — each with scalar, SSE, AVX2, and FMA kernel variants, and GPU kernels for the most common types.

---

## The "JigSaw" dispatch mechanism

Most inference engines pick a kernel implementation with a big `switch` over CPU features, duplicated at every call site. SharpMind instead defines each swappable operation (a vec-dot, a quantized matmul, an activation, a norm, an optimizer step, …) once as an **abstract method** on a small "ops" class, decorated with a `[PuzzleCornerPiece]` attribute that lists the concrete method for each hardware variant:

```csharp
[PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ4K, true, null,
    "q4k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_FMA)}",
    "q4k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_AVX2)}",
    "q4k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}",
    "q4k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}")]
public abstract unsafe float VecDotQ4K(float* input, byte* rawWeights, int col, int inFeatures);
```

At startup, a `MappingBuilder` inspects the detected `HardwareTier` (or an explicit override) and the active `SharpMindConfig` (activation, attention, gating, quantization scheme, etc.) and produces a `Dictionary<string,string>` mapping each operation key to the variant name it should use — `"q4k_fma"`, `"gpu"`, and so on. `Assembler.CreateInstance<QuantizationOps>(mapping)` then builds a concrete implementation of the abstract class at runtime, resolving every abstract method straight to its chosen static kernel. The result is cached by a hash of the mapping, so a given hardware/config combination only pays the assembly cost once.

The part that makes this genuinely extensible rather than just "reflection instead of a switch": **other assemblies can contribute additional variants for an existing key without the core project referencing them.** `SharpMind.GPU` declares its own `[PuzzlePeice]` entries against the *same* keys (`KeyVecDotQ4_0`, `KeyQuantizedMatMulQ4K`, …) pointing at ILGPU-backed kernels. JigSaw discovers these via assembly scanning at startup, so calling `WithGpu()` — which just causes `SharpMind.GPU` to be loaded into the process — is enough for GPU variants to become selectable, with no compile-time dependency from `SharpMind.Core` on the GPU project at all. Adding a future Metal, Vulkan, or SIMD-width-specific backend is the same pattern: a new assembly, new `[PuzzlePeice]` entries, zero changes to existing call sites.

---

## Training (early-stage, actively evolving)

This half of SharpMind is functional today but earlier in its lifecycle than inference — expect the fastest churn here.

- **Autograd** (`SharpMind.Training/Autograd`) — a from-scratch gradient engine (`ForwardContext`, `BlockContext`, `Gradients`) underpinning the training loop.
- **Optimizers & schedulers** — including AdamW, gradient norm, and LR scheduling, each also dispatched through the JigSaw mapping system so training kernels get the same hardware-tier treatment as inference kernels.
- **LoRA** (`SharpMind.Training/LoRA`) — low-rank adapters over attention and FFN layers for parameter-efficient fine-tuning.
- **Distillation** (`SharpMind.Training/Distillation`) — teacher→student training support.
- **Pruning** (`SharpMind.Training/Pruning`) — structured pruning kernels and a scheduler.
- **`ModelSizer`** — given a data source, samples it, trains a throwaway tokenizer, and grid-searches architecture hyperparameters under a `SizingBudget`/`SizingConstraints` to recommend a model configuration that fits a target parameter budget — a small AutoML step for "how big a model should I even train on this data."
- **Synthetic data** (`SharpMind.Data/Sources/PseudoLanguage`) — a generated toy-language pipeline (morphemes, vocabulary, configurable complexity) for exercising the tokenizer/training pipeline without needing a real corpus.
- **Data sources** — CSV, JSONL, plain text, HuggingFace `datasets-server` streaming (dependency-free, via `HttpClient` + `System.Text.Json`), and a composable cleaning `Pipeline` with branch/merge nodes.

Expect the training API surface (config records, trainer entry points) to change as this matures.

---

## Also included

- **Agent framework** (`SharpMind.Inference.Agent`) — tool-calling with a three-state permission model (`Never` / `Ask` / `Always`), tool categories, and auto-named sub-agents (temperature → a "Greek tier" naming scheme, e.g. `Athena-Alpha` at low temperature, `Prometheus-Epsilon` at high).
- **Chat layer** (`SharpMind.Inference.Chat`) — pluggable prompt formatters (ChatML, a small Jinja-template evaluator, a simple formatter), pinned-message-aware context compaction (summarizing or truncating), and a `ChatArtifact` concept for attaching text/image/code/JSON blocks to a response.
- **`SharpMind.CUI`** — a full terminal chat client: model browser, session manager, settings, file picker, plugin loading, and a permission gate UI, shown above.
- **`SharpMind.GPU`** — ILGPU-backed kernels for activations, norms, and quantized ops, isolated from the core so the CPU path has zero GPU dependency.
- **`SharpMind.Benchmarks`** — evaluation kernels for measuring model/generator performance.

---

## Project layout

```
SharpMind.Core          Zero-dependency tensor primitives, quantization, activations, memory pooling
SharpMind.Model         Architectures, layers, GGUF loading, model config
SharpMind.Inference     Generators (standard/speculative/Medusa), chat, agents, sampling
SharpMind.Training      Autograd, optimizers, LoRA, distillation, pruning
SharpMind.Tokenization  BPE tokenizer, vocab, serialization
SharpMind.Data          Data sources, cleaning pipeline, batching
SharpMind.Data.Parquet  Parquet data source
SharpMind.GPU           ILGPU-backed GPU kernels (optional)
SharpMind.CUI           Terminal chat application
SharpMind.Samples       Example programs
SharpMind.Benchmarks    Evaluation harness
SharpMind.Tests         Test suite
```

---

## Status & roadmap

- [x] GGUF loading (full + streaming)
- [x] CPU quantized inference (Q2–Q8, K-quants, classic blocks, 1-bit/ternary)
- [x] GPU kernels for common quant types
- [x] Speculative decoding
- [x] Medusa-style speculative decoding (heads need manual calibration)
- [x] LoRA, distillation, pruning
- [x] Terminal chat app with agent tooling
- [ ] Documentation and getting-started guides
- [ ] Additional import formats beyond GGUF
- [ ] Medusa head auto-calibration during load
- [ ] Stabilized public API / first tagged release

Issues, questions, and early feedback are welcome — this is a pre-release project and things will move.
