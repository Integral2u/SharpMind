<p align="center">
  <img src="SharpMind.Core/sharpmind_logo.svg" alt="SharpMind logo" width="512" height="512"/>
</p>

<p align="center"><b>SharpMind. A pure C# / .NET LLM engine — inference and agent tooling in one solution.</b></p>

<p>
  <img alt="status" src="https://img.shields.io/badge/status-1.0.0.0-brightgreen">
  <img alt="lang" src="https://img.shields.io/badge/language-C%23%20(.NET%2010)-239120">
  <img alt="deps" src="https://img.shields.io/badge/dependencies-near--zero-blue">
</p>

[![NuGet Downloads](https://img.shields.io/nuget/dt/SharpMind.Core.svg)](https://www.nuget.org/packages/SharpMind.Core)
![GitHub Sponsor](https://img.shields.io/github/sponsors/Integral2u?label=Sponsor&logo=GitHub)
[![Last commit](https://img.shields.io/github/last-commit/Integral2u/SharpMind)](https://github.com/Integral2u/SharpMind/commits/main)

---

## What is SharpMind?

SharpMind is an end-to-end LLM stack written entirely in C#, with no dependency on llama.cpp, PyTorch, or any native runtime for its core path. It loads GGUF models, runs quantized CPU/GPU inference with modern decoding acceleration (speculative + Medusa-style drafting), and — unusually for a C# inference engine — also includes its own autograd engine so you can fine-tune (LoRA), distill, and prune models in the same process that serves them.

It ships as a set of composable libraries plus a terminal chat application (`SharpMind.CUI`) built on top of them.

| | |
|---|---|
| **Chat / conversation view** | **Model & session welcome view** |
| ![Chat view](<SharpMind.Core/CUI ChatView.PNG>) | ![Welcome screen](<SharpMind.Core/CUI WelcomeScreen.PNG>) |
| **Runtime options (hardware tier, load mode, sampling)** | |
| ![Options view](<SharpMind.Core/CUI OptionsView.PNG>) | |

---

## Why SharpMind

- **No native runtime required.** Tensor math, quantization kernels, and the training loop are all managed C#. GPU acceleration (via ILGPU) is opt-in and lives in its own assembly — the CPU path never needs it.
- **Runs models bigger than your RAM.** A disk-streaming load mode pages transformer layers in and out during the forward pass instead of holding the whole model resident (details below).
- **Modern decoding, not just greedy/top-p.** Both classic speculative decoding and Medusa-style multi-head speculative decoding are implemented from scratch, with careful KV-cache rollback on rejection.
- **A genuinely pluggable kernel system.** Hardware-specific kernel variants (scalar/SSE/AVX2/FMA/GPU) are wired up at runtime through a small internal dispatch layer [JigSawDotNet](https://github.com/Integral2u/JigSawDotNet) rather than hand-written `if`/`switch` ladders — adding a new backend means adding an assembly, not editing the core.
- **Agent tooling included.** Tool-calling, permission gating (`Never` / `Ask` / `Always`), and sub-agent orchestration ship in `SharpMind.Inference.Agent`.
- **Inference *and* training in one codebase.** Most C# "LLM" libraries are thin bindings around llama.cpp and only run models. SharpMind can load a GGUF checkpoint, chat with it, or train a model from scratch. LoRA fine-tuning and distillation are also present in the training stack but are still experimental — see [Training](#training-experimental).

---

## Install the chat app (Windows)

The quickest way to try the terminal chat client is the installer, which sets up `SharpMind.CUI`, Start Menu + desktop shortcuts, and the app folder:

**[Download SharpMind Console Setup (MSI)](https://github.com/Integral2u/SharpMind/releases/download/v1.0.0.0/SharpMind.Console.Setup.msi)**

Requires the [.NET 10 Runtime](https://dotnet.microsoft.com/download/dotnet/10.0). Running the MSI installs the app and shortcuts; uninstall or repair is available via Apps & Features. Then grab a model as described in [Quick Start](#quick-start-load-and-run-a-model) — the running app also has a built-in model browser.

---
## Quick Start: Load and Run a Model

This walks through the smallest possible program that loads a GGUF model and starts an interactive chat session.

### 1. Get a model

Download a small instruct model to get started — [Qwen3-0.6B-Q8_0](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) is a good first choice: it's under 1GB, loads quickly, and is known to produce coherent output out of the box.

Place the `.gguf` file in a folder, e.g. `C:\Models\Qwen3-0.6B-Q8_0.gguf`.

### 2. Minimal Program.cs

```csharp
using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

var modelPath = @"C:\Models\Qwen3-0.6B-Q8_0.gguf";

// 1. Load model metadata, config, and tokenizer from the GGUF file.
var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);

if (tokenizer == null)
{
    Console.WriteLine("No tokenizer data found in this GGUF file.");
    return;
}

// 2. Resolve hardware/quant mapping and load the weights.
var sharpConfig = modelConfig.ForModel();
var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);

using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, LoadMode.Full);
weights.InitializeWeights();

// 3. Build the transformer.
using var model = ModelFactory.CreateTransformer(weights, sharpConfig);

// 4. Start a chat session.
await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
{
    MaxTokens = 512,
    Temperature = 0.7f,
    TopK = 40,
    TopP = 0.9f,
};
session.InitializeChat();

Console.WriteLine("Chat ready! Type a message (or 'exit' to quit).\n");
var cts = new CancellationTokenSource();

await session.StartChatAsync(Prompt, Response, cts.Token);

async Task<ChatMessage> Prompt()
{
    Console.Write("\nYou: ");
    var input = Console.ReadLine() ?? "exit";
    if (input == "exit") cts.Cancel();
    return new ChatMessage { Content = input, Role = ChatRole.User };
}

void Response(ChatStreamEntry entry) => Console.Write(entry.Token);
```

### What each step does

| Step | Purpose |
|---|---|
| `metaHelper.Load` | Reads the GGUF file's architecture, hyperparameters, and tokenizer vocab. Pass `null` for the second argument unless the model ships an external tokenizer file. |
| `modelConfig.ForModel()` | Resolves the config into a hardware-aware mapping (CPU/GPU, quant ops). |
| `ModelFactory.CreateWeights(..., LoadMode.Full)` | Loads and dequantizes all weights into memory up front. Use `LoadMode.Streaming` instead on memory-constrained machines — it loads one layer at a time during inference rather than holding everything resident. |
| `ModelFactory.CreateTransformer` | Wires the loaded weights into the actual forward-pass graph. |
| `ChatSession<...>` | Manages conversation history, prompt formatting (auto-detected from the model's chat template), and the generation loop. |
| `StartChatAsync(Prompt, Response, token)` | Runs the chat loop — `Prompt` supplies the next user message, `Response` receives streamed output tokens as they're generated. |

### Notes

- `Temperature = 0.7f`, `TopK = 40`, `TopP = 0.9f` give more natural, varied output than greedy decoding (`Temperature = 0`). Set `Temperature = 0` for deterministic, reproducible output when debugging.
- `MaxTokens` caps the response length per turn — lower it (e.g. `128`) on slower hardware if you don't need long responses.
- On memory-constrained machines, prefer `LoadMode.Streaming` over `LoadMode.Full`, and stick to Q4–Q8 quantized models under ~1B–3B params for reasonable throughput.

---

## Compatibility matrix

Validated across two independent runs against 15 model/architecture/quantization combinations. "Runs clean" means the model loaded, built a transformer, and completed all five benchmark prompts with no exceptions or crashes — it is **not** a quality claim; output coherence varies a lot by model size and quant level, which is expected and not specific to SharpMind.

| Architecture | Model | Quant | Runs clean (both runs) |
|---|---|---|---|
| Gemma (function-calling variant) | functiongemma-270m-it | Q8_0 | ✅ |
| Gemma 3 | gemma-3-270m-it | Q8_0 | ✅ |
| Gemma 3 | gemma-3-270m-it | Q4_K_M | ✅ |
| SmolLM2 | SmolLM2-135M-Instruct | Q4_K_M | ✅ |
| SmolLM | SmolLM-135M | Q4_K_M | ✅ |
| Qwen2 | Qwen2-0.5B | Q2_K | ✅ |
| Qwen2 (instruct) | qwen2-0.5b-instruct | Q4_K_M | ✅ |
| Qwen2 (instruct) | qwen2-0.5b-instruct | Q8_0 | ✅ |
| Qwen2.5 (instruct) | qwen2.5-1.5b-instruct | Q8_0 | ✅ |
| Qwen3 | Qwen3-0.6B | Q8_0 | ✅ |
| DeepSeek-R1-Distill-Qwen | DeepSeek-R1-Distill-Qwen-1.5B | Q3_K_M | ✅ |
| DeepSeek-R1-Distill-Qwen | DeepSeek-R1-Distill-Qwen-1.5B | Q8_0 | ✅ |
| TinyLlama | tinyllama-1.1b-chat-v1.0 | Q8_0 | ✅ |
| Llama 3 (small) | llama3-small | Q2_K | ✅ |
| Llama 3 (small) | llama3-small | Q3_K_M | ✅ |

_Tested on: **AMD Ryzen 3 2200U (2C/4T, 2.5GHz base) w/ Radeon Vega Mobile Graphics, 12GB RAM** — a modest mobile/laptop-class chip, not a workstation. Load times and throughput scale heavily with hardware and quant level — as a rough sense of range on this machine, `SmolLM-135M.Q4_K_M` loaded in ~2s, while `qwen2.5-1.5b-instruct-q8_0` (the largest model tested) took ~2 minutes to load and initialize. That everything above ran clean on a 2-core/4-thread laptop CPU is itself a reasonable data point for SharpMind's baseline hardware requirements — run your own copy of the benchmark against your target hardware before relying on these numbers for capacity planning._

---

## Extensibility: plugins

Beyond JigSaw's compile-time kernel dispatch, `SharpMind.CUI` has a separate, simpler **runtime plugin loader** for extending the app itself without touching the core libraries. Drop a `.dll` into the app's `Plugins/` folder and `PluginLoader.LoadFrom` will scan it and wire up anything it recognizes:

- **Tools** — any class with a method tagged `[ToolDesc("...")]` (parameters can carry their own `[ToolDesc]` too, for per-argument descriptions) is picked up as an agent tool automatically, the same mechanism the built-in `EchoTool`, `FileSystemTool`, and `WeatherTool` use:

  ```csharp
  public class WeatherTool
  {
      [ToolDesc("Gets the current weather for a specified city.")]
      public async Task<string> GetCurrentWeather([ToolDesc("The name of the city.")] string city) { ... }
  }
  ```

- **Context compactors** — classes implementing `IContextCompactor` register under their own `Name` and become selectable alongside the built-in summarizing/truncating compactors.
- **Prompt pre/post-processors** — `IPromptPreProcessor` / `IPromptPostProcessor` implementations get to rewrite a prompt before it's sent or a completion after it comes back.
- **Generators** — any type implementing `IGeneratorBuilder<TCache>` is discovered by matching the open generic interface via reflection and added to the generator-strategy list next to Standard/Speculative/Medusa — a third-party decoding strategy can appear in the same options menu as the built-in ones.

The loader is defensive by design: each `.dll` loads independently (a failure is recorded as a warning, not a crash), duplicate names are rejected rather than silently overwritten, and only concrete classes with a public parameterless constructor are considered. Today this loader is wired into `SharpMind.CUI` specifically; nothing about the interfaces is CUI-specific, so the same plugin assemblies work if you host the inference/chat libraries directly.

---

### Streaming model loading (`LoadMode.Streaming`)

Normally (`LoadMode.Full`) every transformer layer's weights are allocated and loaded up front. `LoadMode.Streaming` instead:

1. Memory-maps the GGUF file and reads only tensor **metadata** (offsets, shapes, dtypes) at load time — no weight bytes are touched yet.
2. Before each layer runs in the forward pass, `EnsureLayerLoadedSync` blocks (if needed) on that layer's data being read from disk.
3. While the current layer computes, the **next** layer is prefetched on a background task (`PreloadLayerAsync`), overlapping I/O with compute.
4. Once a layer has been consumed, its weights are released (`FreeLayer`) — only the current and next layer's weights are ever resident at once.
5. `CompleteForward` sweeps any remaining resident layers at the end of a pass so memory doesn't creep up across tokens.

The net effect: a model whose full weights don't fit in RAM can still run, trading some throughput for a small, roughly constant memory footprint (current + next layer, rather than all layers). The quantized LM head is also read directly from its raw on-disk bytes in streaming mode rather than materialized as a float tensor, cutting one of the largest single allocations for typical vocab sizes.

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

---

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

## Training

This half of SharpMind is functional today but earlier in its lifecycle than inference — expect the fastest churn here.

- **Autograd** (`SharpMind.Training/Autograd`) — a from-scratch gradient engine (`ForwardContext`, `BlockContext`, `Gradients`) underpinning the training loop.
- **Optimizers & schedulers** — including AdamW, gradient norm, and LR scheduling, each also dispatched through the JigSaw mapping system so training kernels get the same hardware-tier treatment as inference kernels.
- **LoRA** (`SharpMind.Training/LoRA`) — low-rank adapters over attention and FFN layers for parameter-efficient fine-tuning.
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
SharpMind.Training      Autograd, optimizers, LoRA
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

See [CHANGELOG.md](CHANGELOG.md) for release history.

# Wishlist (not ordered)
- [ ] AVX512 Kernels
- [ ] Additional Model Support
- [ ] Optimiations
- [ ] Microsoft IChatClient and or other services.

Issues, questions, and early feedback are welcome.
