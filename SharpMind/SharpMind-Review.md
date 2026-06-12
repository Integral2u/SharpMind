# SharpMind — Codebase Review
## Performance, Correctness & Experimental-Friendliness

---

## What's Already Good

Before getting into gaps, it's worth acknowledging what's already well-constructed:

- **JigSawDotNet dispatch pattern** — the factory-time SIMD selection (AVX2 / FMA / scalar) is exactly right. No per-call `Avx2.IsSupported` branch, dispatch cost is zero at runtime. This is the hardest pattern to get right and it's clean.
- **NativeBuffer + Workspace** — 32-byte aligned allocations, bump allocator for the inference hot loop, reference-counted shared views. This is proper systems-level thinking for C#.
- **Online softmax** (Milakov & Gimelshein) in the attention kernel — fusing the score accumulation with the softmax statistics is a meaningful bandwidth win over naive 3-pass attention.
- **PagedKVCache** — flat `[batch, pages, heads, pageSize, headDim]` layout with direct pointer arithmetic. The right data structure.
- **Speculative decoding** exists — this is a non-trivial inference optimisation that most C# LLM projects don't have at all.
- **Full quantisation stack** — Q2K through Q8K / Q4_0 through Q8_1, with VecDot kernels per dtype. Good coverage.
- **BPE trainer + multi-format tokenizer serialisation** (GPT-2, LLaMA, Mistral, Qwen, GGUF) — the tokenization layer is production-quality.
- **Data pipeline** — composable cleaning stages, packing batcher, parquet/CSV sources. Solid foundation.
- **LoRA, distillation, pruning** already exist as pluggable modules — the extension surface is there.

---

## Part 1: Performance — What Would Move the Needle

These are ranked roughly by impact-per-effort.

---

### 1. Tiled / Blocked MatMul (Highest Impact)

This is the single biggest missing piece. The current `MatMulInner` kernel loops naively over `M×K×N`:

```csharp
// Current: row of A × col of BT — cache-hostile for large K
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j += 8)   // AVX2 unroll
        ...
```

For anything beyond toy sizes the L1/L2 cache is thrashed. The fix is blocking: partition the output tile into chunks that fit in cache registers, then accumulate inside the tile. A simple 4×1 or 6×1 micro-kernel with register accumulation doubles throughput for typical hidden dims (512–4096):

```csharp
// Sketch: 6-row × 8-col (6 × 1 AVX2 register tile)
const int MR = 6, NR = 8;
for (int ii = 0; ii < M; ii += MR)
    for (int jj = 0; jj < N; jj += NR)
    {
        // Load 6 accumulators
        var c0 = Vector256<float>.Zero; ... var c5 = Vector256<float>.Zero;
        for (int k = 0; k < K; k++)
        {
            var b = Vector256.LoadUnsafe(ref btPtr[k * N + jj]);
            c0 = Fma.MultiplyAdd(Vector256.Create(aPtr[ii*K+k]),     b, c0);
            c1 = Fma.MultiplyAdd(Vector256.Create(aPtr[(ii+1)*K+k]), b, c1);
            // ...
        }
        // Store 6 rows
    }
```

For `K = N = 2048`, this is the difference between ~10 GFLOPS and ~60 GFLOPS on a modern desktop CPU. This is what llama.cpp does in its `ggml_vec_dot_f32` and `ggml_gemm_f32` kernels.

**Effort:** Medium. Implement one AVX2 kernel, plug into the existing JigSaw slot. No API changes needed.

---

### 2. Fused Attention Kernel — RoPE Inside ScaledDotProduct

The codebase's own `Planning.txt` already identifies this:

> Fused RoPE → Attention Kernel — Apply RoPE rotations on-the-fly inside the ScaledDotProduct kernel to reduce memory bandwidth.

Currently, RoPE is applied to Q and K tensors as separate passes before `ScaledDotProduct` is called. Each is a full read-modify-write over `[heads, seqLen, headDim]`. Fusing them into the attention kernel eliminates two bandwidth passes:

```csharp
// Inside ScaledDotProductAVX2, when computing qi · kj:
// Apply RoPE to qi and kj on-the-fly using cos/sin tables
// No separate temporary Q_rotated or K_rotated tensor needed
```

For a 2048-token context with 32 heads × 128 headDim, that's saving ~32 MB of memory traffic per layer, per token. At 32 layers that compounds hard.

**Effort:** Medium-hard. The RoPE tables are already in `RoPE.cs`. The fusion is 40–60 lines of kernel change.

---

### 3. AdamW Optimizer — SIMD the Update Loop

The optimizer inner loop is purely scalar today:

```csharp
for (int j = 0; j < data.Length; j++)
{
    float g = grad[j];
    m[j] = _beta1 * m[j] + (1f - _beta1) * g;
    v[j] = _beta2 * v[j] + (1f - _beta2) * g * g;
    float mHat = m[j] / bc1;
    float vHat = v[j] / bc2;
    float update = _lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
    if (decay) update += _lr * _weightDecay * data[j];
    data[j] -= update;
}
```

This is run once per optimizer step over every parameter, which for even a 120M model is ~120M multiply-accumulate ops. The entire loop is trivially vectorisable:

```csharp
// All five operations (m update, v update, bias-correct, sqrt, weight decay)
// can be expressed as Vector256<float> ops with Fma.MultiplyAdd and
// Avx.Sqrt or the FastSqrt approximation.
```

This is a 4–8× improvement on the optimizer step, which matters for training where it's called every `GradAccumSteps` batches. Also consider parallelising across parameters with `Parallel.For` — each parameter's arrays are independent.

**Effort:** Low-medium. No architecture change.

---

### 4. Parallel Gradient Backward Passes

The gradient backward in `Gradients.cs` (`Linear`, `Attention`, `RMSNorm` etc.) runs entirely single-threaded. The forward pass parallelises heads with `Parallel.For(0, totalHeads, DoHead)` — the backward should mirror this. Specifically:

- `Gradients.Linear` — the `dInput = dOutput @ W` loop over `B` rows is independent; each row can be dispatched to a thread.
- `Gradients.CrossEntropySoftmax` — each token `t` is independent.
- Gradient accumulation into `dW` requires a reduction, but can be done with thread-local accumulators then a final add, same as PyTorch's `GEMM` backward.

**Effort:** Medium. The independence property is already there; the main work is adding thread-local weight-gradient buffers to avoid race conditions.

---

### 5. Fused QKV Projection

Again from `Planning.txt`:

> Fused QKV Projection — Combine Wq, Wk, and Wv into a single fused MatMul for better cache locality.

Currently three separate `LinearLayer.Forward` calls each do their own matmul. Fusing them into `[input @ (Wq | Wk | Wv)]` — a single wide matmul — gives the GEMM kernel a larger N dimension, which is more efficient (better SIMD utilisation, single matrix load of input). This is the standard approach in every production transformer. The API change is small: `AttentionLayer` holds a single `fused_qkv` `LinearLayer` of `OutFeatures = 3 * headDim * numHeads`, then slices.

**Effort:** Low-medium. One API addition, no kernel changes.

---

### 6. Attention V-Accumulation — AVX2 Vectorise Pass 2

In `AttentionKernels.cs`, Pass 2 (the V-weighted output accumulation) is scalar:

```csharp
for (int j = 0; j < kvLen; j++)
{
    float sm = MathF.Exp(scoreRow[j] - max) * invSum;
    float* vj = v + (long)j * headDim;
    for (int d = 0; d < headDim; d++)
        outI[d] += sm * vj[d];   // ← scalar
}
```

The inner `headDim` loop (typically 64–128) is a SAXPY (`y += α * x`) and should be vectorised with FMA:

```csharp
var vSm = Vector256.Create(sm);
for (int d = 0; d <= headDim - 8; d += 8)
    Vector256.StoreUnsafe(
        Fma.MultiplyAdd(Vector256.LoadUnsafe(ref vj[d]), vSm,
                        Vector256.LoadUnsafe(ref outI[d])),
        ref outI[d]);
```

Since this runs `kvLen × headDim` times per query token, and `kvLen` grows during generation, this has increasing impact as context length grows.

**Effort:** Low. A drop-in kernel change.

---

### 7. Parallel.For Granularity — The QuantizedForward Problem

In `LinearLayer.QuantizedForward`:

```csharp
Parallel.For(0, outF, col =>
{
    pOutL[col] = VecDotQxK(pInL, pRawL, col, inF);
});
```

Spawning `outF` (e.g. 4096) tasks for a single row is catastrophically fine-grained — the thread pool overhead exceeds the work per task. This should chunk to at least 64 columns per task, or parallelise over rows instead (which are independent and typically larger work units):

```csharp
Parallel.For(0, m, row =>
{
    for (int col = 0; col < outF; col++)
        result[row * outF + col] = VecDotQxK(input + row * inF, pRaw, col, inF);
});
```

During inference with batch size 1 (the common case), `m = 1` so the outer `Parallel.For` degenerates to serial — which is correct and avoids the overhead entirely.

**Effort:** Very low. A 5-line change with real correctness implications.

---

### 8. `Clamp`, `Sqrt`, `Abs` — Add SIMD

These in `TensorOps.cs` are scalar loops today. `Clamp` in particular runs during gradient clipping and activation bounds. A Vector256 version is 4 lines:

```csharp
var vMin = Vector256.Create(min); var vMax = Vector256.Create(max);
for (; i <= dst.Length - 8; i += 8)
    Avx.Min(vMax, Avx.Max(vMin, Vector256.LoadUnsafe(ref src[i])))
       .StoreUnsafe(ref dst[i]);
```

`Sqrt` can use `Avx.Sqrt` or the `rsqrt` + Newton-Raphson fast path. Mention these in the JigSaw kernel pattern for consistency.

---

## Part 2: Missing Training Infrastructure

---

### 9. Mixed-Precision Training (BF16 / FP16 Storage)

The training path is fully float32. Storing activations and gradients in BF16 halves the memory bandwidth consumed by optimizer state and gradient communication. The core pattern is:

- **Master weights** stay float32 (in `Parameter.Data`)
- **Forward activations** are cast to BF16 on store, FP32 on load
- **Optimizer state** (m, v in AdamW) stays float32

C# `System.Half` exists but BF16 doesn't have native support — you'd represent it as `ushort` with bit manipulation, similar to how the existing `HalfToFloat_F16C` in `QuantizationKernels` works. This would be a meaningful memory-bandwidth improvement for training, though it's the largest effort item here.

**Effort:** High. Worth a `Tensor<BFloat16>` type backed by `ushort[]` + conversion in hot paths.

---

### 10. Gradient Checkpointing

The current backward computes dLogits → `BackwardEmbedding` but there is no mechanism to trade compute for memory during training by recomputing activations. For any model above ~125M parameters training on CPU, activation memory dominates. A simple `CheckpointedBlock` wrapper that discards activations after forward and replays them during backward would let you train much larger models within a given memory budget.

**Effort:** Medium. The `BlockContext` / `ForwardContext` in `Autograd/` are already the right abstraction; they need a `Recompute` flag.

---

### 11. Gradient Accumulation Doesn't Zero Mid-Accumulation

In `TrainLoop.RunAsync`:

```csharp
if (accumCount < _config.GradAccumSteps) continue;
// ...
_optimizer.ZeroGrad();
```

Gradients are zeroed *after* the step, but there's no explicit zero *before* the first accumulation batch (only a zero at the end of the previous step). If the loop resumes from a checkpoint with `ResumeFrom`, the first accumulation window starts with stale gradients. `ZeroGrad()` should be called at the top of each accumulation window, not only at its end.

**Effort:** Trivial. One line moved.

---

### 12. Learning Rate Scheduler Isn't Wired Up

`Planning.txt` flags this directly:

> `SharpMind.Training.Trainer._scheduler never used.`

The `Trainer` class holds a `_scheduler` field that is constructed but never called to update `_optimizer.LearningRate`. `TrainLoop` does call it, but the legacy `Trainer` path doesn't. Any user starting from `Trainer` rather than `TrainLoop` gets a flat learning rate regardless of scheduler choice. Either wire it or remove the dead field.

**Effort:** Trivial.

---

## Part 3: Missing Experimental Infrastructure

These directly serve the "easy to plug in new ideas" goal.

---

### 13. No Hook/Callback System for Activations

Currently there is no way to inspect, modify, or collect intermediate activations during a forward pass without editing the model code. For experimentation this is a major gap — you can't do representation analysis, activation patching, probing classifiers, or intervention experiments.

A lightweight approach: each `TransformerBlock` accepts an optional `IActivationHook`:

```csharp
public interface IActivationHook
{
    void OnPreAttention(int layer, Tensor<float> hidden);
    void OnPostAttention(int layer, Tensor<float> hidden);
    void OnPostFFN(int layer, Tensor<float> hidden);
}
```

Null-checked at call site, zero overhead when not set. This unlocks a huge class of experiments without touching core code.

---

### 14. No Model Surgery / Weight Merging API

`Planning.txt` lists the entire taxonomy of weight merging techniques (SLERP, TIES, DARE, Task Arithmetic) and the "LLM lobotomy" concept. None are implemented. These would be the most interesting experimental features in the repo — a `WeightMerger` static class with the five methods would be a realistic addition that's genuinely useful and doesn't require any new kernel work. The math is straightforward float arithmetic over `Parameter.Data`.

---

### 15. Multi-Token Prediction (MTP)

`Planning.txt` notes:

> MPT multi token prediction

MTP predicts multiple future tokens in parallel from the same hidden state (DeepSeek-V3 style). It's both a training-efficiency technique and an inference speedup when combined with speculative decoding. The structure would be a small `MTPHead` module (`Linear` + `RMSNorm`) applied to the final hidden state, predicting tokens at positions `t+1, t+2, ..., t+k`. No changes to the main transformer block required.

---

### 16. No Benchmark / Profiling Harness

`Planning.txt` mentions a `Benchmarks` project but it doesn't exist yet. For a project whose primary claim is performance, the absence of a systematic benchmark is a real gap. You can't improve what you don't measure, and external contributors won't know whether their kernel changes actually helped.

A minimal benchmark project should cover:
- MatMul throughput at sizes (256, 512, 1024, 2048, 4096) with BenchmarkDotNet
- Attention kernel throughput vs context length
- Tokens/sec prefill and decode for a reference model config
- Optimizer step time vs parameter count

This would also catch regressions from the `Parallel.For` granularity issue above.

---

### 17. No `ILayer` / Module Registry for Composing Novel Architectures

All architecture variants (`DecoderArch`, `TransformerBlock`, attention types, FFN types) are concrete classes. There's no lightweight `ILayer` abstraction that would let a researcher compose a custom architecture from a config string or YAML without writing C# class definitions.

A minimal module registry pattern:

```csharp
// Register a custom FFN variant
LayerRegistry.Register("MoE-FFN", cfg => new MoEFeedForward(cfg));

// Compose from config
var block = LayerRegistry.Build(config.FfnKind, config);
```

The existing `FfnKind`, `AttentionKind`, `NormKind` enums are already doing this conceptually — the gap is that adding a new variant currently requires adding an enum case, a class, and a switch branch, rather than just registering a factory.

---

## Part 4: Quick Wins (Low Effort, Visible Impact)

| Issue | File | Fix |
|---|---|---|
| `TransposeInternal` uses `Parallel.For` but for the small matrices in inference (headDim × headDim) the overhead is higher than the work | `TensorOps.cs` | Add a size threshold: serial below 64×64, parallel above |
| `Workspace.CalculateRequiredSize` caps prefill at 256 tokens for workspace sizing | `Workspace.cs` | Expose this cap as a parameter; long prefill falls back to direct alloc already but the caller can't easily configure the threshold |
| `ArgTopK` copies the entire data array for introselect | `TensorOps.cs` | Use `Span<float>` directly rather than `a.Data.ToArray()` — the copy is unnecessary |
| Backward activation functions (GELU, SiLU) are scalar loops | `Gradients.cs` | Same AVX2 pattern as the forward kernels; both run during training on every token |
| `LayerNormLayer.ApplyRow` still has an inline `Avx.IsSupported` check | `Layers/LayerNormLayer.cs` | Contradicts the JigSaw dispatch pattern; should be a JigSaw `PuzzleCornerPiece` like the matmul and activation kernels |

---

## Part 5: Architectural Suggestions for Experimentation

These don't affect performance directly but make the codebase genuinely better as an experimental platform:

**Separate `IForwardPass` from `ITrainable`** — currently `Transformer` owns both inference-time logic and training-time parameter management. Splitting them makes it trivial to wrap a model in a distillation harness or a LoRA adapter without touching `Transformer` itself.

**Make `Workspace` scope-aware** — a simple stack of offsets would allow nested allocations (useful when an experimental module wants its own scratch space without interfering with the outer workspace). The bump allocator is already there; add `Push()` / `Pop()` scope markers.

**Typed experiment config** — a base `ExperimentConfig : SharpMindConfig` class that carries metadata (name, description, hypothesis) alongside model hyperparameters. Trivial to add, makes checkpoint logs and experiment tracking far easier.

**`IScheduler` chaining** — the existing schedulers (cosine, warmup, linear) are good but not composable. A `ComposedScheduler` that chains stages (warmup → cosine → constant tail) would cover most real training runs without writing a custom scheduler.

---

## Summary Priority Order

| Priority | Item | Impact | Effort |
|---|---|---|---|
| 1 | Tiled/blocked MatMul kernel | Very high | Medium |
| 2 | Fused RoPE + attention kernel | High | Medium |
| 3 | SIMD AdamW + parallel grad step | High | Low-Medium |
| 4 | Vectorise attention V-accumulation | Medium-High | Low |
| 5 | Fix `Parallel.For` granularity in quantized forward | Medium | Very Low |
| 6 | Fused QKV projection | Medium | Low-Medium |
| 7 | Activation hook system | Medium (experimentation) | Low |
| 8 | Benchmark / profiling project | Medium | Low-Medium |
| 9 | Weight merging API | High (experimentation) | Medium |
| 10 | Gradient checkpointing | High (large model training) | Medium |
| 11 | Fix ZeroGrad ordering | Correctness | Trivial |
| 12 | Fix unused scheduler in Trainer | Correctness | Trivial |
| 13 | SIMD backward activation kernels | Low-Medium | Low |
| 14 | LayerNorm JigSaw conversion | Consistency | Low |

The first four items together would make SharpMind's inference loop genuinely competitive — not with a finely-tuned C implementation, but in the "not ignorable" territory you're aiming for. The blocked MatMul alone is probably a 3–5× improvement on the most compute-intensive operation in the system.
