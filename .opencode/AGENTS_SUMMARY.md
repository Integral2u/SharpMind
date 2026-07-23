# SharpMind Optimization Session

## Goal
Apply three performance optimizations to SharpMind's inference kernels: column-parallel decode, thread-local MoE scratch buffers, and FMA VecDot rewrites.

## Constraints & Preferences
- Process in order; test each before moving to next.
- No shared bump allocator in MoE (races across threads); use thread-local arenas.
- FMA fixes priority: Q6_K first, then Q2_K/Q3_K/Q4_K, then Q4_0/Q4_1/Q4_NL.

## Progress
### Done
- **Opt 1 – Column-parallel decode:** Added `DecodeParallel(VecDotFn, ...)` helper. Applied to 54 `*_Parallel_*` methods across 15 files (CPU + GPU).
- **Opt 2 – Thread-local MoE scratch buffers:** `ThreadLocal<Workspace>` in `FfnKernels.cs`, arena reset per token. Eliminates 5 heap allocations per expert per token.
- **Opt 3 – VecDotQ4_1_AVX2 accumulator:** Verified already correct — no change needed.
- **Opt 4 – FMA VecDot rewrites (all 13 functions):**
  - **Q6K_FMA:** Full SIMD rewrite — 4 independent accumulator chains (vacc0–vacc3) for ILP across interleaved quarters. Hoisted outside b/nOff loops; single `HSum256_Avx` at function end.
  - **Q2K_FMA:** Replaced per-chunk HSum with hoisted vacc0/vacc1 accumulators, single final HSum.
  - **Q3K_FMA:** Single vacc hoisted outside b loop, single final HSum.
  - **Q4K_FMA:** Single vacc hoisted outside b/n16 loops, single final HSum.
  - **Q4_0_FMA, Q4_1_FMA, Q4_NL_FMA:** New `_FMA` variants created from AVX2 counterparts with `Fma.MultiplyAdd` (fused multiply-add). Accumulators hoisted outside b loop, single final HSum.
  - **Q5_0_FMA, Q5_1_FMA, Q8_0_FMA, Q8_1_FMA, Q5K_FMA, Q8K_FMA:** Accumulators hoisted outside b loop, moved HSum from per-block to single final call.
  - All registered in `QuantizationOps.cs`.
  - Build: 0 errors. Tests: 64 passed, 9 pre-existing Q4_0 failures unchanged.

### In Progress
- None.

### Blocked
- None.

## Key Decisions
- **Four independent accumulator chains in Q6K_FMA** (vacc0–vacc3, one per interleaved quarter) provide software pipelining — the CPU can overlap FMAs across its FMA ports instead of serializing on one dependency chain.
- **Accumulators hoisted to function scope** across all blocks and sub-blocks: `HSum256_Avx` called exactly once per dot product call instead of once per 128-element sub-block (or once per 32-element block), minimizing reduce overhead.
- `DecodeParallel` uses a `VecDotFn` delegate to avoid 55 separate helpers. Delegate overhead is limited to one call per chunk (~8–16 calls).
- MoE workspace sizing: 1 MB per thread, reset per token. Sufficient for hidden_dim up to 8192 with top-2 experts.

## Next Steps
- No remaining FMA optimizations — all 17 `VecDot*_FMA` functions are fully optimized with hoisted accumulators and single-reduce HSum.

## Critical Context
- NuGet implicit usings provide `System.Threading.Tasks.Parallel` everywhere.
- **Pre-existing test failures:** 9 `VecDotQ4_0_*` tests (1x AgreesAcrossTiers, 4x MultiBlock64, 4x FullBlock32) fail on both original and modified code — test expected computation uses wrong nibble layout (`qs[i/2]` interleaved instead of `qs[i]` half-block). Not caused by changes.
- `VecDotFn` delegate defined in `QuantizationOps.cs` as `unsafe delegate float VecDotFn(float* input, byte* rawWeights, int col, int inFeatures)`.

## Relevant Files
- `SharpMind.Core/Quantization/QuantizationKernels.cs`: `DecodeParallel` helper.
- `SharpMind.Core/Quantization/QuantizationKernels.Q6.cs`: VecDotQ6K_FMA — 4-chain ILP rewrite with hoisted accumulators.
- `SharpMind.Core/Quantization/QuantizationKernels.Q2.cs`: VecDotQ2K_FMA — hoisted vacc0/vacc1.
- `SharpMind.Core/Quantization/QuantizationKernels.Q3.cs`: VecDotQ3K_FMA — hoisted vacc.
- `SharpMind.Core/Quantization/QuantizationKernels.Q4.cs`: VecDotQ4K_FMA, VecDotQ4_0_FMA, VecDotQ4_1_FMA, VecDotQ4_NL_FMA — new and fixed functions.
- `SharpMind.Core/Quantization/QuantizationKernels.Q5.cs`: VecDotQ5_0_FMA, VecDotQ5_1_FMA, VecDotQ5K_FMA — hoisted accumulators.
- `SharpMind.Core/Quantization/QuantizationKernels.Q8.cs`: VecDotQ8_0_FMA, VecDotQ8_1_FMA, VecDotQ8K_FMA — hoisted accumulators.
- `SharpMind.Core/Quantization/QuantizationOps.cs`: Updated registry entries for all new `_FMA` functions.
- `SharpMind.Model/Layers/Ffn/FfnKernels.cs`: Thread-local MoE workspaces.
- `SharpMind.Core/Memory/Workspace.cs`: Bump allocator.
