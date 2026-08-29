# GpuTrainBench

Wall-clock cost of one **GPU LoRA training step**, so a kernel change can be judged against a number
rather than a feeling.

It drives the real production path — `ModelFactory.CreateForTraining` → `CreateTrainingTransformer`
→ `LoRAModel` → `TrainLoop` with `engine: GpuBackpropEngine` — and reports the median ms/step and
tok/s per batch size.

```
dotnet run -c Release --project tools/GpuTrainBench -- --help
dotnet run -c Release --project tools/GpuTrainBench -- --batch 1,2,4 --seq 256
```

| flag | default | |
|---|---|---|
| `--batch` | `1,2,4` | batch sizes to sweep |
| `--seq` | `256` | sequence length |
| `--warmup` | `2` | untimed steps before measuring |
| `--steps` | `5` | timed steps per batch size |
| `--layers` | `30` | transformer layers |
| `--flash` | off | flash attention path instead of the materialised one |
| `--prof` | off | also print a per-phase breakdown (`SM_PROF=1` implies it) |

No model file is needed: the harness builds a randomly initialised model at the SmolLM2-135M shape
and feeds it random in-range token ids. It is a **throughput harness, not a quality eval** — only
the shapes, and the kernels and GEMMs they drive, matter for timing. The loader is deliberately
inert (no file IO, no tokenizer, no shuffling) so it can never be what gets measured, and it is
sized to the step count so a long run cannot quietly go dry mid-measurement.

## Reading the output honestly

**Check the first line before trusting any number.** It is `GpuDevice.Description`, which names the
accelerator and the GEMM path actually taken — `cuBLAS <version>` or the `tiled16` ILGPU fallback.
A machine with no CUDA device silently resolves to OpenCL, or to ILGPU's CPU accelerator, and still
produces a plausible-looking table.

That matters for correctness as well as speed: ILGPU's CUDA backend has **no intrinsic for
`Math.Exp` / `Log` / `Sqrt` / `Tanh` inside a kernel** (`XMath` is required), while OpenCL accepts
them. A fully green run on a non-CUDA accelerator therefore proves less than it appears to.

**Throughput and the breakdown come from separate passes.** `GpuStepProfiler` synchronises the
device at every phase boundary — kernel launches are asynchronous, and without a barrier a phase's
cost lands on whichever later phase happens to block — so a profiled step is meaningfully slower
than a real one. With `--prof` the tool times with the profiler off, then profiles without timing,
over the same warmed engine. Quoting tok/s off a profiled pass understates it by 10-15%.

## Reference point

GTX 1060 (Pascal, sm_61) with cuBLAS 12.8, 135M params, LoRA r=8, seq 256, materialised attention:

| batch | ms/step | tok/s |
|---|---|---|
| 1 | 283.9 | 768 |
| 2 | 443.9 | 1153 |
| 4 | 828.4 | 1236 |

## On a machine with no CUDA Toolkit

`GpuDevice` needs only the driver (`nvcuda.dll`, which ships with any NVIDIA driver) plus cuBLAS. If
`nvcc` / `CUDA_PATH` are absent, the standalone redistributable is enough — no installer, no IDE
integration: download a `libcublas` archive matching the driver's CUDA version (top right of
`nvidia-smi`) and put its `bin\` on `PATH` for the run.

Use a **12.x** build rather than 13.x if the card is Pascal: CUDA 13 dropped sm_61 support, and the
resolver in `SharpMind.GPU/Native/Cuda.cs` probes `cublas64_13.dll` first.
