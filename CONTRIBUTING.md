# Contributing to SharpMind

SharpMind is the C# model, data, inference and training stack. Studio and the CUI sit on top of these libraries.

## Requirements

- Windows, Linux or macOS with the [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- Optional: an NVIDIA GPU and CUDA user-space libraries for `SharpMind.GPU`
- Optional: the `wasm-tools-net10` workload only for `SharpMind.Live`

## Build and test

```powershell
dotnet build SharpMind.slnf -c Debug
dotnet test SharpMind.slnf -c Debug --no-build
```

`SharpMind.slnf` excludes `SharpMind.Live`, which needs the WASM workload. Use the full solution only when you have it installed.

## Encoding guard

```powershell
pwsh ./scripts/check-encoding.ps1
```

CI fails on the UTF-8-decoded-as-Windows-1252 signatures (`U+00E2`, `U+00C3`, `U+00C2`, `U+FFFD`). Keep source comments and log strings as plain ASCII punctuation.

## Style

- C# file-scoped namespaces, nullable enabled.
- Comments explain why, not what; keep public APIs documented with XML comments.
- Tests use xUnit under `SharpMind.Tests`; add a focused test for every behavioural fix.
- Keep allocation and kernel changes measurable; note the benchmark used in the PR.

## Plugins and GPU

`SharpMind.GPU` is loaded as an optional accelerator plugin. It must keep working when the device or CUDA libraries are absent, and the host must fail loudly rather than silently falling back when a GPU run was requested explicitly. See the GPU tests and `README.md` for the current contract.

## Pull requests

- Explain the user-visible effect and the test evidence.
- Do not commit models, datasets, benchmark artifacts, keys or build output.
- Run the build, tests and encoding guard before opening a PR.
- Report vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue.
