# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue:

- Preferred: [GitHub private security advisory](https://github.com/Integral2u/SharpMind/security/advisories/new)
- Alternative: open an issue titled only "Security contact requested" with no details — a
  maintainer will reach out for a private channel.

Please include a description of the issue, steps to reproduce (a minimal GGUF/config/request
that triggers it is ideal), and the affected version/commit. This is a small project without
a dedicated security team; expect an initial response within a few days, not a formal SLA.

## Supported versions

Only the latest release on `master` is supported. There is no long-term-support branch —
fixes land as new releases, not backports.

## Security-relevant design, by area

SharpMind runs entirely locally by default and has no telemetry or network calls of its own.
The things worth being deliberate about are the places it's designed to consume untrusted
input or extend itself with external code:

### Model files (GGUF)

Model loading parses an untrusted binary format, including `unsafe` code paths for
performance (quantization kernels, tensor deserialization). A crafted GGUF file is the
realistic attack surface here — treat model files the same way you'd treat any other
untrusted binary input, and don't load models from sources you don't trust, the same way you
wouldn't run an unknown executable. Parser crashes (DoS) on malformed files are a legitimate
report; if you find one that goes further than a crash (memory corruption reachable from file
content), that's a high-priority report.

### Accelerator plugins

`AcceleratorLoader` loads and executes every `.dll` in a configured plugins folder — this is
arbitrary code execution by design, not a bug. Only point the plugins folder at accelerator
builds you built or obtained from a source you trust. This is equivalent to running any other
native/managed code you downloaded; SharpMind does no sandboxing or signature verification of
plugin assemblies.

### SharpMind.Server

The OpenAI-compatible HTTP server has **no built-in authentication**. It defaults to binding
`localhost` only, which is the safe default — if you set `--host 0.0.0.0` (or otherwise expose
it beyond loopback), put it behind your own authentication and/or a reverse proxy. Treat an
exposed instance as you would any unauthenticated internal service: don't expose it directly
to the internet.

### Agentic tool calls

Chat sessions can dispatch tool calls that touch the filesystem and network on the host
machine. `SharpMind.Server.CLI`'s `--no-files` and `--no-network` flags disable those
categories outright; use them when serving a model to anyone (or anything) you don't fully
trust, since a model's tool-call output is effectively untrusted input once it can trigger
real file/network access.

## Scope

Out of scope: vulnerabilities that require the attacker to already control the plugins folder,
the model files being loaded, or the machine SharpMind runs on — those are trust boundaries
by design, not bugs (see above). In scope: anything reachable from a GGUF file's *content*,
from `SharpMind.Server`'s HTTP surface, or from ordinary (non-tool-privileged) chat input.
