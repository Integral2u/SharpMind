# llama.cpp tokenizer fixtures

Copied unchanged from llama.cpp tag `b8683`, `models/`:
`ggml-vocab-{qwen2,llama-bpe,gpt-2,phi-3}.gguf` (vocab-only GGUF files) with their `.inp` / `.out`
pairs. llama.cpp's `test-tokenizer-0` runs the same files: `.inp` holds the inputs separated by
`\n__ggml_vocab_test__\n`, `.out` one line of expected token ids per input.

`LlamaCppVocabParityTests` encodes every input through SharpMind's GGUF tokenizer path and requires
the ids llama.cpp produces.

llama.cpp is MIT licensed: Copyright (c) 2023-2024 The ggml authors.
https://github.com/ggml-org/llama.cpp/blob/master/LICENSE
