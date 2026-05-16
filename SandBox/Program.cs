using SandBox;

// Verify SharpMind prompt formatting produces correct output
LlamaTest.VerifyPromptFormatting();

// Run LLamaSharp reference inference with the same prompt format
await LlamaTest.RunLlamaReference();
