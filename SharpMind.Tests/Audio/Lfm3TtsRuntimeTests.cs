using SharpMind.Model.Audio;
using Xunit;

namespace SharpMind.Tests.Audio;

public sealed class Lfm3TtsRuntimeTests
{
    [Fact]
    public void BuildPrompt_MatchesTheChatTemplateForAPrefilledAssistantTurn()
    {
        string prompt = Lfm3TtsRuntime.BuildPrompt("Draw me a cat.", "American English", "male");

        Assert.Equal(
            "<|im_start|>system\n" +
            "Perform TTS in American English using a male voice.<|im_end|>\n" +
            "<|im_start|>user\n" +
            "Draw me a cat.<|im_end|>\n" +
            "<|im_start|>assistant\n",
            prompt);
    }

    [Fact]
    public void BuildPrompt_LeavesTheAudioStartTokenToTheRuntime()
    {
        string prompt = Lfm3TtsRuntime.BuildPrompt("Hello.", "American English", "female");

        Assert.DoesNotContain("<|audio_start|>", prompt, StringComparison.Ordinal);
        Assert.EndsWith("<|im_start|>assistant\n", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void BuildPrompt_UsesTheRequestedLanguageAndVoice()
    {
        string prompt = Lfm3TtsRuntime.BuildPrompt("Hola.", "Spanish", "female");

        Assert.Contains("Perform TTS in Spanish using a female voice.", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public void FramesPerSecond_Is12Point5()
    {
        Assert.Equal(12.5, Lfm3TtsRuntime.FramesPerSecond);
        Assert.Equal(1920.0, Lfm3TtsRuntime.SampleRate / Lfm3TtsRuntime.FramesPerSecond);
    }
}
