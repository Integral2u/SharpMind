using SharpMind.Inference.Chat;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// Tokens after a closed think block must stream as Responding. The detector
/// used to re-scan the whole reply for "&lt;think&gt;" on every fragment, so the
/// already-closed tag re-armed it and the answer alternated Thinking/Responding
/// token by token — invisible in the CUI with ShowThinking on, fatal for a host
/// that routes the two to different places.
/// </summary>
public sealed class ChatSessionThinkBlockTests
{
    [Fact]
    public async Task AfterTheThinkBlockCloses_EveryFragmentIsResponding()
    {
        await using var session = ScriptedSession.Create("<think>|plan|</think>|Hel|lo| wor|ld");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        string Text(ChatStatus s) => string.Concat(entries.Where(e => e.Status == s).Select(e => e.Token));
        Assert.Equal("plan", Text(ChatStatus.Thinking));
        Assert.Equal("Hello world", Text(ChatStatus.Responding));
    }

    [Fact]
    public async Task ASecondThinkBlock_IsThinkingAgain()
    {
        await using var session = ScriptedSession.Create("<think>|a|</think>|x|<think>|b|</think>|y");

        var entries = new List<ChatStreamEntry>();
        await foreach (var e in session.GetResponseStreamAsync("hi")) entries.Add(e);

        string Text(ChatStatus s) => string.Concat(entries.Where(e => e.Status == s).Select(e => e.Token));
        Assert.Equal("ab", Text(ChatStatus.Thinking));
        Assert.Equal("xy", Text(ChatStatus.Responding));
    }
}
