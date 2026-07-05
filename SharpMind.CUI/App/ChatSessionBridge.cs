using System.Collections.Concurrent;
using SharpMind.Inference.Chat;

namespace SharpMind.CUI.App;

/// <summary>
/// The shape <see cref="App"/> actually needs from "something driving the
/// chat screen" — submit a line of input, drain whatever stream entries
/// came back, dispose cleanly. <see cref="ChatSessionBridge"/> implements
/// this around a real <c>IChatSession</c>; <see cref="DebugChatBridge"/>
/// implements the same shape around a scripted sequence of canned
/// responses, with no model, tokenizer, or transformer involved at all.
/// <see cref="ChatScreen"/> and <see cref="App"/> don't need to know or care
/// which one they're holding.
/// </summary>
public interface IChatBridge : IAsyncDisposable
{
    void SubmitUserInput(string text);
    IEnumerable<ChatStreamEntry> DrainEntries();
    void Interrupt();
    bool Faulted { get; }
    Exception? Fault { get; }
}

/// <summary>
/// <see cref="IChatSession.StartChatAsync"/> owns its own turn loop: it calls
/// the prompt callback synchronously, in-line, once per turn, and blocks
/// until that callback returns a string. That's the right shape for a
/// console readline loop, but a single-threaded render loop can't block
/// inside a UI callback without freezing the screen between turns.
///
/// This bridge runs the whole session loop on a background task. The prompt
/// callback blocks on a <see cref="SemaphoreSlim"/> that the render thread
/// releases once the user presses Enter; stream entries get pushed into a
/// concurrent queue the render thread drains every frame. No async UI
/// callback gymnastics, no rewriting ChatSession's loop shape — just a
/// thread boundary in the one place it's actually needed.
/// </summary>
public sealed class ChatSessionBridge(IChatSession session, bool disposeUnderlyingSession = true) : IChatBridge
{
    private readonly ConcurrentQueue<ChatStreamEntry> _incoming = new();
    private readonly SemaphoreSlim _inputReady = new(0);
    private string? _pendingInput;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loopTask;

    public bool Faulted { get; private set; }
    public Exception? Fault { get; private set; }

    /// <summary>
    /// Whether DisposeAsync should actually dispose the underlying
    /// ChatSession (and, through it, the shared Transformer) or just unwind
    /// this bridge's own loop and leave the session object alone. Mutable
    /// rather than fixed at construction, specifically so the caller can
    /// decide this right before closing, once ModelCache.Release has
    /// answered "was this the last session using that model?" — that answer
    /// usually isn't known yet when the bridge is first created.
    /// </summary>
    public bool DisposeUnderlyingSession { get; set; } = disposeUnderlyingSession;

    public void Start()
    {
        _loopTask = Task.Run(async () =>
        {
            try
            {
                await session.StartChatAsync(
                    prompt: () =>
                    {
                        _inputReady.Wait(_cts.Token);
                        return ChatMessage.User(_pendingInput ?? "");
                    },
                    response: entry => _incoming.Enqueue(entry),
                    token: _cts.Token);
            }
            catch (OperationCanceledException)
            {
                // expected on shutdown
            }
            catch (Exception ex)
            {
                Faulted = true;
                Fault = ex;
            }
        });
    }

    /// <summary>Called from the render thread when the user submits a line.</summary>
    public void SubmitUserInput(string text)
    {
        _pendingInput = text;
        _inputReady.Release();
    }

    /// <summary>Called once per frame from the render thread to drain anything the session emitted.</summary>
    public IEnumerable<ChatStreamEntry> DrainEntries()
    {
        while (_incoming.TryDequeue(out var entry))
            yield return entry;
    }

    public void Interrupt() => session.Interrupt();

    public async ValueTask DisposeAsync()
    {
        await _cts.CancelAsync();
        _inputReady.Release(); // unblock prompt() if it's waiting
        if (_loopTask is not null)
        {
            try { await _loopTask; } catch { /* already surfaced via Fault if relevant */ }
        }

        // ChatSession.DisposeAsync() unconditionally disposes the Transformer
        // it was built on. When a model is shared across multiple named chat
        // sessions (see ModelCache), only the session that closes last may
        // actually call this — every earlier one must skip it entirely, or
        // closing one tab would destroy the model out from under siblings
        // still using it. The caller (MainWindow) decides this via ref
        // counting and passes the answer in at construction time.
        if (DisposeUnderlyingSession)
            await session.DisposeAsync();

        _cts.Dispose();
        _inputReady.Dispose();
    }
}
