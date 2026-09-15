using SharpMind.Core;
using SharpMind.Model.Config;

namespace SharpMind.CUI.App;

/// <summary>
/// Tracks loaded models by file path (+ the hardware tier baked into them,
/// since that affects the actual Transformer instance, not just
/// session-level behaviour) so that opening a second named chat session
/// against the same GGUF file reuses the already-loaded weights instead of
/// reading them from disk again.
///
/// Streaming mode sessions are never cached — each agent requires its own
/// <see cref="TransformerWeightsStreaming"/> instance with isolated layer
/// load/unload state.
///
/// The one constraint that shapes this whole class:
/// <c>ChatSession</c> does NOT dispose the Transformer it was built on
/// (ChatSession is created with disposeModel=false), so a session may safely
/// end while siblings still use the shared model. The model is disposed when
/// the last session using it closes: <see cref="Release"/> answers "was this
/// the last user?" via ref counting, and the caller (MainWindow.CloseSession)
/// disposes the Transformer from its chat-session state once that answer is
/// yes.
/// </summary>
public sealed class ModelCache
{
    private readonly Dictionary<string, LoadedModel> _byKey = new(StringComparer.OrdinalIgnoreCase);

    private static string KeyFor(SessionOptions options) =>
        $"{options.ModelPath}|{options.HardwareTier}";

    /// <summary>Returns an already-loaded model for these exact settings if one exists,
    /// incrementing its ref count. Returns null for streaming mode (never cached).</summary>
    public LoadedModel? TryAcquire(SessionOptions options)
    {
        if (options.LoadMode == LoadMode.Streaming) return null;
        if (options.ModelPath is null) return null;
        if (!_byKey.TryGetValue(KeyFor(options), out var loaded)) return null;
        loaded.RefCount++;
        return loaded;
    }

    /// <summary>Called once, right after a fresh LoadModelAsync, to make the result available for future reuse.
    /// Streaming mode sessions are not cached — each agent requires its own isolated instance.</summary>
    public void Register(SessionOptions options, LoadedModel loaded)
    {
        if (options.LoadMode == LoadMode.Streaming) return;
        loaded.RefCount = 1;
        _byKey[KeyFor(options)] = loaded;
    }

    /// <summary>
    /// Called when a chat session built on this model is closing. Decrements
    /// the ref count; if this was the last session using it, removes it from
    /// the cache and returns true so the caller knows it's now safe (and
    /// necessary) to dispose the model's Transformer. If other sessions are
    /// still using it, returns false: the caller disposes only the session's
    /// own runtime and leaves the shared Transformer alive.
    ///
    /// Streaming mode sessions are never cached — returns true immediately
    /// so the caller always disposes its own session and Transformer.
    /// </summary>
    public bool Release(SessionOptions options)
    {
        if (options.LoadMode == LoadMode.Streaming) return true;
        if (options.ModelPath is null) return false;
        string key = KeyFor(options);
        if (!_byKey.TryGetValue(key, out var loaded)) return false;

        loaded.RefCount--;
        if (loaded.RefCount <= 0)
        {
            _byKey.Remove(key);
            return true;
        }
        return false;
    }
}
