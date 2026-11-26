namespace SharpMind.Model.Audio;

/// <summary>
/// Common surface of the native speech runtimes so a host can hold either LFM3 or
/// ZONOS3 behind one field and keep the same WAV artifact contract.
/// </summary>
public interface ITtsRuntime : IDisposable
{
    /// <summary>Output sample rate of <see cref="Synthesize"/>.</summary>
    int SampleRate { get; }

    /// <summary>Voice name or phrase used by the runtime's prompt.</summary>
    string Voice { get; set; }

    /// <summary>Upper bound on generated speech frames.</summary>
    int MaxFrames { get; set; }

    /// <summary>Sampling temperature.</summary>
    float Temperature { get; set; }

    /// <summary>Optional per-frame sampling hook.</summary>
    Action<int, int[]>? FrameSampled { get; set; }

    /// <summary>Optional end-of-speech hook.</summary>
    Action<int, int[]>? EndOfSpeech { get; set; }

    /// <summary>Synthesizes <paramref name="text"/> into mono samples.</summary>
    float[] Synthesize(string text, CancellationToken cancellationToken = default);
}
