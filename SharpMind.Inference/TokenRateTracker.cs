using System.Diagnostics;

namespace SharpMind.Inference;

/// <summary>
/// Computes a live rolling average of tokens-per-second.
/// Maintains a ring-buffer of recent token timestamps.
/// </summary>
internal sealed class TokenRateTracker(int windowSize = 10)
{
    private readonly double[] _timestamps = new double[windowSize];
    private int _head;
    private int _count;
    private int _totalTokens;
    private double _startTime = double.NaN;
    private double _firstTokenTime = double.NaN;

    public void Start() { _startTime = Stopwatch.GetTimestamp(); }

    public void RecordToken()
    {
        if (_totalTokens == 0)
            _firstTokenTime = Stopwatch.GetTimestamp();
        _totalTokens++;
        double now = Stopwatch.GetTimestamp();
        _timestamps[_head] = now;
        _head = (_head + 1) % _timestamps.Length;
        if (_count < _timestamps.Length) _count++;
    }

    /// <summary>Rolling average over the last N tokens, or cumulative if fewer than N tokens seen.</summary>
    public float RollingTokensPerSecond
    {
        get
        {
            if (_count < 2) return 0f;
            int oldestIdx = _head >= _count ? 0 : _head;
            double oldest = _timestamps[oldestIdx];
            double latest = _timestamps[(_head - 1 + _timestamps.Length) % _timestamps.Length];
            double elapsedSec = (latest - oldest) / Stopwatch.Frequency;
            if (elapsedSec <= 0) return 0f;
            return (float)((_count - 1) / elapsedSec);
        }
    }

    /// <summary>Seconds from Start() to first RecordToken(), or null if not yet recorded.</summary>
    public float? TimeToFirstToken
    {
        get
        {
            if (double.IsNaN(_firstTokenTime) || double.IsNaN(_startTime)) return null;
            return (float)((_firstTokenTime - _startTime) / Stopwatch.Frequency);
        }
    }

    /// <summary>Cumulative tokens-per-second from the start.</summary>
    public float CumulativeTokensPerSecond
    {
        get
        {
            if (_totalTokens == 0 || double.IsNaN(_startTime)) return 0f;
            double elapsedSec = (Stopwatch.GetTimestamp() - _startTime) / Stopwatch.Frequency;
            if (elapsedSec <= 0) return 0f;
            return (float)(_totalTokens / elapsedSec);
        }
    }
}
