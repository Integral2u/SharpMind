using System.Text;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Incremental equivalent of <c>ChatSession.SanitizeToolMarkup</c> for the
/// streaming loop. It holds only a short marker-sized tail, so a long response
/// costs O(n) instead of re-sanitising the whole accumulated string per token.
/// </summary>
internal sealed class StreamingToolMarkupFilter
{
    private const string OpenTool = "<tool_call";
    private const string CloseTool = "</tool_call>";
    private const string OpenFunction = "<function_call>";
    private const string CloseFunction = "</function_call>";
    private const string MistralMarker = "[TOOL_CALLS]";
    private const string PythonTag = "<|python_tag|>";
    private const string PythonTagBare = "<python_tag>";
    private const string GemmaMarker = "functionCall";
    private const string ThinkOpen = "<think>";
    private const string ThinkClose = "</think>";

    private static readonly string[] Markers =
    [
        OpenTool, CloseTool, OpenFunction, CloseFunction,
        MistralMarker, PythonTag, PythonTagBare, GemmaMarker,
        ThinkOpen, ThinkClose
    ];

    private readonly StringBuilder _pending = new();
    private bool _inToolCall;
    private bool _inFunctionCall;
    private int _visibleLength;

    public void Reset()
    {
        _pending.Clear();
        _inToolCall = false;
        _inFunctionCall = false;
        _visibleLength = 0;
    }

    /// <summary>Feeds a fragment and returns the new user-visible text.</summary>
    public string Append(string fragment)
    {
        if (fragment.Length > 0)
            _pending.Append(fragment);

        var output = new StringBuilder();
        while (_pending.Length > 0)
        {
            if (_inToolCall)
            {
                if (!ConsumeBlock(CloseTool, ref _inToolCall)) break;
                continue;
            }
            if (_inFunctionCall)
            {
                if (!ConsumeBlock(CloseFunction, ref _inFunctionCall)) break;
                continue;
            }

            string text = _pending.ToString();
            int best = -1;
            foreach (string marker in Markers)
            {
                int idx = text.IndexOf(marker, StringComparison.Ordinal);
                if (idx >= 0 && (best < 0 || idx < best))
                    best = idx;
            }

            if (best < 0)
            {
                EmitRemainder(output, text);
                break;
            }

            if (best > 0)
            {
                Emit(output, text[..best]);
                _pending.Remove(0, best);
                continue;
            }

            // A fragment can end in the middle of a marker. Keep that tail and
            // wait for the next fragment instead of leaking it as prose.
            if (IsMarkerPrefix(text))
                break;

            if (text.StartsWith(OpenTool, StringComparison.Ordinal))
            {
                _inToolCall = true;
                _pending.Remove(0, OpenTool.Length);
            }
            else if (text.StartsWith(OpenFunction, StringComparison.Ordinal))
            {
                _inFunctionCall = true;
                _pending.Remove(0, OpenFunction.Length);
            }
            else if (text.StartsWith(MistralMarker, StringComparison.Ordinal))
            {
                _pending.Remove(0, MistralMarker.Length);
            }
            else if (text.StartsWith(PythonTag, StringComparison.Ordinal))
            {
                _pending.Remove(0, PythonTag.Length);
            }
            else if (text.StartsWith(PythonTagBare, StringComparison.Ordinal))
            {
                _pending.Remove(0, PythonTagBare.Length);
            }
            else if (text.StartsWith(GemmaMarker, StringComparison.Ordinal))
            {
                // The old whole-string sanitizer removed this only at the very
                // start of the response (index 0 or 1).
                if (_visibleLength <= 1)
                    _pending.Remove(0, GemmaMarker.Length);
                else
                {
                    Emit(output, GemmaMarker);
                    _pending.Remove(0, GemmaMarker.Length);
                }
            }
            else if (text.StartsWith(ThinkOpen, StringComparison.Ordinal))
            {
                _pending.Remove(0, ThinkOpen.Length);
            }
            else if (text.StartsWith(ThinkClose, StringComparison.Ordinal))
            {
                _pending.Remove(0, ThinkClose.Length);
            }
            else if (text.StartsWith(CloseTool, StringComparison.Ordinal))
            {
                // A stray close tag outside a block is prose, not machinery.
                Emit(output, CloseTool);
                _pending.Remove(0, CloseTool.Length);
            }
            else if (text.StartsWith(CloseFunction, StringComparison.Ordinal))
            {
                Emit(output, CloseFunction);
                _pending.Remove(0, CloseFunction.Length);
            }
            else
            {
                // Unreachable: every marker in Markers has a branch above.
                _pending.Remove(0, 1);
            }
        }

        return output.ToString();
    }

    /// <summary>Emits any trailing text that never completed a marker or block.</summary>
    public string Flush()
    {
        if (_inToolCall || _inFunctionCall)
        {
            _pending.Clear();
            return "";
        }

        string text = _pending.ToString();
        _pending.Clear();
        _visibleLength += text.Length;
        return text;
    }

    private bool ConsumeBlock(string closeMarker, ref bool inBlock)
    {
        string text = _pending.ToString();
        int close = text.IndexOf(closeMarker, StringComparison.Ordinal);
        if (close >= 0)
        {
            _pending.Remove(0, close + closeMarker.Length);
            inBlock = false;
            return true;
        }

        int keep = LongestSuffixPrefix(text, [closeMarker]);
        _pending.Remove(0, text.Length - keep);
        return false;
    }

    private void EmitRemainder(StringBuilder output, string text)
    {
        int keep = LongestSuffixPrefix(text, Markers);
        int emit = text.Length - keep;
        if (emit > 0)
            Emit(output, text[..emit]);
        if (emit > 0 || keep == 0)
            _pending.Remove(0, emit);
    }

    private void Emit(StringBuilder output, string text)
    {
        output.Append(text);
        _visibleLength += text.Length;
    }

    private static bool IsMarkerPrefix(string text)
    {
        foreach (string marker in Markers)
            if (text.Length < marker.Length && marker.StartsWith(text, StringComparison.Ordinal))
                return true;
        return false;
    }

    private static int LongestSuffixPrefix(string text, string[] markers)
    {
        int keep = 0;
        foreach (string marker in markers)
        {
            int max = Math.Min(text.Length, marker.Length - 1);
            for (int k = 1; k <= max; k++)
                if (text.AsSpan(text.Length - k).SequenceEqual(marker.AsSpan(0, k)))
                    keep = Math.Max(keep, k);
        }
        return keep;
    }
}
