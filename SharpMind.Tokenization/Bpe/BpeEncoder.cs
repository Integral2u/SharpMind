using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Bpe;

/// <summary>
/// Encodes strings to token ID sequences by applying BPE merge rules.
///
/// Encoding pipeline:
///   1. Split input on special token boundaries (preserving them as atomic units).
///   2. For each non-special segment, pre-tokenise into words.
///   3. Represent each word as a sequence of byte-level tokens.
///   4. Apply merge rules greedily in priority (rank) order.
///   5. Map the resulting tokens to their IDs in the vocabulary.
///
/// Special tokens (e.g. &lt;|im_start|&gt;, &lt;|im_end|&gt;) are matched verbatim
/// and emitted as their single vocab ID without going through BPE. This is
/// critical: the Qwen/LLaMA pre-tokeniser regex would otherwise shred them
/// into individual characters, producing wrong IDs and near-zero logits.
/// </summary>
public sealed class BpeEncoder
{
    private readonly Vocabulary _vocab;
    private readonly IPreTokeniser _preTokeniser;

    // Merge lookup: (left, right) → (merged token, rank)
    private readonly Dictionary<(string, string), (string Merged, int Rank)> _mergeIndex;

    // Special tokens sorted longest-first so that longer patterns (e.g.
    // "<|im_start|>") match before shorter prefixes (e.g. "<|im").
    private readonly string[] _specialsSortedByLength;

    internal BpeEncoder(
        Vocabulary vocab,
        IReadOnlyList<MergeRule> merges,
        IPreTokeniser preTokeniser)
    {
        _vocab = vocab;
        _preTokeniser = preTokeniser;
        _mergeIndex = new Dictionary<(string, string), (string, int)>(merges.Count);
        foreach (var rule in merges)
            _mergeIndex[(rule.Left, rule.Right)] = (rule.Merged, rule.Rank);

        // Build sorted special-token list for fast scanning.
        _specialsSortedByLength = [.. vocab.Specials.All
            .Where(s => !string.IsNullOrEmpty(s))
            .OrderByDescending(s => s.Length)];
    }

    // ── Encode ────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes a string to a sequence of token IDs.
    /// Special tokens are matched verbatim and never passed through BPE.
    /// Unknown characters fall back to byte-level tokens.
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();

        if (addBos) ids.Add(_vocab.BosId);

        // Split on special token boundaries, then BPE-encode the plain segments.
        foreach (var segment in SplitOnSpecials(text))
        {
            if (segment.IsSpecial)
            {
                ids.Add(_vocab.GetId(segment.Text));
            }
            else
            {
                foreach (string word in _preTokeniser.PreTokenise(segment.Text))
                {
                    var tokens = ByteTokenise(word);
                    ApplyMerges(tokens);
                    foreach (string token in tokens)
                        ids.Add(_vocab.GetId(token));
                }
            }
        }

        if (addEos) ids.Add(_vocab.EosId);
        return [.. ids];
    }

    /// <summary>
    /// Encodes a batch of strings. Each string is encoded independently.
    /// </summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addBos = false, bool addEos = false)
        => [.. texts.Select(t => Encode(t, addBos, addEos))];

    // ── Decode ────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes a sequence of token IDs back to a string.
    /// Byte tokens are reconstructed from UTF-8 bytes.
    /// Special tokens are included unless <paramref name="skipSpecials"/> is true.
    /// </summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecials = true)
    {
        var bytes = new List<byte>(ids.Length * 2);

        foreach (int id in ids)
        {
            string token = _vocab.GetToken(id);

            if (skipSpecials && _vocab.Specials.All.Contains(token))
                continue;

            if (TryDecodeByte(token, out byte b))
            {
                bytes.Add(b);
            }
            else
            {
                foreach (byte tb in System.Text.Encoding.UTF8.GetBytes(token))
                    bytes.Add(tb);
            }
        }

        return System.Text.Encoding.UTF8.GetString([.. bytes]);
    }

    // ── Special-token splitter ────────────────────────────────────────────

    private readonly record struct Segment(string Text, bool IsSpecial);

    /// <summary>
    /// Splits <paramref name="text"/> into alternating plain / special segments.
    /// Special tokens are matched longest-first so overlapping prefixes don't
    /// cause false positives.
    /// </summary>
    private IEnumerable<Segment> SplitOnSpecials(string text)
    {
        if (_specialsSortedByLength.Length == 0)
        {
            if (text.Length > 0) yield return new Segment(text, false);
            yield break;
        }

        int pos = 0;
        while (pos < text.Length)
        {
            // Try to match any special token at the current position.
            string? matched = null;
            foreach (string special in _specialsSortedByLength)
            {
                if (text.AsSpan(pos).StartsWith(special.AsSpan(), StringComparison.Ordinal))
                {
                    matched = special;
                    break;
                }
            }

            if (matched != null)
            {
                yield return new Segment(matched, true);
                pos += matched.Length;
            }
            else
            {
                // Advance to the next potential special-token start (or end of string).
                int next = pos + 1;
                while (next < text.Length)
                {
                    bool startsSpecial = false;
                    foreach (string special in _specialsSortedByLength)
                    {
                        if (text.AsSpan(next).StartsWith(special.AsSpan(), StringComparison.Ordinal))
                        {
                            startsSpecial = true;
                            break;
                        }
                    }
                    if (startsSpecial) break;
                    next++;
                }
                yield return new Segment(text[pos..next], false);
                pos = next;
            }
        }
    }

    // ── BPE merge application ─────────────────────────────────────────────

    private void ApplyMerges(List<string> tokens)
    {
        while (tokens.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIdx = -1;
            string bestMerge = string.Empty;

            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_mergeIndex.TryGetValue((tokens[i], tokens[i + 1]), out var entry)
                    && entry.Rank < bestRank)
                {
                    bestRank = entry.Rank;
                    bestIdx = i;
                    bestMerge = entry.Merged;
                }
            }

            if (bestIdx < 0) break;

            tokens[bestIdx] = bestMerge;
            tokens.RemoveAt(bestIdx + 1);
        }
    }

    private static List<string> ByteTokenise(string word)
    {
        var result = new List<string>();
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(word))
            result.Add(Vocabulary.ByteTokenString(b));
        return result;
    }

    private static bool TryDecodeByte(string token, out byte b)
    {
        if (token.Length == 1 && char.IsAscii(token[0]) && !char.IsControl(token[0]))
        {
            b = (byte)token[0];
            return true;
        }
        if (token.StartsWith("<0x", StringComparison.Ordinal) && token.EndsWith('>') &&
            token.Length == 6 &&
            byte.TryParse(token[3..5], System.Globalization.NumberStyles.HexNumber, null, out b))
            return true;

        b = 0;
        return false;
    }
}