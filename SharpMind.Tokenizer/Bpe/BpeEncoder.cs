using SharpMind.Tokenizer.PreTokeniser;
using SharpMind.Tokenizer.Vocab;

namespace SharpMind.Tokenizer.Bpe;

/// <summary>
/// Encodes strings to token ID sequences by applying BPE merge rules.
///
/// Encoding pipeline:
///   1. Pre-tokenise the string into words.
///   2. Represent each word as a sequence of byte-level tokens.
///   3. Apply merge rules greedily in priority (rank) order.
///   4. Map the resulting tokens to their IDs in the vocabulary.
/// </summary>
public sealed class BpeEncoder
{
    private readonly Vocabulary _vocab;
    private readonly IPreTokeniser _preTokeniser;

    // Merge lookup: (left, right) → (merged token, rank)
    // Rank is used to apply the highest-priority merge at each step
    private readonly Dictionary<(string, string), (string Merged, int Rank)> _mergeIndex;

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
    }

    // ── Encode ────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes a string to a sequence of token IDs.
    /// Unknown characters fall back to byte-level tokens; with byte-level vocab
    /// the result is always lossless (no [UNK] in the output).
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();

        if (addBos) ids.Add(_vocab.BosId);

        foreach (string word in _preTokeniser.PreTokenise(text))
        {
            var tokens = ByteTokenise(word);
            ApplyMerges(tokens);
            foreach (string token in tokens)
                ids.Add(_vocab.GetId(token));
        }

        if (addEos) ids.Add(_vocab.EosId);
        return [.. ids];
    }

    /// <summary>
    /// Encodes a batch of strings. Each string is encoded independently.
    /// Returns a jagged array — sequences have different lengths.
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
                // Multi-character merged token — encode as UTF-8
                foreach (byte tb in System.Text.Encoding.UTF8.GetBytes(token))
                    bytes.Add(tb);
            }
        }

        return System.Text.Encoding.UTF8.GetString([.. bytes]);
    }

    // ── BPE merge application ─────────────────────────────────────────────

    /// <summary>
    /// Applies BPE merge rules to a mutable list of tokens in-place.
    /// Uses the priority-queue approach: always finds and applies the
    /// lowest-rank (highest-priority) merge available.
    /// </summary>
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

            if (bestIdx < 0) break; // no more merges applicable

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