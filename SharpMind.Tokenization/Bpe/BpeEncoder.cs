using System.Collections.Generic;
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
    private string[] _specialsSortedByLength;

    // O(1) special-token lookup for Decode — avoids allocating a new array per Specials.All call.
    private HashSet<string> _specialsSet;

    // First-character set for fast SplitOnSpecials scanning — skip positions
    // whose char can't start any special token.
    private HashSet<char> _specialsFirstChars;

    // Per-vocab-id scores (tokenizer.ggml.scores), used only in SentencePiece
    // mode where there's no explicit merges list to rank pairs with.
    private readonly IReadOnlyList<float>? _scores;

    // True for SentencePiece-style vocabularies (original LLaMA/LLaMA-2,
    // Mistral, TinyLlama, etc.) that ship no merges array. These rank
    // candidate merges by token score instead of an explicit rule list.
    public readonly bool UseSentencePieceMerge;

    internal BpeEncoder(
        Vocabulary vocab,
        IReadOnlyList<MergeRule> merges,
        IPreTokeniser preTokeniser,
        IReadOnlyList<float>? tokenScores = null)
    {
        _vocab = vocab;
        _preTokeniser = preTokeniser;
        _mergeIndex = new Dictionary<(string, string), (string, int)>(merges.Count);
        foreach (var rule in merges)
            _mergeIndex[(rule.Left, rule.Right)] = (rule.Merged, rule.Rank);

        _scores = tokenScores;
        // SentencePiece-style vocabularies ship no usable byte-level merges
        // (or none at all) and mark word boundaries with ▁ (U+2581). Byte-level
        // BPE vocabularies (GPT-2, tiktoken) use the GPT-2 byte map instead.
        // TinyLlama's GGUF carries a merge list in SentencePiece format, so we
        // must detect the ▁ marker rather than relying on merges being absent.
        UseSentencePieceMerge = merges.Count == 0 || VocabContainsMetaspace(vocab);
        //Duplicate setting but removes warning
        RebuildSpecialsCache(out _specialsSortedByLength, out _specialsSet, out _specialsFirstChars);
    }

    /// <summary>Refreshes the sorted specials cache after adding new special tokens.</summary>
    internal void RefreshSpecials() => RebuildSpecialsCache(out _, out _, out _);

    /// <summary>
    /// True if the vocab uses the SentencePiece metaspace marker ▁ (U+2581)
    /// to represent word boundaries — i.e. it is SentencePiece-style rather
    /// than GPT-2/tiktoken byte-level BPE.
    /// </summary>
    private static bool VocabContainsMetaspace(Vocabulary vocab)
    {
        if (vocab.Contains("\u2581")) return true;
        foreach (string token in vocab.AllTokens)
            if (token.Contains('\u2581')) return true;
        return false;
    }

    private void RebuildSpecialsCache(out string[] specials, out HashSet<string> specialSet, out HashSet<char> specialsFirstChars)
    {
        specials = [.. _vocab.Specials.All
            .Where(s => !string.IsNullOrEmpty(s))
            .OrderByDescending(s => s.Length)];

        _specialsSortedByLength = specials;
        specialSet = _specialsSet = new HashSet<string>(specials, StringComparer.Ordinal);
        specialsFirstChars = _specialsFirstChars = [.. specials.Select(s => s[0])];
    }

    /// <summary>
    /// Encodes a string to a sequence of token IDs.
    /// Special tokens are matched verbatim and never passed through BPE.
    /// Unknown characters fall back to byte-level tokens.
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false)
    {
        var ids = new List<int>();

        if (addBos) ids.Add(_vocab.BosId);

        // Split on special token boundaries, then encode the plain segments.
        bool isFirstPlainSegment = true;
        foreach (var segment in SplitOnSpecials(text))
        {
            if (segment.IsSpecial)
            {
                ids.Add(_vocab.GetId(segment.Text));
            }
            else if (UseSentencePieceMerge)
            {
                EncodeSentencePieceSegment(segment.Text, isFirstPlainSegment, ids);
                isFirstPlainSegment = false;
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

    /// <summary>
    /// Decodes a sequence of token IDs back to a string.
    /// Each token's GPT-2 byte-encoded characters are reverse-mapped to raw bytes.
    /// Special tokens are included unless <paramref name="skipSpecials"/> is true.
    /// </summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecials = true)
    {
        var bytes = new List<byte>(ids.Length * 2);
        byte[]? reusableBuf = null;

        foreach (int id in ids)
        {
            string token = _vocab.GetToken(id);

            if (skipSpecials && _specialsSet.Contains(token))
                continue;

            // Check for SentencePiece byte token (<0xNN> format) first
            if (Vocabulary.TryDecodeByteToken(token, out byte bt))
            {
                bytes.Add(bt);
                continue;
            }

            // Try GPT-2 style: every character in the token maps to a byte
            bool allGpt2 = true;
            if (reusableBuf is null || reusableBuf.Length < token.Length)
                reusableBuf = new byte[token.Length];
            Span<byte> gpt2Bytes = reusableBuf.AsSpan(0, token.Length);
            for (int i = 0; i < token.Length; i++)
            {
                if (TryDecodeByte(token[i], out byte b))
                    gpt2Bytes[i] = b;
                else
                {
                    allGpt2 = false;
                    break;
                }
            }

            if (allGpt2)
                bytes.AddRange(gpt2Bytes);
            else
                bytes.AddRange(System.Text.Encoding.UTF8.GetBytes(token));
        }

        string result = System.Text.Encoding.UTF8.GetString([.. bytes]);
        // SentencePiece uses ▁ (U+2581) to mark word boundaries — convert to regular space
        result = result.Replace('\u2581', ' ');
        return result;
    }

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
                // Use first-char set to skip positions that can't start a special token.
                int next = pos + 1;
                while (next < text.Length)
                {
                    if (_specialsFirstChars.Contains(text[next]) && TryMatchSpecialAt(text, next))
                        break;
                    next++;
                }
                yield return new Segment(text[pos..next], false);
                pos = next;
            }
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private bool TryMatchSpecialAt(string text, int pos)
    {
        foreach (string special in _specialsSortedByLength)
        {
            if (text.AsSpan(pos).StartsWith(special.AsSpan(), StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>
    /// O(n) BPE merge using a linked-list-like structure with a priority queue to avoid O(N) removals.
    /// </summary>
    private void ApplyMerges(List<string> tokens)
    {
        if (tokens.Count <= 1) return;

        int n = tokens.Count;
        var next = new int[n];
        var prev = new int[n];
        for (int i = 0; i < n; i++)
        {
            next[i] = i + 1;
            prev[i] = i - 1;
        }
        next[n - 1] = -1;

        var queue = new PriorityQueue<int, int>(); // (pairIndex, rank)
        for (int i = 0; i < n - 1; i++)
        {
            if (_mergeIndex.TryGetValue((tokens[i], tokens[next[i]]), out var entry))
                queue.Enqueue(i, entry.Rank);
        }

        while (queue.TryDequeue(out int idx, out int rank))
        {
            if (idx == -1 || next[idx] == -1) continue;
            
            // Re-validate
            if (!_mergeIndex.TryGetValue((tokens[idx], tokens[next[idx]]), out var entry)) continue;
            if (entry.Rank != rank) continue;

            // Merge
            tokens[idx] = entry.Merged;
            int rightIdx = next[idx];
            next[idx] = next[rightIdx];
            if (next[idx] != -1) prev[next[idx]] = idx;
            
            // Re-enqueue
            if (prev[idx] != -1)
                EnqueueSinglePair(queue, tokens, prev[idx], next);
            EnqueueSinglePair(queue, tokens, idx, next);
        }

        // Rebuild tokens list
        var result = new List<string>(n);
        int curr = 0;
        while (curr != -1)
        {
            result.Add(tokens[curr]);
            curr = next[curr];
        }
        tokens.Clear();
        tokens.AddRange(result);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private void EnqueueSinglePair(PriorityQueue<int, int> queue, List<string> tokens, int idx, int[] next)
    {
        int right = next[idx];
        if (right != -1 && _mergeIndex.TryGetValue((tokens[idx], tokens[right]), out var entry))
            queue.Enqueue(idx, entry.Rank);
    }

    private void EncodeSentencePieceSegment(string text, bool isFirstSegment, List<int> ids)
    {
        if (text.Length == 0) return;

        string normalized = text.Replace(' ', '\u2581');
        if (isFirstSegment) normalized = "\u2581" + normalized;

        var symbols = SplitIntoCodepoints(normalized);
        ApplySentencePieceMerges(symbols);

        foreach (var symMemory in symbols)
        {
            string sym = symMemory.ToString();
            if (_vocab.TryGetId(sym, out int id))
            {
                ids.Add(id);
                continue;
            }

            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(sym))
            {
                string bt = $"<0x{b:X2}>";
                ids.Add(_vocab.Contains(bt) ? _vocab.GetId(bt) : _vocab.UnkId);
            }
        }
    }

    private static List<ReadOnlyMemory<char>> SplitIntoCodepoints(string text)
    {
        var result = new List<ReadOnlyMemory<char>>(text.Length);
        int i = 0;
        while (i < text.Length)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                result.Add(text.AsMemory(i, 2));
                i += 2;
            }
            else
            {
                result.Add(text.AsMemory(i, 1));
                i++;
            }
        }
        return result;
    }

    private void ApplySentencePieceMerges(List<ReadOnlyMemory<char>> tokens)
    {
        if (tokens.Count <= 1) return;

        var queue = new PriorityQueue<int, float>();

        void TryEnqueue(int idx)
        {
            if (idx < 0 || idx >= tokens.Count - 1) return;
            string merged = tokens[idx].ToString() + tokens[idx + 1].ToString();
            if (_vocab.TryGetId(merged, out int id))
                queue.Enqueue(idx, -ScoreOf(id));
        }

        for (int i = 0; i < tokens.Count - 1; i++) TryEnqueue(i);

        while (queue.TryDequeue(out int idx, out _))
        {
            if (idx >= tokens.Count - 1) continue;
            string merged = tokens[idx].ToString() + tokens[idx + 1].ToString();
            if (!_vocab.TryGetId(merged, out _)) continue;

            tokens[idx] = merged.AsMemory();
            tokens.RemoveAt(idx + 1);

            if (idx > 0) TryEnqueue(idx - 1);
            if (idx < tokens.Count - 1) TryEnqueue(idx);
        }
    }

    private float ScoreOf(int id) => _scores != null && id >= 0 && id < _scores.Count ? _scores[id] : 0f;

    private List<string> ByteTokenise(string word)
    {
        var result = new List<string>();
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(word))
        {
            string gpt2 = Vocabulary.ByteTokenString(b);
            if (_vocab.Contains(gpt2))
            {
                result.Add(gpt2);
            }
            else
            {
                // SentencePiece stores byte tokens as "<0xNN>" rather than
                // GPT-2 Unicode characters.  Try that format as a fallback.
                string sp = $"<0x{b:X2}>";
                result.Add(_vocab.Contains(sp) ? sp : gpt2);
            }
        }
        return result;
    }

    private static bool TryDecodeByte(char ch, out byte b)
    {
        if (Vocabulary.TryReverseByteMap(ch, out b))
            return true;
        b = 0;
        return false;
    }

    private static bool TryDecodeByte(string token, out byte b) => Vocabulary.TryDecodeByteToken(token, out b);
}
