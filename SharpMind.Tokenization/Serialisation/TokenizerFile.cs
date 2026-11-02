using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Tokenization.Vocab;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Bpe;

namespace SharpMind.Tokenization.Serialisation;

/// <summary>
/// Saves and loads SharpMind native tokenizer JSON files.
///
/// Format:
/// <code>
/// {
///   "version": "1.0",
///   "pre_tokeniser": "gpt2" | "cl100k" | "whitespace",
///   "cl100k_variant": "cl100k" | "llama3" | "qwen2" | "qwen35",   ← only with "cl100k"
///   "gguf_pre": "...", "gguf_model": "...",                       ← source GGUF names, when known
///   "special_tokens": { "unk": "...", "bos": "...", "eos": "...", "pad": "...",
///                       "additional": [...] },
///   "vocab": { "token": id, ... },
///   "merges": [ "left right", ... ]   ← ordered by rank (0 = highest priority)
/// }
/// </code>
///
/// To load third-party tokenizers see:
///   <see cref="Gpt2Converter"/>
///   <see cref="LlamaConverter"/>
///   <see cref="MistralConverter"/>
/// </summary>
public static class TokenizerFile
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    /// <summary>Saves a trained <see cref="BpeModel"/> to a SharpMind JSON file.</summary>
    public static void Save(BpeModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        File.WriteAllText(path, ToJson(model));
    }

    /// <summary>Serialises a trained <see cref="BpeModel"/> to its SharpMind JSON string.</summary>
    public static string ToJson(BpeModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        var obj = new JsonObject
        {
            ["version"] = "1.0",
            ["kind"] = model.IsCharMode ? "char" : "bpe",
            ["pre_tokeniser"] = PreTokeniserName(model.PreTokeniser),
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = model.Vocab.Specials.Unk,
                ["bos"] = model.Vocab.Specials.Bos,
                ["eos"] = model.Vocab.Specials.Eos,
                ["pad"] = model.Vocab.Specials.Pad,
                ["additional"] = new JsonArray(
                    [.. model.Vocab.Specials.Additional.Select(t => JsonValue.Create(t)!)]),
            },
            ["vocab"] = BuildVocabObject(model.Vocab),
            ["merges"] = new JsonArray(
                [.. model.Merges
                     .OrderBy(m => m.Rank)
                     .Select(m => JsonValue.Create($"{m.Left} {m.Right}")!)]),
        };
        // The variant goes in its own key so "pre_tokeniser" stays "cl100k", which older readers know.
        if (model.PreTokeniser is Cl100kPreTokeniser cl)
            obj["cl100k_variant"] = cl.Variant.ToString().ToLowerInvariant();
        if (model.GgufPreTokenizer is { Length: > 0 } pre)
            obj["gguf_pre"] = pre;
        if (model.GgufTokenizerModel is { Length: > 0 } ggufModel)
            obj["gguf_model"] = ggufModel;

        // SentencePiece models rank merges by per-token score. Persist them
        // parallel to vocab id order so a save/load keeps merge ordering.
        if (model.TokenScores is { } scores)
            obj["scores"] = new JsonArray([.. scores.Select(s => JsonValue.Create(s))]);

        return obj.ToJsonString(JsonOpts);
    }

    /// <summary>Loads a SharpMind native tokenizer JSON file.</summary>
    public static BpeModel Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Tokenizer file not found: {path}");

        return FromJson(File.ReadAllText(path));
    }

    /// <summary>Parses a SharpMind native tokenizer JSON string.</summary>
    public static BpeModel FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // Special tokens
        var st = root.GetProperty("special_tokens");
        var additional = st.TryGetProperty("additional", out var addEl)
            ? [.. addEl.EnumerateArray().Select(e => e.GetString()!)]
            : (IReadOnlyList<string>)[];

        var specials = new SpecialTokens(
            unk: st.GetProperty("unk").GetString()!,
            bos: st.GetProperty("bos").GetString()!,
            eos: st.GetProperty("eos").GetString()!,
            pad: st.GetProperty("pad").GetString()!,
            additional: additional);

        // Vocab — rebuild ordered list sorted by ID
        var ordered = root.GetProperty("vocab")
                          .EnumerateObject()
                          .OrderBy(p => p.Value.GetInt32())
                          .Select(p => p.Name)
                          .ToList();
        var vocab = new Vocabulary(ordered, specials);

        // Merges — ordered by rank (index in array = rank)
        var merges = root.GetProperty("merges")
                         .EnumerateArray()
                         .Select((el, rank) =>
                         {
                             string[] parts = el.GetString()!.Split(' ', 2);
                             return new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank);
                         })
                         .ToList();

        string? ggufPre = root.TryGetProperty("gguf_pre", out var preEl) ? preEl.GetString() : null;
        string? variant = root.TryGetProperty("cl100k_variant", out var varEl) ? varEl.GetString() : null;
        var preTokeniser = ParsePreTokeniserName(
            root.TryGetProperty("pre_tokeniser", out var ptEl)
                ? ptEl.GetString() ?? "gpt2" : "gpt2",
            variant, ggufPre);

        // Character-level tokenizers are stored with a "kind": "char" marker.
        // They carry a per-character vocab and no merge rules; the encoder
        // maps characters directly instead of running byte-level BPE.
        bool charMode = root.TryGetProperty("kind", out var kindEl)
                        && kindEl.GetString() == "char";

// SentencePiece merge-ranking scores, parallel to vocab id order.
        IReadOnlyList<float>? scores = null;
        if (root.TryGetProperty("scores", out var scoresEl))
            scores = [.. scoresEl.EnumerateArray().Select(e => e.GetSingle())];

        return new BpeModel(vocab, merges, preTokeniser, scores, charMode: charMode)
        {
            GgufPreTokenizer = ggufPre,
            GgufTokenizerModel = root.TryGetProperty("gguf_model", out var modelEl) ? modelEl.GetString() : null,
        };
    }

    // Helpers

    private static JsonObject BuildVocabObject(Vocabulary vocab)
    {
        var obj = new JsonObject();
        for (int i = 0; i < vocab.AllTokens.Count; i++)
            obj[vocab.AllTokens[i]] = i;
        return obj;
    }

    internal static string PreTokeniserName(IPreTokeniser pt) => pt switch
    {
        Gpt2PreTokeniser => "gpt2",
        Cl100kPreTokeniser => "cl100k",
        WhitespacePreTokeniser => "whitespace",
        _ => pt.GetType().Name.ToLowerInvariant()
    };

    /// <param name="variant">The stored <c>cl100k_variant</c>, written since the variants exist.</param>
    /// <param name="ggufPre">
    /// The stored source <c>tokenizer.ggml.pre</c>. Older files say only "cl100k" for every
    /// tiktoken-family model; the GGUF pre name picks the variant for them. Without either, plain
    /// cl100k: on a Qwen vocab that gives the same ids as Qwen2, because Qwen has no multi-digit tokens.
    /// </param>
    internal static IPreTokeniser ParsePreTokeniserName(string name, string? variant = null, string? ggufPre = null) => name.ToLowerInvariant() switch
    {
        "gpt2" => new Gpt2PreTokeniser(),
        "cl100k" or "cl100k_base" or "cl100kpretokeniser" =>
            Enum.TryParse<Cl100kVariant>(variant, ignoreCase: true, out var v) && Enum.IsDefined(v) ? new Cl100kPreTokeniser(v)
            : GgufConverter.FromGgufPre(ggufPre) as Cl100kPreTokeniser ?? new Cl100kPreTokeniser(Cl100kVariant.Cl100k),
        "whitespace" => new WhitespacePreTokeniser(),
        _ => new Gpt2PreTokeniser()
    };
}
