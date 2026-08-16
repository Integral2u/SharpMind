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
///   "pre_tokeniser": "gpt2" | "whitespace",
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

        var preTokeniser = ParsePreTokeniserName(
            root.TryGetProperty("pre_tokeniser", out var ptEl)
                ? ptEl.GetString() ?? "gpt2" : "gpt2");

        // Character-level tokenizers are stored with a "kind": "char" marker.
        // They carry a per-character vocab and no merge rules; the encoder
        // maps characters directly instead of running byte-level BPE.
        bool charMode = root.TryGetProperty("kind", out var kindEl)
                        && kindEl.GetString() == "char";

        return new BpeModel(vocab, merges, preTokeniser, charMode: charMode);
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
        WhitespacePreTokeniser => "whitespace",
        _ => pt.GetType().Name.ToLowerInvariant()
    };

    internal static IPreTokeniser ParsePreTokeniserName(string name) => name switch
    {
        "gpt2" => new Gpt2PreTokeniser(),
        "whitespace" => new WhitespacePreTokeniser(),
        _ => new Gpt2PreTokeniser()
    };
}
