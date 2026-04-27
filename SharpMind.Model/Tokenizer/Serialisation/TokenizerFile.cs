using SharpMind.Model.Tokenizer.Vocab;
using SharpMind.Model.Tokenizer.Bpe;
using SharpMind.Model.Tokenizer.PreTokeniser;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SharpMind.Model.Tokenizer.Serialisation;

/// <summary>
/// Saves and loads tokenizer state as JSON.
///
/// SharpMind native format:
/// {
///   "version": "1.0",
///   "pre_tokeniser": "gpt2" | "whitespace",
///   "special_tokens": { "unk": "...", "bos": "...", "eos": "...", "pad": "...",
///                       "additional": [...] },
///   "vocab": { "token": id, ... },
///   "merges": [ "left right", ... ]       ← ordered by rank (0 = highest priority)
/// }
///
/// HuggingFace tokenizer.json is also supported for loading (not saving).
/// </summary>
public static class TokenizerFile
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };

    // ── Save ──────────────────────────────────────────────────────────────

    /// <summary>Saves a trained <see cref="BpeModel"/> to a JSON file.</summary>
    public static void Save(BpeModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var obj = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokeniser"] = PreTokeniserName(model.PreTokeniser),
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = model.Vocab.Specials.Unk,
                ["bos"] = model.Vocab.Specials.Bos,
                ["eos"] = model.Vocab.Specials.Eos,
                ["pad"] = model.Vocab.Specials.Pad,
                ["additional"] = new JsonArray(
                    model.Vocab.Specials.Additional
                         .Select(t => JsonValue.Create(t)!).ToArray()),
            },
            ["vocab"] = BuildVocabObject(model.Vocab),
            ["merges"] = new JsonArray(
                model.Merges
                     .OrderBy(m => m.Rank)
                     .Select(m => JsonValue.Create($"{m.Left} {m.Right}")!)
                     .ToArray()),
        };

        string dir = Path.GetDirectoryName(path) ?? ".";
        Directory.CreateDirectory(dir);
        File.WriteAllText(path, obj.ToJsonString(JsonOpts));
    }

    // ── Load native ───────────────────────────────────────────────────────

    /// <summary>Loads a SharpMind native tokenizer JSON file.</summary>
    public static BpeModel Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Tokenizer file not found: {path}");

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;

        // Special tokens
        var st = root.GetProperty("special_tokens");
        var additional = st.TryGetProperty("additional", out var addEl)
            ? addEl.EnumerateArray().Select(e => e.GetString()!).ToList()
            : (IReadOnlyList<string>)[];

        var specials = new SpecialTokens(
            st.GetProperty("unk").GetString()!,
            st.GetProperty("bos").GetString()!,
            st.GetProperty("eos").GetString()!,
            st.GetProperty("pad").GetString()!,
            additional);

        // Vocab — rebuild ordered list from id values
        var vocabEl = root.GetProperty("vocab");
        var ordered = vocabEl.EnumerateObject()
                             .OrderBy(p => p.Value.GetInt32())
                             .Select(p => p.Name)
                             .ToList();
        var vocab = new Vocabulary(ordered, specials);

        // Merges
        var merges = root.GetProperty("merges").EnumerateArray()
            .Select((el, rank) =>
            {
                string[] parts = el.GetString()!.Split(' ', 2);
                string left = parts[0];
                string right = parts[1];
                return new MergeRule(left, right, left + right, rank);
            })
            .ToList();

        var preTokeniser = LoadPreTokeniser(
            root.TryGetProperty("pre_tokeniser", out var ptEl)
                ? ptEl.GetString() ?? "gpt2"
                : "gpt2");

        return new BpeModel(vocab, merges, preTokeniser);
    }

    // ── Load HuggingFace tokenizer.json ───────────────────────────────────

    /// <summary>
    /// Loads a HuggingFace <c>tokenizer.json</c> file (BPE models only).
    /// Supports GPT-2, LLaMA 2/3, Mistral, Falcon, and any HF BPE tokenizer.
    /// </summary>
    public static BpeModel LoadHuggingFace(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"HuggingFace tokenizer file not found: {path}");

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;

        // Verify model type
        if (root.TryGetProperty("model", out var modelEl) &&
            modelEl.TryGetProperty("type", out var typeEl) &&
            typeEl.GetString() is string modelType &&
            !string.Equals(modelType, "BPE", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"Only BPE tokenizers are supported. Got: {modelType}");

        // Special tokens — HF stores as added_tokens array
        var (unk, bos, eos, pad, additional) = ExtractHfSpecials(root);
        var specials = new SpecialTokens(unk, bos, eos, pad, additional);

        // Vocab from model.vocab
        var hfVocab = root.GetProperty("model").GetProperty("vocab");
        var ordered = hfVocab.EnumerateObject()
                             .OrderBy(p => p.Value.GetInt32())
                             .Select(p => p.Name)
                             .ToList();
        var vocab = new Vocabulary(ordered, specials);

        // Merges from model.merges
        var merges = root.GetProperty("model").GetProperty("merges")
            .EnumerateArray()
            .Select((el, rank) =>
            {
                string[] parts = el.GetString()!.Split(' ', 2);
                return new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank);
            })
            .ToList();

        // Pre-tokeniser type from normalizer/pre_tokenizer fields
        var preTokeniser = DetectHfPreTokeniser(root);

        return new BpeModel(vocab, merges, preTokeniser);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static JsonObject BuildVocabObject(Vocabulary vocab)
    {
        var obj = new JsonObject();
        for (int i = 0; i < vocab.AllTokens.Count; i++)
            obj[vocab.AllTokens[i]] = i;
        return obj;
    }

    private static string PreTokeniserName(IPreTokeniser pt) => pt switch
    {
        Gpt2PreTokeniser => "gpt2",
        WhitespacePreTokeniser => "whitespace",
        _ => pt.GetType().Name.ToLowerInvariant()
    };

    private static IPreTokeniser LoadPreTokeniser(string name) => name switch
    {
        "gpt2" => new Gpt2PreTokeniser(),
        "whitespace" => new WhitespacePreTokeniser(),
        _ => new Gpt2PreTokeniser() // safe default
    };

    private static (string unk, string bos, string eos, string pad,
                    IReadOnlyList<string> additional)
        ExtractHfSpecials(JsonElement root)
    {
        string unk = SpecialTokens.DefaultUnk;
        string bos = SpecialTokens.DefaultBos;
        string eos = SpecialTokens.DefaultEos;
        string pad = SpecialTokens.DefaultPad;
        var additional = new List<string>();

        if (!root.TryGetProperty("added_tokens", out var tokens))
            return (unk, bos, eos, pad, additional);

        foreach (var t in tokens.EnumerateArray())
        {
            string? content = t.TryGetProperty("content", out var c) ? c.GetString() : null;
            bool special = t.TryGetProperty("special", out var s) && s.GetBoolean();
            if (content is null || !special) continue;

            string lower = content.ToLowerInvariant();
            if (lower is "<unk>" or "[unk]") unk = content;
            else if (lower is "<s>" or "[bos]" or "<bos>") bos = content;
            else if (lower is "</s>" or "[eos]" or "<eos>") eos = content;
            else if (lower is "<pad>" or "[pad]") pad = content;
            else additional.Add(content);
        }

        return (unk, bos, eos, pad, additional);
    }

    private static IPreTokeniser DetectHfPreTokeniser(JsonElement root)
    {
        if (root.TryGetProperty("pre_tokenizer", out var pt) &&
            pt.TryGetProperty("type", out var typeEl))
        {
            string type = typeEl.GetString() ?? "";
            if (type.Contains("ByteLevel", StringComparison.OrdinalIgnoreCase) ||
                type.Contains("GPT2", StringComparison.OrdinalIgnoreCase))
                return new Gpt2PreTokeniser();
        }
        return new Gpt2PreTokeniser(); // safe default for BPE
    }
}
