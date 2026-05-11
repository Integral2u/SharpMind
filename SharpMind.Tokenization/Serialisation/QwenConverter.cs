using System.Text.Json;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Serialisation;

public static class QwenConverter
{
    public static BpeModel Convert(string tokenizerJsonPath)
    {
        Console.WriteLine($"[QwenConverter] Starting: {tokenizerJsonPath}");
        
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerJsonPath);
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"Qwen tokenizer not found: {tokenizerJsonPath}");

        var fileInfo = new FileInfo(tokenizerJsonPath);
        Console.WriteLine($"[QwenConverter] File size: {fileInfo.Length} bytes");

        Console.WriteLine("[QwenConverter] Parsing JSON...");
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;
        
        if (!root.TryGetProperty("model", out var modelElement))
            throw new InvalidDataException("Qwen tokenizer.json missing 'model'");
        
        Console.WriteLine("[QwenConverter] Extracting special tokens...");
        var (unk, bos, eos, pad, additional) = QwenExtractSpecials(root);
        Console.WriteLine($"[QwenConverter] Specials: unk='{unk}', bos='{bos}', eos='{eos}', pad='{pad}'");
        
        var specials = new SpecialTokens(unk, bos, eos, pad, additional);
        Console.WriteLine("[QwenConverter] Building vocabulary...");
        
        var vocabList = new List<string>();
        
        // Load from model.vocab - get ordered by ID
        if (root.TryGetProperty("model", out var model) && 
            model.TryGetProperty("vocab", out var vocabElement))
        {
            vocabList = vocabElement.EnumerateObject()
                .OrderBy(p => p.Value.GetInt32())
                .Select(p => p.Name)
                .ToList();
            Console.WriteLine($"[QwenConverter] Raw vocab: {vocabList.Count}");

            // Add special tokens not already in vocab (at the end)
            foreach (var token in specials.All)
            {
                if (!vocabList.Contains(token))
                {
                    vocabList.Add(token);
                    Console.WriteLine($"[QwenConverter] Added special: '{token}'");
                }
            }
        }

        var vocab = new Vocabulary(vocabList, specials);
        Console.WriteLine($"[QwenConverter] Vocabulary created");
        
        Console.WriteLine("[QwenConverter] Building merges...");
        var merges = BuildMerges(root);
        Console.WriteLine($"[QwenConverter] Merges: {merges.Count} rules");

        Console.WriteLine("[QwenConverter] Creating BpeModel...");
        return new BpeModel(vocab, merges, new Gpt2PreTokeniser());
    }

    private static (string? unk, string? bos, string? eos, string? pad, List<string> additional) QwenExtractSpecials(JsonElement root)
    {
        // Qwen uses these special tokens
        string unk = "<|endoftext|>";
        string bos = "<|im_start|>";
        string eos = "<|im_end|>";
        string pad = "<|endoftext|>";
        var additional = new List<string>();

        // Check added_tokens
        if (root.TryGetProperty("added_tokens", out var added))
        {
            Console.WriteLine($"[QwenConverter] Found {added.GetArrayLength()} added tokens");
            foreach (var t in added.EnumerateArray())
            {
                string? content = t.TryGetProperty("content", out var c) ? c.GetString() : null;
                bool isSpecial = t.TryGetProperty("special", out var s) && s.GetBoolean();
                
                if (content != null && isSpecial)
                {
                    if (content.Contains("im_start")) bos = content;
                    else if (content.Contains("im_end")) eos = content;
                    else if (!additional.Contains(content)) additional.Add(content);
                    
                    var id = t.TryGetProperty("id", out var idEl) ? idEl.GetInt32() : -1;
                    Console.WriteLine($"[QwenConverter]   special: id={id} '{content}'");
                }
            }
        }

        return (unk, bos, eos, pad, additional);
    }

    private static List<MergeRule> BuildMerges(JsonElement root)
    {
        var merges = new List<MergeRule>();

        if (root.TryGetProperty("model", out var model) && 
            model.TryGetProperty("merges", out var mergesElement))
        {
            int rank = 0;
            foreach (var merge in mergesElement.EnumerateArray())
            {
                var mergeStr = merge.GetString();
                if (string.IsNullOrEmpty(mergeStr)) continue;
                
                var parts = mergeStr.Split(' ', 2);
                if (parts.Length == 2)
                {
                    var combined = parts[0] + parts[1];
                    merges.Add(new MergeRule(parts[0], parts[1], combined, rank++));
                }
            }
        }

        return merges;
    }
}