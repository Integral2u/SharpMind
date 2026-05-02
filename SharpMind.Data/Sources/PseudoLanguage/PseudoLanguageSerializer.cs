using System.Text.Json;
using System.Text.Json.Nodes;

namespace SharpMind.Data.Sources.PseudoLanguage;

public static class PseudoLanguageSerializer
{
    public static void SaveVocabToJsonl(PseudoLanguageGenerator generator, string filePath)
    {
        using var writer = new StreamWriter(filePath);
        foreach (var word in generator.Vocabulary)
        {
            var entry = new JsonObject
            {
                ["text"] = word.Text,
                ["token_id"] = word.TokenId,
                ["category"] = word.BaseCategory.ToString(),
                ["base"] = word.Base,
            };
            writer.WriteLine(entry.ToJsonString());
        }
    }

    public static void SaveSequencesToJsonl(
        PseudoLanguageGenerator generator, 
        ComplexityLevel level, 
        int count,
        string filePath,
        bool includeGroundTruth = true)
    {
        var sequences = generator.GenerateSyntactic(count, level);
        
        using var writer = new StreamWriter(filePath);
        foreach (var seq in sequences)
        {
            var entry = new JsonObject
            {
                ["text"] = seq.RawText,
                ["token_ids"] = new JsonArray(seq.TokenIds.Select(i => JsonValue.Create(i)).ToArray()),
            };
            
            if (includeGroundTruth)
            {
                entry["ground_truth_text"] = seq.GroundTruthText;
                entry["ground_truth_ids"] = new JsonArray(seq.GroundTruthIds.Select(i => JsonValue.Create(i)).ToArray());
            }
            
            writer.WriteLine(entry.ToJsonString());
        }
    }

    public static void SaveModelConfig(PseudoLanguageGenerator generator, string filePath)
    {
        var rec = generator.GetModelSizeRecommendation();
        
        var config = new JsonObject
        {
            ["vocab_size"] = rec.VocabSize,
            ["embedding_dim"] = rec.EmbeddingDim,
            ["hidden_dim"] = rec.HiddenDim,
            ["num_layers"] = rec.NumLayers,
            ["head_dim"] = rec.HeadDim,
            ["num_heads"] = rec.NumHeads,
            ["ffn_dim"] = rec.FfnDim,
            ["estimated_params"] = rec.EstimatedParams,
        };
        
        File.WriteAllText(filePath, config.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    public static void SaveToDirectory(
        PseudoLanguageGenerator generator,
        string directory,
        ComplexityLevel level,
        int sequenceCount)
    {
        Directory.CreateDirectory(directory);
        
        SaveVocabToJsonl(generator, Path.Combine(directory, "vocab.jsonl"));
        SaveSequencesToJsonl(generator, level, sequenceCount, Path.Combine(directory, "sequences.jsonl"));
        SaveModelConfig(generator, Path.Combine(directory, "model_config.json"));
    }
}