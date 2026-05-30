using SharpMind.Core.Training;

namespace SharpMind.Model.Format;
public static partial class ModelConverter
{
    public static class Convert
    {
        public static void ToNative(string ggufPath, string outputDir)
        {
            var meta = GgufLoader.LoadMeta(ggufPath);
            var weights = GgufLoader.LoadWeights(ggufPath);
            
            int vocabSize = 32000;
            if (meta.KvPairs.Any(k => k.Key == "tokenizer.ggml.tokens"))
            {
                var kv = meta.KvPairs.First(k => k.Key == "tokenizer.ggml.tokens");
                if (kv.Value is List<string> list)
                    vocabSize = list.Count;
            }
            
            var config = new SharpMindModelConfig
            {
                VocabSize = vocabSize,
                HiddenDim = (int)meta.GetLong("embedding", meta.GetLong("llama.embedding_length", 2048)),
                NumLayers = (int)meta.GetLong("llama.block_count", 22),
                NumHeads = (int)meta.GetLong("llama.attention.head_count", 32),
                NumKvHeads = (int)meta.GetLong("llama.attention.head_count_kv", 4),
                FfnDim = (int)meta.GetLong("llama.feed_forward_length", 5632),
                MaxSeqLen = (int)meta.GetLong("llama.context_length", 2048),
                Source = meta.GetString("general.architecture", "llama") + "/" + meta.GetString("general.name", "model"),
            };
            
            var parameters = new List<SharpMind.Core.Training.Parameter>();
            foreach (var kvp in weights)
            {
                var name = MapWeightName(kvp.Key);
                if (name != null)
                    parameters.Add(new Parameter(name, kvp.Value));
            }
                  
            SaveSharpMind(parameters, config, outputDir);
          
            foreach (var w in weights)
                w.Value.Dispose();
        }

        private static string? MapWeightName(string ggufName)
        {
            if (ggufName.Contains("token_embd") || ggufName.Contains("output"))
                return null;
            return ggufName.Replace(".", "_").Replace("-", "_");
        }

        public static ConversionResult FromNative(string modelDir) => LoadSharpMind(modelDir);
    }
}