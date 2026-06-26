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
            string arch = meta.GetString("general.architecture", "llama");

            // Derive vocab size from tokenizer tokens or embedding shape
            int vocabSize = 32000;
            var tokKv = meta.KvPairs.FirstOrDefault(k => k.Key == "tokenizer.ggml.tokens");
            if (tokKv.Value is List<string> list)
                vocabSize = list.Count;
            else
            {
                var embdInfo = meta.Tensors.FirstOrDefault(
                    t => t.Name.Contains("token_embd") && t.Name.Contains("weight"));
                if (embdInfo.Shape is { Length: >= 2 })
                {
                    long d0 = embdInfo.Shape[0], d1 = embdInfo.Shape[1];
                    vocabSize = (int)(d0 > d1 ? d0 : d1);
                }
            }
            vocabSize = (int)meta.GetLong($"{arch}.vocab_size",
                        meta.GetLong("tokenizer.ggml.token_count", vocabSize));

            var config = new SharpMindModelConfig
            {
                VocabSize = vocabSize,
                HiddenDim = (int)meta.GetLong($"{arch}.embedding_length", 2048),
                NumLayers = (int)meta.GetLong($"{arch}.block_count", 22),
                NumHeads = (int)meta.GetLong($"{arch}.attention.head_count", 32),
                NumKvHeads = (int)meta.GetLong($"{arch}.attention.head_count_kv", 4),
                FfnDim = (int)meta.GetLong($"{arch}.feed_forward_length", 5632),
                MaxSeqLen = (int)meta.GetLong($"{arch}.context_length", 2048),
                Source = $"{arch}/{meta.GetString("general.name", "model")}",
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