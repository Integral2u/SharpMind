using SharpMind.Model.Config;
using SharpMind.Tokenization;

namespace SharpMind.Model.Format
{
    public interface IModelFormatMetaHelper
    {
        public void Load(string ggufPath, string? tokenizerPath, out ModelMetaData meta, out ModelConfig config, out Tokenizer? tokenizer);
        public ModelMetaData LoadMeta(string path);
        public ModelConfig? LoadConfig(ModelMetaData meta);
        public Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta);
    }
}
