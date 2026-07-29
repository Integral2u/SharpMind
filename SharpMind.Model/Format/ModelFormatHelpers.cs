using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Tokenization;
using System;
using System.Collections.Generic;
using System.Data;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpMind.Model.Format
{
    public static class ModelFormatHelpers
    {
        public static string GetExtension(this ModelFormat format) => format switch
        {
            ModelFormat.Gguf => ".gguf",
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };

        private static HashSet<string>? extenstions;
        public static HashSet<string> GetExtensions()
        {
            if(extenstions!=null) return extenstions;
            extenstions = [];
            foreach (var fmt in Enum.GetValues<ModelFormat>())
            {
                extenstions.Add(GetExtension(fmt));
            }
            return extenstions;
        }

        public static ModelFormat? GetFormatForExtension(string extension)
        {
            if (string.IsNullOrWhiteSpace(extension)) return null;
            foreach (var fmt in Enum.GetValues<ModelFormat>())
            {
                if (extension.EndsWith(GetExtension(fmt), StringComparison.InvariantCultureIgnoreCase)) return fmt;
            }
            return null;
        }

        public static IModelLoader GetModelLoaderFor(this ModelFormat format, QuantizationOps qOps, string path, ModelConfig config) => format switch
        {
            ModelFormat.Gguf => new GgufLoader(qOps, path, config),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };
        private readonly static GgufModelFormatMetaHelper ggufModelFormatMetaHelper = new();
        public static IModelFormatMetaHelper GgufMetaHelper { get { return ggufModelFormatMetaHelper; } }
        public static IModelFormatMetaHelper GetModelMetaHelperFor(this ModelFormat format) => format switch 
        {
            ModelFormat.Gguf => GgufMetaHelper,
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };
        private class GgufModelFormatMetaHelper: IModelFormatMetaHelper
        {
            public void Load(string ggufPath, string? tokenizerPath, out ModelMetaData meta, out ModelConfig config, out Tokenizer? tokenizer) => GgufLoader.Load(ggufPath, tokenizerPath, out meta, out config, out tokenizer);
            public ModelMetaData LoadMeta(string path) => GgufLoader.LoadMeta(path);
            public ModelConfig? LoadConfig(ModelMetaData meta) => GgufLoader.LoadConfig(meta);
            public Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta) => GgufLoader.LoadTokenizerFromMeta(meta);
        }
    }
}
