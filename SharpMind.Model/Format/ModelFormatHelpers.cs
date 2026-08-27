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
            ModelFormat.Smm => ".smm",
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

        /// <param name="useSafeIo">
        /// False (default) preserves current behavior exactly: tensor data is read
        /// via a memory-mapped view. Set true on platforms where memory-mapped
        /// files aren't available -- notably wasm-browser, where
        /// System.IO.MemoryMappedFiles throws PlatformNotSupportedException --
        /// to fall back to a plain FileStream instead. See WeightStreamFactory.
        /// </param>
        public static IModelLoader GetModelLoaderFor(this ModelFormat format, QuantizationOps qOps, string path, ModelConfig config, bool useSafeIo = false) => format switch
        {
            ModelFormat.Gguf => new GgufLoader(qOps, path, config, useSafeIo),
            ModelFormat.Smm => new SmmLoader(qOps, path, config, useSafeIo),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };
        private readonly static GgufModelFormatMetaHelper ggufModelFormatMetaHelper = new();
        public static IModelFormatMetaHelper GgufMetaHelper { get { return ggufModelFormatMetaHelper; } }
        private readonly static SmmModelFormatMetaHelper smmModelFormatMetaHelper = new();
        public static IModelFormatMetaHelper SmmMetaHelper { get { return smmModelFormatMetaHelper; } }
        public static IModelFormatMetaHelper GetModelMetaHelperFor(this ModelFormat format) => format switch 
        {
            ModelFormat.Gguf => GgufMetaHelper,
            ModelFormat.Smm => SmmMetaHelper,
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, null)
        };

        /// <summary>
        /// Loads metadata for a model file by dispatching on its extension.
        /// Used by streaming weight init so both GGUF and SMM files share one path.
        /// </summary>
        public static ModelMetaData LoadMetaForFile(string path)
        {
            var fmt = GetFormatForExtension(path) ?? throw new InvalidDataException($"File type not supported: {path}");
            return GetModelMetaHelperFor((ModelFormat)fmt).LoadMeta(path);
        }

        private class GgufModelFormatMetaHelper: IModelFormatMetaHelper
        {
            public void Load(string ggufPath, string? tokenizerPath, out ModelMetaData meta, out ModelConfig config, out Tokenizer? tokenizer) => GgufLoader.Load(ggufPath, tokenizerPath, out meta, out config, out tokenizer);
            public ModelMetaData LoadMeta(string path) => GgufLoader.LoadMeta(path);
            public ModelConfig? LoadConfig(ModelMetaData meta) => GgufLoader.LoadConfig(meta);
            public Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta) => GgufLoader.LoadTokenizerFromMeta(meta);
        }

        private class SmmModelFormatMetaHelper : IModelFormatMetaHelper
        {
            public void Load(string path, string? tokenizerPath, out ModelMetaData meta, out ModelConfig config, out Tokenizer? tokenizer) => SmmLoader.Load(path, tokenizerPath, out meta, out config, out tokenizer);
            public ModelMetaData LoadMeta(string path) => SmmLoader.LoadMeta(path);
            public ModelConfig? LoadConfig(ModelMetaData meta) => SmmLoader.LoadConfig(meta);
            public Tokenizer? LoadTokenizerFromMeta(ModelMetaData meta) => SmmLoader.LoadTokenizerFromMeta(meta);
        }
    }
}
