using System.Text.Json.Nodes;
using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using SharpMind.Training;

namespace SharpMind.Tests;

/// <summary>
/// Deterministic tiny reference model used by the CUI/session tests in place of
/// a real downloaded GGUF. Building a fixed-seed, 2048-vocab / hidden-64 /
/// 2-layer decoder and exporting it to a temp .SMM takes milliseconds, so the
/// full session path (SessionLauncher.LoadModelAsync → BuildSession → chat)
/// is exercised end-to-end without ever loading a real model file.
///
/// Weights are random GPT-2-style (std 0.02) from a fixed seed — the tests
/// never assert on trained quality, only on plumbing (token semantics, prompt
/// sizing, streaming, progress events), so deterministic garbage weights are
/// exactly as good as a real checkpoint and infinitely cheaper.
/// </summary>
internal sealed class TinyReferenceModel
{
    public const int VocabSize = 2048;
    public const int MaxSeqLen = 2048;

    private TinyReferenceModel(string smmPath) => SmmPath = smmPath;

    /// <summary>Path to the temp .SMM container hosting the reference weights + tokenizer.</summary>
    public string SmmPath { get; }

    public static TinyReferenceModel Create(TempDirectory temp)
    {
        var config = new ModelConfig
        {
            Architecture = "gpt2",
            VocabSize = VocabSize,
            HiddenDim = 64,
            NumLayers = 2,
            NumHeads = 4,
            NumKvHeads = 4,
            FfnDim = 256,
            MaxSeqLen = MaxSeqLen,
        };
        var sharpConfig = SharpMindConfig.ForModel(4, 4, "gpt2", HardwareTier.Scalar);
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        try
        {
            WeightInitializer.InitializeRandomly(weights, seed: 20260819);
            string path = Path.Combine(temp.Path, "tiny-reference.smm");
            SmmTrainingExporter.Export(weights, BuildTokenizer(), path, new SmmWriteOptions { Source = "reference" });
            return new TinyReferenceModel(path);
        }
        finally
        {
            weights.Dispose();
        }
    }

    /// <summary>
    /// Whitespace tokenizer whose vocab covers every model output id — filler
    /// <c>&lt;t0&gt;</c> … plus the specials, which are deliberately reserved
    /// <em>inside</em> [0, VocabSize) (the last four rows) so a BOS/EOS the
    /// chat pipeline injects has a real embedding row instead of throwing
    /// "Token ID out of range" at Forward. Unknown words map to <c>&lt;unk&gt;</c>;
    /// encode just counts whitespace-separated chunks, which is all the session
    /// tests need (token counts / streaming plumbing).
    /// </summary>
    private static Tokenizer BuildTokenizer()
    {
        var vocabObj = new JsonObject();
        int specialStart = VocabSize - 4;
        for (int i = 0; i < specialStart; i++)
            vocabObj[$"<t{i}>"] = i;
        vocabObj["<unk>"] = specialStart;
        vocabObj["<s>"] = specialStart + 1;
        vocabObj["</s>"] = specialStart + 2;
        vocabObj["<pad>"] = specialStart + 3;

        var root = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokeniser"] = "whitespace",
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = "<unk>",
                ["bos"] = "<s>",
                ["eos"] = "</s>",
                ["pad"] = "<pad>",
                ["additional"] = new JsonArray(),
            },
            ["vocab"] = vocabObj,
            ["merges"] = new JsonArray(),
        };
        return Tokenizer.FromJson(root.ToJsonString());
    }
}