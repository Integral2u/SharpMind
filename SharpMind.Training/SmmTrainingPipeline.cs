using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Training;

/// <summary>
/// End-to-end helpers for shipping a freshly trained weight set: reloading an
/// .SMM file back into an inference transformer and running greedy generation.
/// </summary>
public static class SmmTrainingPipeline
{
    /// <summary>
    /// Loads an .SMM file and rebuilds an inference <see cref="Transformer"/>.
    /// </summary>
    /// <param name="path">Path to the .SMM file.</param>
    /// <param name="tokenizer">The tokenizer embedded in the file.</param>
    /// <param name="config">The model config embedded in the file.</param>
    public static Transformer LoadForInference(string path, out Tokenizer tokenizer, out ModelConfig config)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        SmmLoader.Load(path, null, out _, out config, out var rawTokenizer);
        tokenizer = rawTokenizer ?? throw new InvalidDataException("SMM file is missing its tokenizer (smm.tokenizer).");

        var sharpConfig = config.ForModel();
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        var reloaded = ModelFactory.CreateWeights(config, sharpConfig, qOps, path, LoadMode.Full);
        reloaded.InitializeWeights();
        return ModelFactory.CreateTransformer(reloaded, sharpConfig, null, false);
    }

    /// <summary>
    /// Reads the chat template embedded in an .SMM file (meta key
    /// <c>tokenizer.chat_template</c>, the Jinja string used by
    /// <see cref="SharpMind.Inference.Chat.PromptFormatters.JinjaTemplateFormatter"/>).
    /// Returns <see langword="null"/> when the file carries no template.
    /// </summary>
    public static string? LoadChatTemplate(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return SmmLoader.LoadMeta(path).GetChatTemplate();
    }

    /// <summary>
    /// Runs greedy (argmax) generation from <paramref name="prompt"/>, appending
    /// <paramref name="steps"/> new token ids. Returns the full id sequence
    /// (prompt followed by the generated ids).
    /// </summary>
    public static List<int> GenerateGreedy(Transformer model, IReadOnlyList<int> prompt, int vocab, int steps)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);

        var ids = new List<int>(prompt);
        for (int step = 0; step < steps; step++)
        {
            using var tokens = Tensor<int>.From(ids.ToArray(), 1, ids.Count);
            using var logits = model.Forward(tokens);
            int s = ids.Count;
            float max = float.NegativeInfinity;
            int best = -1;
            for (int v = 0; v < vocab; v++)
            {
                float l = logits.Data[(s - 1) * vocab + v];
                if (float.IsFinite(l) && l > max) { max = l; best = v; }
            }
            if (best < 0 || best >= vocab)
                throw new InvalidOperationException($"Generation produced invalid id {best}.");
            ids.Add(best);
        }
        return ids;
    }
}
