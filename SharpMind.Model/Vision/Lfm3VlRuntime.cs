using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Model.Vision;

/// <summary>
/// Native LFM3-VL image path. It encodes the image with <see cref="Lfm3VlVisionEncoder"/>,
/// scatters the projected features into the expanded <c>&lt;|image|&gt;</c> placeholders and
/// runs the ordinary LFM3 text backbone to produce a caption or answer.
/// </summary>
public sealed class Lfm3VlRuntime : IDisposable
{
    private const int DefaultImageTokenId = 125030;
    private const int DefaultVisionStartTokenId = 125009;
    private const int DefaultVisionEndTokenId = 125010;

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly ModelConfig _config;
    private readonly GgufTensorReader _reader;
    private readonly Lfm3VlVisionEncoder _vision;
    private readonly bool _ownsModel;
    private readonly int _imageTokenId;
    private readonly int _visionStartTokenId;
    private readonly int _visionEndTokenId;
    private bool _disposed;

    public Lfm3VlRuntime(
        Transformer model,
        Tokenizer tokenizer,
        ModelConfig config,
        string projectorPath,
        bool ownsModel = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectorPath);
        if (!config.Architecture.StartsWith("lfm3", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"Vision input needs an LFM3 text backbone; this model is '{config.Architecture}'.");

        _model = model;
        _tokenizer = tokenizer;
        _config = config;
        _reader = new GgufTensorReader(projectorPath);
        try
        {
            _vision = new Lfm3VlVisionEncoder(_reader);
        }
        catch
        {
            _reader.Dispose();
            throw;
        }
        _ownsModel = ownsModel;
        _imageTokenId = ResolveTokenId(tokenizer, "<|image|>", DefaultImageTokenId);
        _visionStartTokenId = ResolveTokenId(tokenizer, "<|vision_start|>", DefaultVisionStartTokenId);
        _visionEndTokenId = ResolveTokenId(tokenizer, "<|vision_end|>", DefaultVisionEndTokenId);
    }

    /// <summary>True when the projector carries the LFM3-VL vision tower.</summary>
    public static bool IsVisionProjector(string path) => Lfm3VlVisionEncoder.IsVisionProjector(path);

    /// <summary>Number of merged text-space tokens one image produces.</summary>
    public static int ImageTokens(int height, int width) =>
        (height / Lfm3VlVisionEncoder.PatchSize) * (width / Lfm3VlVisionEncoder.PatchSize) /
        (Lfm3VlVisionEncoder.Merge * Lfm3VlVisionEncoder.Merge);

    /// <summary>
    /// Describes an RGB image (CHW, values 0..1, size a multiple of 32) with the given prompt.
    /// The prompt is placed after the image placeholders in a ChatML user turn.
    /// </summary>
    public string Describe(
        ReadOnlySpan<float> chw,
        int height,
        int width,
        string prompt = "Describe this image in one sentence.",
        int maxNewTokens = 128,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prompt);
        if (maxNewTokens <= 0) return "";

        float[] features = _vision.Encode(chw, height, width);
        int imageTokens = features.Length / Lfm3VlVisionEncoder.TextHidden;
        if (imageTokens == 0) return "";

        int[] prefix = _tokenizer.Encode(
            "<|im_start|>user\n", addBos: true, addEos: false);
        int[] start = _tokenizer.Encode("<|vision_start|>", addBos: false, addEos: false);
        int[] end = _tokenizer.Encode("<|vision_end|>", addBos: false, addEos: false);
        int[] suffix = _tokenizer.Encode(
            $"{prompt}<|im_end|>\n<|im_start|>assistant\n", addBos: false, addEos: false);

        var ids = new List<int>(prefix.Length + start.Length + imageTokens + end.Length + suffix.Length);
        ids.AddRange(prefix);
        ids.AddRange(start);
        int imageStart = ids.Count;
        for (int i = 0; i < imageTokens; i++) ids.Add(_imageTokenId);
        ids.AddRange(end);
        ids.AddRange(suffix);

        var idTensor = new Tensor<int>(1, ids.Count);
        for (int i = 0; i < ids.Count; i++) idTensor.Data[i] = ids[i];
        Tensor<float> embeddings = _model.ForwardEmbedding(idTensor);
        idTensor.Dispose();

        int hidden = embeddings.Shape[2];
        for (int t = 0; t < imageTokens; t++)
        {
            int source = t * Lfm3VlVisionEncoder.TextHidden;
            int destination = (imageStart + t) * hidden;
            for (int d = 0; d < hidden; d++)
                embeddings.Data[destination + d] = features[source + d];
        }

        IKVCache[] caches = CreateCaches(Math.Max(2048, ids.Count + maxNewTokens + 8));
        try
        {
            var generated = new List<int>(maxNewTokens);
            Tensor<float> logits = _model.ForwardLastLogitsEmbedding(embeddings, caches, 0);
            int position = ids.Count;
            int next = ArgMax(logits.Data[.._config.VocabSize]);
            logits.Dispose();

            while (generated.Count < maxNewTokens && next != _tokenizer.EosId)
            {
                cancellationToken.ThrowIfCancellationRequested();
                generated.Add(next);
                var token = new Tensor<int>(1, 1);
                token.Data[0] = next;
                logits = _model.ForwardLastLogits(token, caches, position++);
                token.Dispose();
                next = ArgMax(logits.Data[.._config.VocabSize]);
                logits.Dispose();
            }

            return _tokenizer.Decode(generated.ToArray()).Trim();
        }
        finally
        {
            foreach (IKVCache cache in caches)
                if (cache is IDisposable disposable) disposable.Dispose();
        }
    }

    private IKVCache[] CreateCaches(int maxSeq)
    {
        var builder = new KVCacherBuilder();
        var caches = new IKVCache[_config.NumLayers];
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            caches[layer] = _config.IsShortConvLayer(layer)
                ? new ShortConvCache(_config.ShortConvCacheLength - 1, _config.HiddenDim, 1, maxSeq)
                : _config.IsRecurrentLayer(layer)
                    ? new DeltaNetCache(_config.SsmConvKernel, _config.SsmConvDim, _config.SsmTimeStepRank,
                        _config.SsmStateSize, _config.SsmValueHeadDim, 1, maxSeq)
                    : builder.CreateKVCache(1, _config.NumKvHeads, maxSeq, _config.HeadDimForLayer(layer));
        }
        return caches;
    }

    private static int ResolveTokenId(Tokenizer tokenizer, string token, int fallback)
    {
        try
        {
            int id = tokenizer.TokenToId(token);
            if (id >= 0) return id;
        }
        catch
        {
            // The vocabulary may not carry the placeholder; the LFM3-VL config does.
        }
        return fallback;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vision.Dispose();
        _reader.Dispose();
        if (_ownsModel) _model.Dispose();
    }
}
