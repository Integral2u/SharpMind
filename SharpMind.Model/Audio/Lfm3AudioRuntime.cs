using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Model.Audio;

/// <summary>
/// End-to-end LFM3-Audio speech-to-text runtime. It pairs an already-loaded
/// LFM3-Audio text backbone with the audio projector (mmproj) and turns 16 kHz
/// samples into a transcript.
///
/// The runtime does not own the transformer by default: Studio keeps the model
/// in its host and disposes it separately. It does own the projector reader and
/// the speech encoder.
/// </summary>
public sealed class Lfm3AudioRuntime : IDisposable
{
    private const int DefaultAudioTokenId = 125031;

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly ModelConfig _config;
    private readonly GgufTensorReader _reader;
    private readonly Lfm3AudioEncoder _encoder;
    private readonly bool _ownsModel;
    private readonly int _audioTokenId;
    private bool _disposed;

    public Lfm3AudioRuntime(
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
                $"Audio input needs an LFM3 model; this model is '{config.Architecture}'.");

        _model = model;
        _tokenizer = tokenizer;
        _config = config;
        _reader = new GgufTensorReader(projectorPath);
        _encoder = new Lfm3AudioEncoder(_reader);
        _ownsModel = ownsModel;
        _audioTokenId = ResolveAudioTokenId(tokenizer);
    }

    /// <summary>True when the projector exposes the LFM3-Audio encoder tensors.</summary>
    public static bool IsAudioProjector(string path)
    {
        try
        {
            using var reader = new GgufTensorReader(path);
            return reader.TryGetInfo("a.blk.0.conv_dw.weight", out _) ||
                   reader.TryGetInfo("a.pre_encode.out.weight", out _);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Finds the audio projector next to a backbone. It looks in the backbone's
    /// folder and in sibling folders whose name starts with the backbone stem.
    /// </summary>
    public static string? FindProjector(string backbonePath)
    {
        string? directory = Path.GetDirectoryName(backbonePath);
        if (string.IsNullOrEmpty(directory) || !Directory.Exists(directory)) return null;

        string stem = Path.GetFileNameWithoutExtension(backbonePath);
        foreach (string suffix in new[] { "-Q8_0", "-Q4_K_M", "-Q4_0", "-F16", "-BF16", "-Q6_K", "-Q5_K_M" })
        {
            if (stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            {
                stem = stem[..^suffix.Length];
                break;
            }
        }

        var candidates = new List<string>();
        candidates.AddRange(Directory.EnumerateFiles(directory, "mmproj-*.gguf", SearchOption.TopDirectoryOnly));
        foreach (string child in Directory.EnumerateDirectories(directory))
        {
            if (!Path.GetFileName(child).StartsWith(stem, StringComparison.OrdinalIgnoreCase)) continue;
            candidates.AddRange(Directory.EnumerateFiles(child, "mmproj-*.gguf", SearchOption.TopDirectoryOnly));
        }
        if (candidates.Count == 0)
            candidates.AddRange(Directory.EnumerateFiles(directory, "mmproj-*.gguf", SearchOption.AllDirectories));

        return candidates
            .Where(IsAudioProjector)
            .OrderByDescending(path => Path.GetFileName(path).Contains(stem, StringComparison.OrdinalIgnoreCase))
            .ThenBy(path => path, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    /// <summary>Transcribes a WAV file; audio is resampled to 16 kHz mono.</summary>
    public string TranscribeWav(
        string path,
        string language = "English",
        int maxNewTokens = 256,
        CancellationToken cancellationToken = default)
    {
        float[] samples = WavFile.ReadMono(path, 16000);
        return Transcribe(samples, language, maxNewTokens, cancellationToken);
    }

    /// <summary>Transcribes WAV bytes; audio is resampled to 16 kHz mono.</summary>
    public string TranscribeWav(
        ReadOnlySpan<byte> wav,
        string language = "English",
        int maxNewTokens = 256,
        CancellationToken cancellationToken = default)
    {
        using var stream = new MemoryStream(wav.ToArray(), writable: false);
        float[] samples = WavFile.ReadMono(stream, 16000);
        return Transcribe(samples, language, maxNewTokens, cancellationToken);
    }

    /// <summary>Transcribes 16 kHz mono samples.</summary>
    public string Transcribe(
        ReadOnlySpan<float> samples,
        string language = "English",
        int maxNewTokens = 256,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (samples.Length == 0) return "";

        float[,] mel = LogMelSpectrogram.Compute(samples);
        float[,] audio = _encoder.Encode(mel);
        int audioCount = audio.GetLength(0);

        string prefix = $"<|im_start|>system\nPerform ASR in {language}.<|im_end|>\n<|im_start|>user\n";
        string suffix = "<|im_end|>\n<|im_start|>assistant\n";
        int[] prefixIds = _tokenizer.Encode(prefix, addBos: true, addEos: false);
        int[] suffixIds = _tokenizer.Encode(suffix, addBos: false, addEos: false);

        var ids = new List<int>(prefixIds.Length + audioCount + suffixIds.Length);
        ids.AddRange(prefixIds);
        for (int i = 0; i < audioCount; i++) ids.Add(_audioTokenId);
        ids.AddRange(suffixIds);

        var idTensor = new Tensor<int>(1, ids.Count);
        for (int i = 0; i < ids.Count; i++) idTensor.Data[i] = ids[i];
        Tensor<float> embeddings = _model.ForwardEmbedding(idTensor);
        idTensor.Dispose();

        int hidden = embeddings.Shape[2];
        int audioStart = prefixIds.Length;
        for (int t = 0; t < audioCount; t++)
            for (int d = 0; d < hidden; d++)
                embeddings.Data[(audioStart + t) * hidden + d] = audio[t, d];

        IKVCache[] caches = CreateCaches(Math.Max(2048, ids.Count + maxNewTokens + 8));
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

        foreach (IKVCache cache in caches)
            if (cache is IDisposable disposable) disposable.Dispose();

        return _tokenizer.Decode(generated.ToArray()).Trim();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _encoder.Dispose();
        _reader.Dispose();
        if (_ownsModel) _model.Dispose();
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

    private static int ResolveAudioTokenId(Tokenizer tokenizer)
    {
        try
        {
            int id = tokenizer.TokenToId("<|audio|>");
            if (id >= 0) return id;
        }
        catch
        {
            // The vocabulary may not carry the placeholder; the LFM3-Audio config does.
        }
        return DefaultAudioTokenId;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
