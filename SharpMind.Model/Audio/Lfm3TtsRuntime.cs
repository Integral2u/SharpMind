using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Model.Audio;

/// <summary>
/// Native LFM3-Audio text-to-speech runtime. It drives the text backbone with a TTS
/// prompt, samples one speech frame per backbone step with the MaskGIT depth decoder,
/// feeds the frame embeddings back into the backbone, and synthesizes the codes with
/// the LiquidCodec decoder.
///
/// The runtime does not own the transformer by default: Studio keeps the model in its
/// host and disposes it separately. It owns the projector reader, the depth decoder and
/// the codec decoder.
/// </summary>
public sealed class Lfm3TtsRuntime : IDisposable
{
    public const int SampleRate = LiquidCodecDecoder.SampleRate;

    /// <summary>Codec frames per second: 1920 samples per frame at 24 kHz.</summary>
    public const double FramesPerSecond = 12.5;

    private const int DefaultAudioStartTokenId = 125012;

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly ModelConfig _config;
    private readonly GgufTensorReader _reader;
    private readonly Lfm3AudioCodeGenerator _codeGenerator;
    private readonly LiquidCodecDecoder _codec;
    private readonly bool _ownsModel;
    private readonly int _audioStartTokenId;
    private bool _disposed;

    public Lfm3TtsRuntime(
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
                $"Audio output needs an LFM3 model; this model is '{config.Architecture}'.");

        _model = model;
        _tokenizer = tokenizer;
        _config = config;
        _reader = new GgufTensorReader(projectorPath);
        try
        {
            _codeGenerator = new Lfm3AudioCodeGenerator(_reader);
            _codec = new LiquidCodecDecoder(_reader);
        }
        catch
        {
            _reader.Dispose();
            throw;
        }
        _ownsModel = ownsModel;
        _audioStartTokenId = ResolveTokenId(tokenizer, "<|audio_start|>", DefaultAudioStartTokenId);
    }

    /// <summary>Upper bound on generated speech frames (one frame is 80 ms).</summary>
    public int MaxFrames { get; init; } = 256;

    /// <summary>Language phrase used in the system prompt, for example "American English".</summary>
    public string Language { get; init; } = "American English";

    /// <summary>
    /// Voice phrase used in the system prompt: "male" or "female". The checkpoint's TTS
    /// template defaults to "female"; other voices are not equally well covered.
    /// </summary>
    public string Voice { get; init; } = "female";

    /// <summary>Depth-decoder sampling temperature (the checkpoint uses 1.0).</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Depth-decoder top-k (the checkpoint uses 128).</summary>
    public int TopK { get; init; } = 128;

    /// <summary>Per-codebook repetition penalty (the checkpoint uses 1.1).</summary>
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>Frames of code history covered by the repetition penalty (4096 in the checkpoint).</summary>
    public int RepetitionWindow { get; init; } = 4096;

    /// <summary>Sampling seed; null picks a time-based seed.</summary>
    public int? Seed { get; init; }

    /// <summary>Optional per-frame hook: (frame index, twelve codes).</summary>
    public Action<int, int[]>? FrameSampled { get; init; }

    /// <summary>Optional hook invoked when a frame contains the end-of-audio code.</summary>
    public Action<int, int[]>? EndOfSpeech { get; init; }

    /// <summary>True when the projector carries the depth decoder and the LiquidCodec weights.</summary>
    public static bool IsTtsProjector(string path)
    {
        try
        {
            using var reader = new GgufTensorReader(path);
            return reader.TryGetInfo("a.gen.code.proj_in.weight", out _) &&
                   reader.TryGetInfo("a.gen.wav.tfm.in_proj.weight", out _);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Builds the prompt the chat template renders for a prefilled assistant turn.</summary>
    public string BuildPrompt(string text) => BuildPrompt(text, Language, Voice);

    /// <summary>
    /// Renders the ChatML prompt for a prefilled assistant turn. The assistant turn stops at
    /// <c>assistant\n</c>: the runtime appends the <c>&lt;|audio_start|&gt;</c> token itself.
    /// </summary>
    public static string BuildPrompt(string text, string language, string voice)
    {
        ArgumentNullException.ThrowIfNull(text);
        string system = $"Perform TTS in {language} using a {voice} voice.";
        return $"<|im_start|>system\n{system}<|im_end|>\n" +
               $"<|im_start|>user\n{text}<|im_end|>\n" +
               "<|im_start|>assistant\n";
    }

    /// <summary>Synthesizes <paramref name="text"/> into 24 kHz mono samples.</summary>
    public float[] Synthesize(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);
        if (MaxFrames <= 0) return [];

        int[] promptIds = _tokenizer.Encode(BuildPrompt(text), addBos: true, addEos: false);
        if (promptIds.Length == 0) return [];

        var ids = new int[promptIds.Length + 1];
        promptIds.CopyTo(ids, 0);
        ids[^1] = _audioStartTokenId;

        IKVCache[] caches = CreateCaches(Math.Max(2048, ids.Length + MaxFrames + 8));
        try
        {
            float[] hidden = Prefill(ids, caches, cancellationToken);
            List<int[]> frames = GenerateFrames(hidden, ids.Length, caches, cancellationToken);
            if (frames.Count == 0) return [];
            return _codec.Decode(frames);
        }
        finally
        {
            foreach (IKVCache cache in caches)
                if (cache is IDisposable disposable) disposable.Dispose();
        }
    }

    /// <summary>Synthesizes <paramref name="text"/> and writes a 24 kHz mono WAV.</summary>
    public float[] SynthesizeToWav(string text, string path, CancellationToken cancellationToken = default)
    {
        float[] audio = Synthesize(text, cancellationToken);
        if (audio.Length > 0) WavFile.Write(path, audio, SampleRate);
        return audio;
    }

    /// <summary>
    /// Describes the backbone hidden state and the first MaskGIT pass. The top codes per codebook
    /// show whether the depth decoder is discriminating or sampling from a flat distribution.
    /// </summary>
    public string DescribeFirstFrame(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int[] promptIds = _tokenizer.Encode(BuildPrompt(text), addBos: true, addEos: false);
        var ids = new int[promptIds.Length + 1];
        promptIds.CopyTo(ids, 0);
        ids[^1] = _audioStartTokenId;

        IKVCache[] caches = CreateCaches(Math.Max(2048, ids.Length + 64));
        try
        {
            (float[] hidden, float[] textLogits) = PrefillWithLogits(ids, caches, default);
            double sum = 0;
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            foreach (float value in hidden)
            {
                sum += (double)value * value;
                if (value < min) min = value;
                if (value > max) max = value;
            }

            float[] full = PrefillWholePrompt(ids);
            double diffSum = 0;
            float maxDiff = 0;
            for (int i = 0; i < hidden.Length; i++)
            {
                float diff = Math.Abs(hidden[i] - full[i]);
                diffSum += (double)diff * diff;
                if (diff > maxDiff) maxDiff = diff;
            }

            Tensor<float> normWeight = _model.FinalNorm.NormWeight;
            double weightMean = 0, weightRms = 0;
            foreach (float value in normWeight.Data)
            {
                weightMean += value;
                weightRms += (double)value * value;
            }
            weightMean /= normWeight.Data.Length;
            weightRms = Math.Sqrt(weightRms / normWeight.Data.Length);

            // If the hidden is norm(raw) * weight, then mean((hidden / weight)^2) == 1.
            double consistency = 0;
            for (int i = 0; i < hidden.Length; i++)
            {
                double ratio = hidden[i] / normWeight.Data[i];
                consistency += ratio * ratio;
            }
            consistency /= hidden.Length;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"tokens={ids.Length} audioStart={_audioStartTokenId} " +
                          $"hiddenRms={Math.Sqrt(sum / hidden.Length):F4} min={min:F4} max={max:F4}");
            sb.AppendLine($"finalNorm mean={weightMean:F4} rms={weightRms:F4} " +
                          $"normConsistency={consistency:F4} (1 = weight applied)");
            sb.AppendLine($"audioStartLookup={_tokenizer.TokenToId("<|audio_start|>")} " +
                          $"audioLookup={_tokenizer.TokenToId("<|audio|>")}");
            sb.Append("ids[0..10]=");
            for (int i = 0; i < Math.Min(10, ids.Length); i++) sb.Append($"{ids[i]} ");
            sb.AppendLine();
            sb.Append("ids[end]=");
            for (int i = Math.Max(0, ids.Length - 6); i < ids.Length; i++) sb.Append($"{ids[i]} ");
            sb.AppendLine();
            sb.Append("textTop=");
            foreach ((int token, float logit) in TopTokens(textLogits, 8))
                sb.Append($"{token}({logit:F2}) ");
            sb.AppendLine($"audioOut={(125032 < textLogits.Length ? textLogits[125032] : float.NaN):F2}");
            sb.AppendLine();
            sb.AppendLine($"splitVsFull rmsDiff={Math.Sqrt(diffSum / hidden.Length):G4} maxDiff={maxDiff:G4}");
            sb.Append("hidden[0..8]=");
            for (int i = 0; i < Math.Min(8, hidden.Length); i++) sb.Append($"{hidden[i]:F4} ");
            sb.AppendLine();

            int[] codes = new int[Lfm3AudioCodeGenerator.Codebooks];
            var revealed = new bool[Lfm3AudioCodeGenerator.Codebooks];
            float[] logits = _codeGenerator.ComputeLogits(hidden, codes, revealed);
            for (int cb = 0; cb < Lfm3AudioCodeGenerator.Codebooks; cb++)
            {
                var top = TopCodes(logits, cb, 5);
                sb.Append($"cb{cb,2}: ");
                foreach ((int code, float logit) in top) sb.Append($"{code}({logit:F2}) ");
                sb.AppendLine($"eos={logits[cb * Lfm3AudioCodeGenerator.CodebookSize + Lfm3AudioCodeGenerator.EosCode]:F2}");
            }

            var history = new List<int>[Lfm3AudioCodeGenerator.Codebooks];
            for (int cb = 0; cb < history.Length; cb++) history[cb] = new List<int>();
            int[] frame = _codeGenerator.GenerateFrame(hidden, history, new Random(1234));
            sb.AppendLine($"firstFrame={string.Join(",", frame)}");
            return sb.ToString();
        }
        finally
        {
            foreach (IKVCache cache in caches)
                if (cache is IDisposable disposable) disposable.Dispose();
        }
    }

    private float[] PrefillWholePrompt(int[] ids)
    {
        IKVCache[] caches = CreateCaches(Math.Max(2048, ids.Length + 64));
        try
        {
            var idTensor = new Tensor<int>(1, ids.Length);
            for (int i = 0; i < ids.Length; i++) idTensor.Data[i] = ids[i];
            Tensor<float> embeddings = _model.ForwardEmbedding(idTensor);
            idTensor.Dispose();
            using Tensor<float> last = _model.ForwardLastNormedEmbedding(embeddings, caches, 0);
            var result = new float[_config.HiddenDim];
            last.Data[.._config.HiddenDim].CopyTo(result);
            return result;
        }
        finally
        {
            foreach (IKVCache cache in caches)
                if (cache is IDisposable disposable) disposable.Dispose();
        }
    }

    private static List<(int Code, float Logit)> TopCodes(float[] logits, int codebook, int count)
    {
        int offset = codebook * Lfm3AudioCodeGenerator.CodebookSize;
        var top = new List<(int Code, float Logit)>(count);
        for (int code = 0; code < Lfm3AudioCodeGenerator.CodebookSize; code++)
        {
            float logit = logits[offset + code];
            if (top.Count == count && logit <= top[^1].Logit) continue;
            top.Add((code, logit));
            top.Sort((a, b) => b.Logit.CompareTo(a.Logit));
            if (top.Count > count) top.RemoveAt(top.Count - 1);
        }
        return top;
    }

    private float[] Prefill(int[] ids, IKVCache[] caches, CancellationToken cancellationToken) =>
        PrefillWithLogits(ids, caches, cancellationToken).Hidden;

    private (float[] Hidden, float[] Logits) PrefillWithLogits(
        int[] ids, IKVCache[] caches, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Prefill every prompt token except the last, then step the final <|audio_start|>
        // on its own: the single-token path publishes the normalised hidden state the
        // depth decoder needs.
        var prefill = new Tensor<int>(1, ids.Length - 1);
        for (int i = 0; i < ids.Length - 1; i++) prefill.Data[i] = ids[i];
        Tensor<float> embeddings = _model.ForwardEmbedding(prefill);
        prefill.Dispose();
        _model.ForwardLastLogitsEmbedding(embeddings, caches, 0).Dispose();

        var last = new Tensor<int>(1, 1);
        last.Data[0] = ids[^1];
        Tensor<float> logitsTensor = _model.ForwardLastLogits(last, caches, ids.Length - 1);
        last.Dispose();
        var logits = new float[logitsTensor.Data.Length];
        logitsTensor.Data.CopyTo(logits);
        logitsTensor.Dispose();

        return (CopyLastNormed(), logits);
    }

    private static List<(int Token, float Logit)> TopTokens(float[] logits, int count)
    {
        var top = new List<(int Token, float Logit)>(count);
        for (int token = 0; token < logits.Length; token++)
        {
            float logit = logits[token];
            if (top.Count == count && logit <= top[^1].Logit) continue;
            top.Add((token, logit));
            top.Sort((a, b) => b.Logit.CompareTo(a.Logit));
            if (top.Count > count) top.RemoveAt(top.Count - 1);
        }
        return top;
    }

    private List<int[]> GenerateFrames(float[] hidden, int position, IKVCache[] caches, CancellationToken cancellationToken)
    {
        var frames = new List<int[]>(Math.Min(MaxFrames, 512));
        var history = new List<int>[Lfm3AudioCodeGenerator.Codebooks];
        for (int cb = 0; cb < history.Length; cb++) history[cb] = new List<int>(Math.Min(MaxFrames, 512));

        _codeGenerator.Temperature = Temperature;
        _codeGenerator.TopK = TopK;
        _codeGenerator.RepetitionPenalty = RepetitionPenalty;
        _codeGenerator.RepetitionWindow = RepetitionWindow;

        var random = new Random(Seed ?? Environment.TickCount);
        while (frames.Count < MaxFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int[] codes = _codeGenerator.GenerateFrame(hidden, history, random);
            if (Array.IndexOf(codes, Lfm3AudioCodeGenerator.EosCode) >= 0)
            {
                EndOfSpeech?.Invoke(frames.Count, codes);
                break;
            }

            frames.Add(codes);
            FrameSampled?.Invoke(frames.Count - 1, codes);
            for (int cb = 0; cb < codes.Length; cb++) history[cb].Add(codes[cb]);

            float[] feedback = _codeGenerator.Feedback(codes);
            var embedding = new Tensor<float>(1, 1, _config.HiddenDim);
            feedback.CopyTo(embedding.Data);
            _model.ForwardLastLogitsEmbedding(embedding, caches, position++).Dispose();
            hidden = CopyLastNormed();
        }

        return frames;
    }

    private float[] CopyLastNormed()
    {
        Tensor<float>? normed = _model.LastNormed;
        if (normed is null)
            throw new InvalidOperationException("The backbone did not publish its last hidden state.");
        var result = new float[_config.HiddenDim];
        normed.Data[.._config.HiddenDim].CopyTo(result);
        return result;
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
            // The vocabulary may not carry the placeholder; the LFM3-Audio config does.
        }
        return fallback;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _codeGenerator.Dispose();
        _codec.Dispose();
        _reader.Dispose();
        if (_ownsModel) _model.Dispose();
    }
}
