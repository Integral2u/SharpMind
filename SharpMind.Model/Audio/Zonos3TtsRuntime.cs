using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Model.Audio;

/// <summary>
/// Native ZONOS3 Mini text-to-speech runtime. The backbone is a frame model: one row of
/// ten audio codebooks per step, with codebook <c>cb</c> delayed by <c>cb</c> frames. Each
/// step the mmproj codebook head scores ten 4098-way codebooks, the frame is sampled and
/// fed back as the sum of its codebook embeddings, and the finished frames are decoded by
/// the ZAC codec to 44.1 kHz mono.
///
/// The runtime does not own the transformer by default: Studio keeps the model in its
/// host and disposes it separately. It owns the projector reader, the codebook tables and
/// the ZAC decoder.
/// </summary>
public sealed class Zonos3TtsRuntime : ITtsRuntime
{
    public const int SampleRate = ZacCodecDecoder.SampleRate;
    public const int SamplesPerFrame = ZacCodecDecoder.SamplesPerFrame;

    int ITtsRuntime.SampleRate => SampleRate;

    /// <summary>The release's default predefined voice.</summary>
    public const string DefaultVoice = "Savannah_en-American_F_1";

    private const int Codebooks = Zonos3Codebook.Codebooks;
    private const int CodebookVocab = Zonos3Codebook.Vocab;
    private const int Eoa = Zonos3Codebook.Eoa;
    private const int Pad = Zonos3Codebook.Pad;
    private const int Hidden = Zonos3Codebook.EmbedDim;
    private const int PrefillFrames = 3;

    private static readonly int[] SilenceFrame =
        [1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337];

    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly ModelConfig _config;
    private readonly GgufTensorReader _reader;
    private readonly Zonos3Codebook _codebook;
    private readonly ZacCodecDecoder _codec;
    private readonly Dictionary<string, float[]> _voices;
    private readonly bool _ownsModel;
    private bool _disposed;

    public Zonos3TtsRuntime(
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
        if (!config.UsesFullAttnRes)
            throw new NotSupportedException(
                $"ZONOS3 TTS needs a zonos3 model; this model is '{config.Architecture}'.");

        _model = model;
        _tokenizer = tokenizer;
        _config = config;
        _ownsModel = ownsModel;
        _reader = new GgufTensorReader(projectorPath);
        try
        {
            _codebook = new Zonos3Codebook(_reader);
            _codec = new ZacCodecDecoder(_reader);
            _voices = new Dictionary<string, float[]>(StringComparer.Ordinal);
            foreach (string name in _reader.TensorNames)
            {
                if (!name.StartsWith("voice.", StringComparison.Ordinal)) continue;
                _voices[name["voice.".Length..]] = _reader.ReadFloat32(name);
            }
        }
        catch
        {
            _reader.Dispose();
            throw;
        }

        if (_voices.Count > 0 && !_voices.ContainsKey(Voice))
            Voice = _voices.ContainsKey(DefaultVoice)
                ? DefaultVoice
                : _voices.Keys.OrderBy(k => k, StringComparer.Ordinal).First();
    }

    /// <summary>Upper bound on generated frames (one frame is about 67 ms).</summary>
    public int MaxFrames { get; set; } = 512;

    /// <summary>Predefined voice name; the voice row becomes the prompt prefix.</summary>
    public string Voice { get; set; } = DefaultVoice;

    /// <summary>Sampling temperature (the reference uses 0.75).</summary>
    public float Temperature { get; set; } = 0.75f;

    /// <summary>Top-k cutoff (80 in the reference).</summary>
    public int TopK { get; set; } = 80;

    /// <summary>Top-p cutoff (0.69 in the reference).</summary>
    public float TopP { get; set; } = 0.69f;

    /// <summary>Per-codebook repetition penalty over the last frames (2.4 in the reference).</summary>
    public float RepetitionPenalty { get; set; } = 2.4f;

    /// <summary>Frames covered by the repetition penalty (96 in the reference).</summary>
    public int RepetitionWindow { get; set; } = 96;

    /// <summary>Sampling seed; null picks a time-based seed.</summary>
    public int? Seed { get; set; }

    /// <summary>Optional per-frame hook: (frame index, ten sampled codes).</summary>
    public Action<int, int[]>? FrameSampled { get; set; }

    /// <summary>Optional hook invoked when a frame contains the end-of-audio code.</summary>
    public Action<int, int[]>? EndOfSpeech { get; set; }

    /// <summary>Predefined voice names carried by the projector.</summary>
    public IReadOnlyCollection<string> Voices => _voices.Keys;

    /// <summary>
    /// Lists the predefined voices in a ZONOS3 projector, default voice first. Reads only
    /// the GGUF tensor table, so a host can populate a picker without loading the weights.
    /// </summary>
    public static IReadOnlyList<string> ListVoices(string projectorPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectorPath);
        using var reader = new GgufTensorReader(projectorPath);
        return reader.TensorNames
            .Where(name => name.StartsWith("voice.", StringComparison.Ordinal))
            .Select(name => name["voice.".Length..])
            .OrderBy(name => name == DefaultVoice ? 0 : 1)
            .ThenBy(name => name, StringComparer.Ordinal)
            .ToArray();
    }

    /// <summary>True when the projector carries the ZONOS3 codebook head and the ZAC decoder.</summary>
    public static bool IsTtsProjector(string path)
    {
        try
        {
            using var reader = new GgufTensorReader(path);
            return ZacCodecDecoder.IsZacProjector(reader);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Synthesizes <paramref name="text"/> into 44.1 kHz mono samples.</summary>
    public float[] Synthesize(string text, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);
        if (MaxFrames <= 0) return [];
        if (!_voices.TryGetValue(Voice, out float[]? voiceRow))
            throw new InvalidOperationException($"The projector has no voice '{Voice}'.");

        int voiceId = ResolveToken("<|loudness_unk|>");
        int fmaxId = ResolveToken("<|fmax_unk|>");
        int rateId = ResolveToken("<|speaking_rate_unk|>");
        int[] textIds = _tokenizer.Encode(text, addBos: true, addEos: true);
        if (textIds.Length == 0) return [];

        int[] condition = [voiceId, fmaxId, rateId];
        int tokenCount = condition.Length + textIds.Length;
        int rows = 1 + tokenCount + PrefillFrames;

        using var prompt = new Tensor<float>(rows, Hidden);
        voiceRow.AsSpan().CopyTo(prompt.Data);

        var idTensor = new Tensor<int>(1, tokenCount);
        for (int i = 0; i < condition.Length; i++) idTensor.Data[i] = condition[i];
        for (int i = 0; i < textIds.Length; i++) idTensor.Data[condition.Length + i] = textIds[i];
        using (idTensor)
        using (Tensor<float> tokenEmbeddings = _model.ForwardEmbedding(idTensor))
        {
            if (tokenEmbeddings.ElementCount != (long)tokenCount * Hidden)
                throw new InvalidDataException("Unexpected token embedding shape.");
            tokenEmbeddings.Data[..((int)tokenCount * Hidden)]
                .CopyTo(prompt.Data.Slice(Hidden));
        }

        var frame = new int[Codebooks];
        for (int k = 0; k < PrefillFrames; k++)
        {
            Array.Fill(frame, Pad);
            for (int cb = 0; cb <= k; cb++) frame[cb] = SilenceFrame[cb];
            _codebook.EmbedFrame(frame, prompt.Data.Slice((1 + tokenCount + k) * Hidden, Hidden));
        }

        int maxSeq = rows + Math.Max(1, MaxFrames) + 8;
        IKVCache[] caches = CreateCaches(maxSeq);
        try
        {
            float[] hidden = new float[Hidden];
            using (Tensor<float> prefill = _model.ForwardAttnResHidden(prompt, caches, 0))
            {
                prefill.Data.Slice((rows - 1) * Hidden, Hidden).CopyTo(hidden);
            }

            List<int> delayed = new((PrefillFrames + MaxFrames) * Codebooks);
            for (int k = 0; k < PrefillFrames; k++)
            {
                Array.Fill(frame, Pad);
                for (int cb = 0; cb <= k; cb++) frame[cb] = SilenceFrame[cb];
                delayed.AddRange(frame);
            }

            List<int[]> finished = Generate(hidden, rows, caches, delayed, cancellationToken, out int eosRow);
            int ready = delayed.Count / Codebooks - (Codebooks - 1);
            if (eosRow >= 0) ready = Math.Min(ready, eosRow);
            if (ready <= 0) return [];

            var codes = new int[ready * Codebooks];
            for (int f = 0; f < ready; f++)
            {
                for (int cb = 0; cb < Codebooks; cb++)
                {
                    int code = delayed[(f + cb) * Codebooks + cb];
                    codes[f * Codebooks + cb] = Math.Clamp(code, 0, Eoa - 1);
                }
            }

            float[] audio = _codec.Decode(codes, ready);
            return TrimAndLimit(audio);
        }
        finally
        {
            foreach (IKVCache cache in caches)
                if (cache is IDisposable disposable) disposable.Dispose();
        }
    }

    /// <summary>Synthesizes <paramref name="text"/> and writes a 44.1 kHz mono WAV.</summary>
    public float[] SynthesizeToWav(string text, string path, CancellationToken cancellationToken = default)
    {
        float[] audio = Synthesize(text, cancellationToken);
        if (audio.Length > 0) WavFile.Write(path, audio, SampleRate);
        return audio;
    }

    private List<int[]> Generate(
        float[] hidden, int position, IKVCache[] caches, List<int> delayed,
        CancellationToken cancellationToken, out int eosRow)
    {
        var random = new Random(Seed ?? Environment.TickCount);
        var frame = new int[Codebooks];
        var finished = new List<int[]>(Math.Min(MaxFrames, 512));
        int step = 0, countdown = 0;
        eosRow = -1;
        bool done = false;

        while (!done && step < MaxFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();

            float[] logits = _codebook.ComputeLogits(hidden);
            for (int cb = 0; cb < Codebooks; cb++)
                frame[cb] = Sample(logits, cb, delayed, random);

            int r = PrefillFrames + step;
            for (int cb = r + 1; cb < Codebooks; cb++) frame[cb] = Pad;
            delayed.AddRange(frame);
            step++;

            int maxEoa = -1;
            for (int cb = 0; cb < Codebooks; cb++)
                if (frame[cb] == Eoa) maxEoa = cb;
            if (eosRow < 0 && maxEoa >= 0)
            {
                eosRow = Math.Max(0, r - maxEoa);
                countdown = Codebooks + 1;
                EndOfSpeech?.Invoke(eosRow, (int[])frame.Clone());
            }

            var copy = (int[])frame.Clone();
            finished.Add(copy);
            FrameSampled?.Invoke(step - 1, copy);

            if ((countdown > 0 && --countdown == 0) || step >= MaxFrames || position >= _config.MaxSeqLen)
            {
                done = true;
                break;
            }

            var embedding = new Tensor<float>(1, Hidden);
            _codebook.EmbedFrame(frame, embedding.Data);
            using (embedding)
            using (Tensor<float> next = _model.ForwardAttnResHidden(embedding, caches, position))
            {
                next.Data[..Hidden].CopyTo(hidden);
            }
            position++;
        }

        return finished;
    }

    private int Sample(float[] logits, int codebook, List<int> delayed, Random random)
    {
        int offset = codebook * CodebookVocab;
        var scores = new float[CodebookVocab];
        Array.Copy(logits, offset, scores, 0, CodebookVocab);
        scores[Pad] = float.NegativeInfinity;

        if (RepetitionPenalty != 1f && codebook < 6)
        {
            int frames = delayed.Count / Codebooks - PrefillFrames;
            int window = Math.Min(frames, Math.Max(RepetitionWindow, 0));
            var seen = new bool[Eoa];
            for (int f = PrefillFrames + frames - window; f < PrefillFrames + frames; f++)
            {
                int code = delayed[f * Codebooks + codebook];
                if (code >= 0 && code < Eoa && !seen[code])
                {
                    seen[code] = true;
                    scores[code] = scores[code] > 0
                        ? scores[code] / RepetitionPenalty
                        : scores[code] * RepetitionPenalty;
                }
            }
        }

        if (Temperature <= 0f) return ArgMax(scores);

        for (int i = 0; i < scores.Length; i++)
            scores[i] /= Temperature;

        // Top-k: keep the k largest (the reference uses nth_element semantics).
        float kth = float.NegativeInfinity;
        if (TopK > 0 && TopK < CodebookVocab)
        {
            var copy = (float[])scores.Clone();
            Array.Sort(copy);
            kth = copy[CodebookVocab - TopK];
        }

        var indices = new List<int>(CodebookVocab);
        for (int i = 0; i < CodebookVocab; i++)
            if (scores[i] >= kth && scores[i] > float.NegativeInfinity)
                indices.Add(i);
        if (indices.Count == 0) return ArgMax(scores);

        float max = float.NegativeInfinity;
        foreach (int i in indices) max = MathF.Max(max, scores[i]);
        var probabilities = new double[indices.Count];
        double sum = 0;
        for (int j = 0; j < indices.Count; j++)
        {
            double p = Math.Exp(scores[indices[j]] - max);
            probabilities[j] = p;
            sum += p;
        }

        if (TopP > 0f && TopP < 1f)
        {
            var order = Enumerable.Range(0, indices.Count)
                .OrderByDescending(j => probabilities[j])
                .ToArray();
            double before = 0;
            foreach (int j in order)
            {
                double p = probabilities[j];
                if (before > TopP * sum) probabilities[j] = 0;
                before += p;
            }
        }

        double kept = 0;
        foreach (double p in probabilities) kept += p;
        double u = random.NextDouble() * kept;
        for (int j = 0; j < probabilities.Length; j++)
        {
            u -= probabilities[j];
            if (u <= 0 && probabilities[j] > 0) return indices[j];
        }
        return indices[^1];
    }

    private float[] TrimAndLimit(float[] audio)
    {
        int skip = PrefillFrames * SamplesPerFrame;
        if (audio.Length <= skip) return [];
        const float gain = 0.70794578f;
        const float knee = 0.89125094f; // 10^(-1/20)
        var output = new float[audio.Length - skip];
        for (int i = skip; i < audio.Length; i++)
        {
            float x = audio[i] * gain;
            float a = MathF.Abs(x);
            output[i - skip] = a <= knee
                ? x
                : MathF.CopySign(knee + (1f - knee) * MathF.Tanh((a - knee) / (1f - knee)), x);
        }
        return output;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private int ResolveToken(string token)
    {
        int id = _tokenizer.TokenToId(token);
        if (id < 0)
            throw new InvalidOperationException($"The tokenizer has no '{token}' token.");
        return id;
    }

    private IKVCache[] CreateCaches(int maxSeq)
    {
        var builder = new KVCacherBuilder();
        var caches = new IKVCache[_config.NumLayers];
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            caches[layer] = _config.IsRecurrentLayer(layer)
                ? new DeltaNetCache(_config.SsmConvKernel, _config.SsmConvDim,
                    _config.SsmTimeStepRank, _config.SsmStateSize,
                    _config.SsmValueHeadDim, 1, maxSeq)
                : builder.CreateKVCache(1, _config.NumKvHeads, maxSeq, _config.HeadDimForLayer(layer));
        }
        return caches;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _reader.Dispose();
        _codec.Dispose();
        if (_ownsModel) _model.Dispose();
    }
}
